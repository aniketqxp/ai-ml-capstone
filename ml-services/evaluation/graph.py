"""
Graph-based call evaluation pipeline (v0.4.1).

Decomposes the monolithic extract.py into five parallel LLM nodes
(compliance, quality, escalation, chapters, workflow) orchestrated by
LangGraph. Each node gets a focused prompt, validates its own output
fragment, and retries independently on failure. After aggregation, a
deterministic fuse step blends the audio sentiment model's output into the
audio-informed scores; when the text and acoustic escalation tiers DISAGREE
the graph summons an arbitration LLM node (v0.4.1) instead of merging by
rule. Anchoring + runtime routing (evidence re-runs, conditional
investigation) close the loop.

Graph shape:
    assemble → [compliance, quality, escalation, chapters, workflow]
             → aggregate → fuse → (arbitrate?) → anchor
             → (re-run | investigate | END)

Usage:
  python graph.py --call_id en_CA_Banking_1592237
  python graph.py --call_id en_CA_Banking_1592237 --results_dir results_channels
"""
import os
import json
import time
import argparse
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END


def _merge_meta(a, b):
    """Reducer: parallel nodes each contribute their own node_meta entry."""
    return {**(a or {}), **(b or {})}

from nodes import (
    assemble_node,
    compliance_node,
    quality_node,
    escalation_node,
    chapter_node,
    workflow_node,
    aggregate_node,
    fuse_node,
    arbitrate_node,
    anchor_node,
    investigate_node,
)
from rubric import RUBRIC_VERSION_GRAPH


# ─────────────────────────────────────────────────────────────────────────────
# State schema — keys are populated incrementally by each node
# ─────────────────────────────────────────────────────────────────────────────

class EvalState(TypedDict, total=False):
    # inputs
    call_id: str
    results_dir: str

    # set by assemble
    packet: str
    meta: dict
    turns: list

    # set by parallel LLM nodes
    compliance: object
    quality: object
    escalation: object
    chapters: dict
    workflow: dict

    # set by aggregate + anchor
    evaluation: dict

    # v0.3.0 control fields
    anchor_attempts: int      # anchor passes so far (loop guard)
    anchor_feedback: dict     # {section: [unanchored quotes]} from last pass

    # v0.4.1: set by fuse — text vs acoustic escalation tiers disagree
    fusion_disputed: bool

    # per-node execution metadata (duration, served model, run count);
    # parallel nodes merge via reducer instead of clobbering each other
    node_meta: Annotated[dict, _merge_meta]


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

# which evaluation section is produced by which LLM node
SECTION_NODE = {"compliance": "compliance",
                "quality": "quality",
                "escalation": "escalation"}

MAX_ANCHOR_PASSES = 2  # first pass + one corrective re-run


def route_after_fuse(state):
    """
    v0.4.1 routing. Deterministic merging is only defensible when the text
    and acoustic escalation tiers AGREE (they carry no information to break
    their own tie). On disagreement, summon the arbitration LLM node.
    """
    if state.get("fusion_disputed"):
        h = state["evaluation"]["escalation"].get("hybrid") or {}
        print(f"  [router] escalation tiers disagree "
              f"(text={h.get('text_risk')}, acoustic={h.get('acoustic_risk')})"
              f" -> arbitration")
        return "arbitrate"
    return "anchor"


def route_after_anchor(state):
    """
    v0.3.0 runtime routing. After anchoring, decide:
      1. Any section cited quotes that don't exist in the transcript, and we
         still have a corrective pass left? -> re-run ONLY those LLM nodes
         with the failed quotes as a revision note.
      2. Otherwise, does the call carry escalation risk? -> deep-dive
         investigation node.
      3. Otherwise -> done.
    """
    feedback = state.get("anchor_feedback") or {}
    if feedback and state.get("anchor_attempts", 0) < MAX_ANCHOR_PASSES:
        targets = [SECTION_NODE[s] for s in feedback if s in SECTION_NODE]
        if targets:
            print(f"  [router] unverified evidence -> re-running: "
                  f"{', '.join(targets)}")
            return targets

    risk = state["evaluation"]["escalation"]["risk_level"]
    if risk != "none":
        print(f"  [router] risk_level={risk} -> investigation")
        return "investigate"
    return END


def build_graph():
    g = StateGraph(EvalState)

    g.add_node("assemble", assemble_node)
    g.add_node("compliance", compliance_node)
    g.add_node("quality", quality_node)
    g.add_node("escalation", escalation_node)
    g.add_node("chapters", chapter_node)
    g.add_node("workflow", workflow_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("fuse", fuse_node)
    g.add_node("arbitrate", arbitrate_node)
    g.add_node("anchor", anchor_node)
    g.add_node("investigate", investigate_node)

    # assemble fans out to 5 parallel nodes
    g.add_edge(START, "assemble")
    g.add_edge("assemble", "compliance")
    g.add_edge("assemble", "quality")
    g.add_edge("assemble", "escalation")
    g.add_edge("assemble", "chapters")
    g.add_edge("assemble", "workflow")

    # all 5 converge into aggregate
    g.add_edge("compliance", "aggregate")
    g.add_edge("quality", "aggregate")
    g.add_edge("escalation", "aggregate")
    g.add_edge("chapters", "aggregate")
    g.add_edge("workflow", "aggregate")

    # aggregate → fuse (acoustic-text fusion), then RUNTIME routing:
    #   text/acoustic tiers disagree → arbitrate (LLM weighs both in context;
    #     agreement merges deterministically and skips this — v0.4.1)
    #   unverified evidence  → back to the offending LLM node(s) (once)
    #   escalation risk      → investigation deep-dive
    #     (fuse/arbitrate run BEFORE routing, so an acoustically-detected
    #      risk can summon the investigation even when the words look calm)
    #   clean + calm         → END
    g.add_edge("aggregate", "fuse")
    g.add_conditional_edges(
        "fuse", route_after_fuse, ["arbitrate", "anchor"])
    g.add_edge("arbitrate", "anchor")
    g.add_conditional_edges(
        "anchor", route_after_anchor,
        ["compliance", "quality", "escalation", "investigate", END])
    g.add_edge("investigate", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def print_graph_report(ev, dt):
    meta = ev.get("metadata", {})
    print("\n" + "=" * 68)
    print(f"  GRAPH EVALUATION  {meta.get('call_id', '?')}  "
          f"({meta.get('domain', '?')})  [{dt:.1f}s total]")
    print(f"  rubric {ev.get('rubric_version', '?')}")
    print("=" * 68)

    # compliance
    print("\n  COMPLIANCE")
    c = ev.get("compliance", {})
    for key, name in [("name_announced", "Name announced"),
                      ("company_announced", "Company announced"),
                      ("recording_disclosure", "Recording disclosure"),
                      ("identity_verified", "Identity verified"),
                      ("resolution_provided", "Resolution provided"),
                      ("transfer_next_steps", "Transfer next-steps")]:
        item = c.get(key, {})
        mark = {True: "PASS", False: "FAIL", None: "N/A "}[item.get("passed")]
        evd = item.get("evidence")
        when = (f" @{_mmss(evd['sec'])}" if evd and evd.get("sec") is not None
                else "")
        quote = (f'  "{evd["quote"][:50]}"' if evd
                 else (f"  ({item.get('note')})" if item.get("note") else ""))
        print(f"    [{mark}]{when:>7} {name:<22}{quote}")
    if c.get("identity_method"):
        print(f"              method: {c['identity_method']}")

    # quality
    print("\n  QUALITY (1-5, hybrid text+acoustic)")
    q = ev.get("quality", {})
    dims = [("efficiency", "Efficiency"),
            ("problem_resolution", "Problem resolution"),
            ("clarity", "Clarity"),
            ("professionalism", "Professionalism"),
            ("empathy", "Empathy"),
            ("customer_satisfaction", "Customer satisfaction")]
    for key, name in dims:
        d = q.get(key)
        if d is None:
            continue
        h = d.get("hybrid") or {}
        if h.get("method") == "weighted_mean":
            how = (f" (text {h['text_score']} x {h['text_weight']} + "
                   f"audio {h['acoustic_score']} x {h['acoustic_weight']}, "
                   f"{h['channel'].lower()} ch)")
        else:
            how = " (text only)"
        print(f"    {d['score']}/5  {name:<24}{how}")
        if d.get("signals_absent"):
            print(f"          missing: {', '.join(d['signals_absent'][:3])}")

    # escalation
    e = ev.get("escalation", {})
    flags = e.get("red_flags", [])
    eh = e.get("hybrid") or {}
    fused = (f"  [text: {eh['text_risk']} | acoustic: {eh['acoustic_risk']} "
             f"(late-mean {eh['late_mean_escalation']}, peak {eh['peak_escalation']})"
             f" via {eh['method']}]"
             if eh.get("acoustic_risk") is not None else "")
    print(f"\n  ESCALATION: {e.get('risk_level', '?').upper()}{fused}  "
          f"| emotion(text): {e.get('customer_emotion_text', '?')}"
          f" | flags: {', '.join(flags) if flags else 'none'}")
    if eh.get("arbitration_rationale"):
        print(f"    arbitrator: {eh['arbitration_rationale']}")

    # workflow (v0.4.0)
    wf = ev.get("workflow")
    if wf:
        print(f"\n  EXPECTED WORKFLOW  (subject: {wf.get('subject', '?')})")
        for s in wf.get("expected_steps", []):
            mark = {True: "[x]", False: "[ ]", None: "[-]"}[s.get("met")]
            evd = s.get("evidence")
            when = (f" @{_mmss(evd['sec'])}" if evd and evd.get("sec") is not None
                    else "")
            print(f"    {mark}{when:>7} {s['step']}")

    # investigation (v0.3.0 — only when risk_level != none)
    inv = ev.get("investigation")
    if inv:
        print(f"\n  INVESTIGATION [{inv.get('priority', '?').upper()} priority]")
        print(f"    {inv.get('summary', '')}")
        for f_ in inv.get("contributing_factors", []):
            print(f"    - {f_}")
        print(f"    ACTION: {inv.get('recommended_action', '')}")

    # summary
    summary = ev.get("overall_summary")
    if summary:
        print(f"\n  SUMMARY: {summary}")

    # anchor stats
    astats = ev.get("_anchor_stats", {})
    if astats:
        print(f"\n  EVIDENCE: {astats.get('anchored', 0)}/{astats.get('total', 0)} "
              f"quotes anchored to transcript timestamps")

    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Graph-based call evaluation (parallel LLM nodes)")
    ap.add_argument("--call_id", required=True)
    ap.add_argument("--results_dir", default="results_channels")
    args = ap.parse_args()

    graph = build_graph()

    print(f"\nEvaluating {args.call_id} via graph pipeline "
          f"(rubric {RUBRIC_VERSION_GRAPH})...")
    print("-" * 68)

    t0 = time.time()
    final = graph.invoke({
        "call_id": args.call_id,
        "results_dir": args.results_dir,
    })
    dt = time.time() - t0

    ev = final["evaluation"]

    # execution trace for the frontend pipeline strip
    ev["_pipeline"] = {
        "nodes": final.get("node_meta", {}),
        "anchor_passes": final.get("anchor_attempts", 1),
        "wall_clock": round(dt, 1),
    }

    # save
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.call_id}_graph.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ev, f, indent=2)

    print_graph_report(ev, dt)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
