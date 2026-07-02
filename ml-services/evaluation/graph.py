"""
Graph-based call evaluation pipeline (v0.2.0).

Decomposes the monolithic extract.py into four parallel LLM nodes
(compliance, quality, escalation, chapters) orchestrated by LangGraph.
Each node gets a focused prompt with enriched signal guidance, validates
its own output fragment, and retries independently on failure.

Graph shape:
    assemble → [compliance, quality, escalation, chapters] → aggregate → anchor

Usage:
  python graph.py --call_id en_CA_Banking_1592237
  python graph.py --call_id en_CA_Banking_1592237 --results_dir results_channels
"""
import os
import json
import time
import argparse
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from nodes import (
    assemble_node,
    compliance_node,
    quality_node,
    escalation_node,
    chapter_node,
    aggregate_node,
    anchor_node,
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

    # set by aggregate + anchor
    evaluation: dict


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(EvalState)

    g.add_node("assemble", assemble_node)
    g.add_node("compliance", compliance_node)
    g.add_node("quality", quality_node)
    g.add_node("escalation", escalation_node)
    g.add_node("chapters", chapter_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("anchor", anchor_node)

    # assemble fans out to 4 parallel nodes
    g.add_edge(START, "assemble")
    g.add_edge("assemble", "compliance")
    g.add_edge("assemble", "quality")
    g.add_edge("assemble", "escalation")
    g.add_edge("assemble", "chapters")

    # all 4 converge into aggregate
    g.add_edge("compliance", "aggregate")
    g.add_edge("quality", "aggregate")
    g.add_edge("escalation", "aggregate")
    g.add_edge("chapters", "aggregate")

    # aggregate → anchor → done
    g.add_edge("aggregate", "anchor")
    g.add_edge("anchor", END)

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
    print("\n  QUALITY (1-5, text-derived)")
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
        aud = " (+audio needed)" if d.get("requires_audio") else ""
        print(f"    {d['score']}/5  {name:<24}{aud}")
        if d.get("signals_absent"):
            print(f"          missing: {', '.join(d['signals_absent'][:3])}")

    # escalation
    e = ev.get("escalation", {})
    flags = e.get("red_flags", [])
    print(f"\n  ESCALATION: {e.get('risk_level', '?').upper()}  "
          f"| emotion(text): {e.get('customer_emotion_text', '?')}"
          f" | flags: {', '.join(flags) if flags else 'none'}")

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
