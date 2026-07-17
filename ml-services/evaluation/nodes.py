"""
Node functions for the graph-based evaluation pipeline.

Each node is a callable that takes EvalState and returns a partial state dict.
LLM nodes use a shared retry helper; deterministic nodes (assemble, aggregate,
anchor) do pure data transformation.
"""
import os
import json
import time

from pydantic import ValidationError

from env_util import load_env
from assemble import load_manifest, build_packet, assemble_turns, estimate_duration
from pydantic import BaseModel
from typing import List, Literal

from rubric import (
    ComplianceChecklist, QualityDimensions, EscalationRisk,
    InvestigationReport, CallMetadata, CallEvaluation, RUBRIC_VERSION_GRAPH,
    CallWorkflow, WorkflowStep, Evidence,
)
from prompts import (
    COMPLIANCE_SYSTEM, COMPLIANCE_SKELETON,
    QUALITY_SYSTEM, QUALITY_SKELETON,
    ESCALATION_SYSTEM, ESCALATION_SKELETON,
    INVESTIGATE_SYSTEM, INVESTIGATE_SKELETON,
    ARBITRATE_SYSTEM, ARBITRATE_SKELETON,
    SUBJECT_SYSTEM, SUBJECT_SKELETON,
    WORKFLOW_GEN_SYSTEM, WORKFLOW_GEN_SKELETON,
    WORKFLOW_CHECK_SYSTEM, WORKFLOW_CHECK_SKELETON,
    WORKFLOW_RECHECK_SYSTEM, WORKFLOW_RECHECK_SKELETON,
)

load_env()
DATA = r"d:\Desktop\Main\Projects\ai-ml-capstone\data\na_testset"


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM call + retry helper
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s[3:]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _feedback_text(quotes):
    """Build a revision note for a node whose cited quotes failed anchoring."""
    if not quotes:
        return None
    listed = "\n".join(f'- "{q}"' for q in quotes)
    return ("REVISION NOTE: in a previous evaluation of this call, the "
            "following cited quotes were NOT found verbatim in the "
            f"transcript:\n{listed}\n"
            "Re-evaluate. Every quote must be copied EXACTLY, character for "
            "character, from a single transcript turn. Do not paraphrase, "
            "do not merge text from multiple turns.")


def _llm_eval_with_retry(system, skeleton, packet, validator, max_attempts=2,
                         feedback=None):
    """
    Call the LLM router, parse JSON, validate with `validator`, retry on
    failure with error feedback. Returns (validated_model, dt, served_model).
    `feedback` is an optional revision note (e.g. failed anchor quotes)
    appended to the user prompt.
    """
    from router import chat_json_routed

    user = f"{packet}\n\n{skeleton}"
    if feedback:
        user += f"\n\n{feedback}"
    last_err = None

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        raw, served = chat_json_routed(system, user, return_meta=True)
        dt = time.time() - t0
        try:
            data = json.loads(_strip_fences(raw))
            result = validator(data)
            return result, dt, served
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            print(f"  [{system[:40]}...] attempt {attempt} failed "
                  f"({type(e).__name__}); retrying...")
            user = (f"{packet}\n\n{skeleton}"
                    + (f"\n\n{feedback}" if feedback else "")
                    + f"\n\nYour previous output was invalid: "
                    f"{str(e)[:400]}. Return corrected JSON only.")

    raise RuntimeError(f"Node failed after {max_attempts} attempts: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic nodes
# ─────────────────────────────────────────────────────────────────────────────

def _runs(state, node):
    """How many times `node` has executed in this invocation (1-based)."""
    return ((state.get("node_meta") or {}).get(node, {}).get("runs", 0)) + 1


def assemble_node(state):
    """Load transcript, build packet and turns for downstream nodes."""
    t0 = time.time()
    call_id = state["call_id"]
    results_dir = state.get("results_dir", "results_channels")
    manifest = load_manifest()
    meta = manifest[call_id]

    result_path = os.path.join(DATA, results_dir, meta["accent"],
                               call_id + ".json")
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    packet, stats = build_packet(result, meta)
    turns = assemble_turns(result)

    cm = CallMetadata(
        call_id=call_id,
        domain=result.get("domain", meta.get("domain", "unknown")),
        accent=meta.get("accent"),
        duration_seconds=stats["duration_s"],
        transcript_model=result.get("model"),
    )

    return {
        "packet": packet,
        "meta": cm.model_dump(),
        "turns": turns,
        "node_meta": {"assemble": {"duration": round(time.time() - t0, 2)}},
    }


def aggregate_node(state):
    """Merge parallel node outputs into a single CallEvaluation."""
    ev = CallEvaluation(
        rubric_version=RUBRIC_VERSION_GRAPH,
        metadata=CallMetadata(**state["meta"]),
        compliance=state["compliance"],
        quality=state["quality"],
        escalation=state["escalation"],
        workflow=state.get("workflow"),
        overall_summary=None,
    )
    return {"evaluation": ev.model_dump()}


def _collect_unanchored(node, out):
    """Recursively collect evidence quotes that anchoring could not locate."""
    if isinstance(node, dict):
        if "quote" in node and "speaker" in node and node.get("sec") is None:
            out.append(node["quote"])
        else:
            for v in node.values():
                _collect_unanchored(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_unanchored(v, out)


def anchor_node(state):
    """
    Attach real transcript timestamps to every evidence quote.
    Also reports WHICH sections cited quotes that could not be located, so
    the graph can route those sections back for re-evaluation (v0.3.0).
    """
    from eval_call_data import anchor_evidence

    ev_dict = state["evaluation"]
    if not isinstance(ev_dict, dict):
        ev_dict = ev_dict.model_dump()

    turns_indexed = [{"i": i, **t} for i, t in enumerate(state["turns"])]
    stats = {"total": 0, "anchored": 0}
    anchor_evidence(ev_dict, turns_indexed, stats)
    ev_dict["_anchor_stats"] = stats

    # per-section unanchored quotes -> feedback for the re-anchor loop
    feedback = {}
    for section in ("compliance", "quality", "escalation"):
        missing = []
        _collect_unanchored(ev_dict.get(section, {}), missing)
        if missing:
            feedback[section] = missing

    attempts = state.get("anchor_attempts", 0) + 1
    if feedback:
        n = sum(len(v) for v in feedback.values())
        print(f"  [anchor] pass {attempts}: {n} quote(s) not found verbatim "
              f"in sections: {', '.join(feedback)}")
    else:
        print(f"  [anchor] pass {attempts}: all quotes anchored")

    return {
        "evaluation": ev_dict,
        "anchor_attempts": attempts,
        "anchor_feedback": feedback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM nodes — each validates its own Pydantic fragment
# ─────────────────────────────────────────────────────────────────────────────

def _section_feedback(state, section):
    """Revision note if this section's quotes failed anchoring last pass."""
    fb = (state.get("anchor_feedback") or {}).get(section)
    if fb:
        print(f"  [{section}] re-running with {len(fb)} failed-quote note(s)")
    return _feedback_text(fb)


def compliance_node(state):
    """Evaluate 6 compliance checks."""
    result, dt, served = _llm_eval_with_retry(
        system=COMPLIANCE_SYSTEM,
        skeleton=COMPLIANCE_SKELETON,
        packet=state["packet"],
        validator=lambda d: ComplianceChecklist.model_validate(d),
        feedback=_section_feedback(state, "compliance"),
    )
    print(f"  [compliance] done in {dt:.1f}s via {served}")
    return {"compliance": result,
            "node_meta": {"compliance": {"duration": round(dt, 1),
                                         "served_by": served,
                                         "runs": _runs(state, "compliance")}}}


def quality_node(state):
    """Score 6 quality dimensions with enriched signal guidance."""
    result, dt, served = _llm_eval_with_retry(
        system=QUALITY_SYSTEM,
        skeleton=QUALITY_SKELETON,
        packet=state["packet"],
        validator=lambda d: QualityDimensions.model_validate(d),
        feedback=_section_feedback(state, "quality"),
    )
    print(f"  [quality] done in {dt:.1f}s via {served}")
    return {"quality": result,
            "node_meta": {"quality": {"duration": round(dt, 1),
                                      "served_by": served,
                                      "runs": _runs(state, "quality")}}}


def escalation_node(state):
    """Assess escalation risk: red flags, emotion, risk level."""
    result, dt, served = _llm_eval_with_retry(
        system=ESCALATION_SYSTEM,
        skeleton=ESCALATION_SKELETON,
        packet=state["packet"],
        validator=lambda d: EscalationRisk.model_validate(d),
        feedback=_section_feedback(state, "escalation"),
    )
    print(f"  [escalation] done in {dt:.1f}s via {served}")
    return {"escalation": result,
            "node_meta": {"escalation": {"duration": round(dt, 1),
                                         "served_by": served,
                                         "runs": _runs(state, "escalation")}}}


def investigate_node(state):
    """
    Conditional deep-dive (v0.3.0): runs ONLY when the anchored evaluation
    carries risk_level != "none". Produces a manager-ready incident report
    and attaches it to the evaluation.
    """
    ev = state["evaluation"]
    esc = ev["escalation"]
    context = (
        f"{state['packet']}\n\n"
        f"FIRST-PASS ESCALATION SCREENING RESULT:\n"
        f"{json.dumps({k: esc[k] for k in ('red_flags', 'customer_emotion_text', 'risk_level')}, indent=2)}"
    )
    report, dt, served = _llm_eval_with_retry(
        system=INVESTIGATE_SYSTEM,
        skeleton=INVESTIGATE_SKELETON,
        packet=context,
        validator=lambda d: InvestigationReport.model_validate(d),
    )
    print(f"  [investigate] done in {dt:.1f}s via {served} "
          f"(priority: {report.priority})")

    # anchor the report's own evidence quotes too
    from eval_call_data import anchor_evidence
    report_dict = report.model_dump()
    turns_indexed = [{"i": i, **t} for i, t in enumerate(state["turns"])]
    anchor_evidence(report_dict, turns_indexed, {"total": 0, "anchored": 0})

    ev["investigation"] = report_dict
    return {"evaluation": ev,
            "node_meta": {"investigate": {"duration": round(dt, 1),
                                          "served_by": served,
                                          "priority": report.priority}}}


# ── workflow track (v0.4.0) — three isolated LLM phases in one node ─────────

class _Subject(BaseModel):
    subject: str


class _StepsDraft(BaseModel):
    expected_steps: List[WorkflowStep]


def workflow_node(state):
    """
    Subject-derived expected workflow (v0.4.0). Three phases with strict
    prompt isolation:
      a. subject   — reads the transcript, states the REQUEST in one sentence
      b. expected  — sees ONLY domain + subject (never the transcript), lists
                     the boxes a competent agent should check for such a call
      c. check     — audits the transcript against that independent checklist
    The isolation in (b) is the point: expectations are formed before knowing
    what actually happened, so the audit cannot rationalize the observed call.
    """
    domain = state["meta"]["domain"]

    subj, dt_a, served_a = _llm_eval_with_retry(
        system=SUBJECT_SYSTEM, skeleton=SUBJECT_SKELETON,
        packet=state["packet"],
        validator=lambda d: _Subject.model_validate(d))
    print(f'  [workflow] subject in {dt_a:.1f}s: "{subj.subject}"')

    # phase b: NO transcript — only domain + subject
    gen_packet = f"DOMAIN: {domain}\nCALL SUBJECT: {subj.subject}"
    draft, dt_b, served_b = _llm_eval_with_retry(
        system=WORKFLOW_GEN_SYSTEM, skeleton=WORKFLOW_GEN_SKELETON,
        packet=gen_packet,
        validator=lambda d: _StepsDraft.model_validate(d))
    print(f"  [workflow] {len(draft.expected_steps)} expected steps in {dt_b:.1f}s")

    checklist = json.dumps(
        {"expected_steps": [{"step": s.step, "rationale": s.rationale}
                            for s in draft.expected_steps]}, indent=2)
    check_packet = (f"EXPECTED WORKFLOW (written without seeing this call):\n"
                    f"{checklist}\n\nTRANSCRIPT:\n{state['packet']}")
    # completeness is part of validity: a truncated audit (fewer entries than
    # the draft) would otherwise be silently padded with met=null at the join
    def _check_validator(d):
        r = _StepsDraft.model_validate(d)
        if len(r.expected_steps) != len(draft.expected_steps):
            raise ValueError(
                f"audit returned {len(r.expected_steps)} entries for "
                f"{len(draft.expected_steps)} steps; return one entry per "
                f"given step, same order")
        return r

    checked, dt_c, served_c = _llm_eval_with_retry(
        system=WORKFLOW_CHECK_SYSTEM, skeleton=WORKFLOW_CHECK_SKELETON,
        packet=check_packet, validator=_check_validator)

    # join by index against the DRAFT's step text — the draft is the contract;
    # the checker only contributes met/evidence. Missing entries stay unchecked.
    steps = []
    for i, ds in enumerate(draft.expected_steps):
        cs = checked.expected_steps[i] if i < len(checked.expected_steps) else None
        steps.append(WorkflowStep(
            step=ds.step, rationale=ds.rationale,
            met=cs.met if cs else None,
            evidence=cs.evidence if cs else None))

    # single-item recheck pass: fires ONLY on met=false steps, isolated from
    # the other steps so it can never perturb something the main pass already
    # got right. Catches the specific failure mode where the main pass wants
    # a direct question and misses that the value was established via a
    # later read-back/confirmation instead.
    def _recheck_validator(d):
        if "met" not in d:
            raise ValueError("missing met")
        return d

    dt_d = 0.0
    for s in steps:
        if s.met is not False:
            continue
        recheck_packet = (
            f"CHECKLIST ITEM: {s.step}\n"
            f"RATIONALE: {s.rationale}\n"
            f"FIRST-PASS VERDICT: met=false\n\n"
            f"TRANSCRIPT:\n{state['packet']}")
        try:
            rc, dt_r, _ = _llm_eval_with_retry(
                system=WORKFLOW_RECHECK_SYSTEM, skeleton=WORKFLOW_RECHECK_SKELETON,
                packet=recheck_packet, validator=_recheck_validator)
        except RuntimeError:
            continue
        dt_d += dt_r
        if rc.get("met") is True and rc.get("evidence"):
            s.met = True
            s.evidence = Evidence.model_validate(rc["evidence"])

    met = sum(1 for s in steps if s.met is True)
    missed = sum(1 for s in steps if s.met is False)
    print(f"  [workflow] audit in {dt_c:.1f}s: {met} met, {missed} missed, "
          f"{len(steps) - met - missed} n/a of {len(steps)}"
          + (f"  (+{dt_d:.1f}s recheck)" if dt_d else ""))

    dt = dt_a + dt_b + dt_c + dt_d
    return {"workflow": CallWorkflow(subject=subj.subject,
                                     expected_steps=steps).model_dump(),
            "node_meta": {"workflow": {"duration": round(dt, 1),
                                       "served_by": served_c,
                                       "phases": {"subject": round(dt_a, 1),
                                                  "expected": round(dt_b, 1),
                                                  "check": round(dt_c, 1)},
                                       "runs": _runs(state, "workflow")}}}


def fuse_node(state):
    """
    Deterministic acoustic-text fusion (v0.4.0). Loads the audio sentiment
    model's output (if any) and fuses it into the three audio-informed
    quality dimensions + the escalation risk level, attaching provenance.
    Runs between aggregate and anchor so the fused risk drives routing.
    """
    from fusion import load_acoustic, fuse_evaluation
    t0 = time.time()
    ev = state["evaluation"]
    rows = load_acoustic(state["call_id"], state["meta"]["domain"])
    summary = fuse_evaluation(ev, rows)
    if summary["acoustic_available"]:
        print(f"  [fuse] acoustic data found: fused "
              f"{', '.join(summary['fused_dims']) or 'nothing'}; "
              f"acoustic risk = {summary['acoustic_risk']}")
    else:
        print("  [fuse] no acoustic data for this call -> text-only scores")
    return {"evaluation": ev,
            "fusion_disputed": summary["disputed"],
            "node_meta": {"fuse": {"duration": round(time.time() - t0, 2),
                                   **summary}}}


class _Arbitration(BaseModel):
    risk_level: Literal["none", "review", "escalate"]
    rationale: str


def arbitrate_node(state):
    """
    Conditional arbitration (v0.4.1): runs ONLY when the text tier and the
    acoustic tier disagree about escalation risk. Deterministic merging is
    fine when the channels agree; when they disagree, a fixed rule has no
    basis to prefer one -- an LLM weighs both signals against the transcript
    (was the issue resolved? does vocal escalation RISE late in the call?).
    """
    ev = state["evaluation"]
    esc = ev["escalation"]
    h = esc.get("hybrid") or {}
    traj = (ev.get("_acoustic") or {}).get("trajectory") or []
    # compact trajectory line: customer escalation score per position bucket
    esc_series = " ".join(f"{t['p']:.2f}:{t['esc']:.2f}"
                          for t in traj if t.get("esc") is not None)

    context = (
        f"{state['packet']}\n\n"
        f"TEXT ASSESSMENT:\n"
        f"{json.dumps({k: esc[k] for k in ('red_flags', 'customer_emotion_text', 'risk_level')}, indent=2)}\n\n"
        f"ACOUSTIC ASSESSMENT (audio model, customer channel):\n"
        f"  tier: {h.get('acoustic_risk')}\n"
        f"  mean escalation over final third: {h.get('late_mean_escalation')}\n"
        f"  peak escalation: {h.get('peak_escalation')}\n"
        f"  trajectory (position:score, 0=start 1=end): {esc_series or 'n/a'}"
    )

    verdict, dt, served = _llm_eval_with_retry(
        system=ARBITRATE_SYSTEM, skeleton=ARBITRATE_SKELETON,
        packet=context,
        validator=lambda d: _Arbitration.model_validate(d))

    print(f"  [arbitrate] text={h.get('text_risk')} vs "
          f"acoustic={h.get('acoustic_risk')} -> {verdict.risk_level} "
          f"in {dt:.1f}s via {served}")

    esc["risk_level"] = verdict.risk_level
    esc["hybrid"] = {**h, "method": "llm_arbitration",
                     "arbitration_rationale": verdict.rationale}
    return {"evaluation": ev,
            "node_meta": {"arbitrate": {"duration": round(dt, 1),
                                        "served_by": served,
                                        "verdict": verdict.risk_level}}}


def chapter_node(state):
    """Segment the call into coherent chapters (reuses segment.py)."""
    from segment import segment
    cc, dt, issues = segment(
        state["call_id"],
        provider="router",
        results_dir=state.get("results_dir", "results_channels"),
    )
    if issues:
        print(f"  [chapters] {len(issues)} invariant issue(s): {issues}")
    print(f"  [chapters] {cc.n_chapters} chapters in {dt:.1f}s")
    return {"chapters": cc.model_dump(),
            "node_meta": {"chapters": {"duration": round(dt, 1),
                                       "served_by": "router",
                                       "runs": _runs(state, "chapters")}}}
