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
from rubric import (
    ComplianceChecklist, QualityDimensions, EscalationRisk,
    CallMetadata, CallEvaluation, RUBRIC_VERSION_GRAPH,
)
from prompts import (
    COMPLIANCE_SYSTEM, COMPLIANCE_SKELETON,
    QUALITY_SYSTEM, QUALITY_SKELETON,
    ESCALATION_SYSTEM, ESCALATION_SKELETON,
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


def _llm_eval_with_retry(system, skeleton, packet, validator, max_attempts=2):
    """
    Call the LLM router, parse JSON, validate with `validator`, retry on
    failure with error feedback. Returns (validated_model, dt, served_model).
    """
    from router import chat_json_routed

    user = f"{packet}\n\n{skeleton}"
    last_err = None

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        raw, served = chat_json_routed(system, user, return_meta=True)
        dt = time.time() - t0
        try:
            data = json.loads(_strip_fences(raw))
            result = validator(data)
            return result, dt, served
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            print(f"  [{system[:40]}...] attempt {attempt} failed "
                  f"({type(e).__name__}); retrying...")
            user = (f"{packet}\n\n{skeleton}\n\nYour previous output was "
                    f"invalid: {str(e)[:400]}. Return corrected JSON only.")

    raise RuntimeError(f"Node failed after {max_attempts} attempts: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic nodes
# ─────────────────────────────────────────────────────────────────────────────

def assemble_node(state):
    """Load transcript, build packet and turns for downstream nodes."""
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
    }


def aggregate_node(state):
    """Merge parallel node outputs into a single CallEvaluation."""
    ev = CallEvaluation(
        rubric_version=RUBRIC_VERSION_GRAPH,
        metadata=CallMetadata(**state["meta"]),
        compliance=state["compliance"],
        quality=state["quality"],
        escalation=state["escalation"],
        overall_summary=None,
    )
    return {"evaluation": ev.model_dump()}


def anchor_node(state):
    """Attach real transcript timestamps to every evidence quote."""
    from eval_call_data import anchor_evidence

    ev_dict = state["evaluation"]
    if not isinstance(ev_dict, dict):
        ev_dict = ev_dict.model_dump()

    turns_indexed = [{"i": i, **t} for i, t in enumerate(state["turns"])]
    stats = {"total": 0, "anchored": 0}
    anchor_evidence(ev_dict, turns_indexed, stats)
    ev_dict["_anchor_stats"] = stats

    return {"evaluation": ev_dict}


# ─────────────────────────────────────────────────────────────────────────────
# LLM nodes — each validates its own Pydantic fragment
# ─────────────────────────────────────────────────────────────────────────────

def compliance_node(state):
    """Evaluate 6 compliance checks."""
    result, dt, served = _llm_eval_with_retry(
        system=COMPLIANCE_SYSTEM,
        skeleton=COMPLIANCE_SKELETON,
        packet=state["packet"],
        validator=lambda d: ComplianceChecklist.model_validate(d),
    )
    print(f"  [compliance] done in {dt:.1f}s via {served}")
    return {"compliance": result}


def quality_node(state):
    """Score 6 quality dimensions with enriched signal guidance."""
    result, dt, served = _llm_eval_with_retry(
        system=QUALITY_SYSTEM,
        skeleton=QUALITY_SKELETON,
        packet=state["packet"],
        validator=lambda d: QualityDimensions.model_validate(d),
    )
    print(f"  [quality] done in {dt:.1f}s via {served}")
    return {"quality": result}


def escalation_node(state):
    """Assess escalation risk: red flags, emotion, risk level."""
    result, dt, served = _llm_eval_with_retry(
        system=ESCALATION_SYSTEM,
        skeleton=ESCALATION_SKELETON,
        packet=state["packet"],
        validator=lambda d: EscalationRisk.model_validate(d),
    )
    print(f"  [escalation] done in {dt:.1f}s via {served}")
    return {"escalation": result}


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
    return {"chapters": cc.model_dump()}
