"""Create or verify a compact snapshot of existing graph evaluation results."""
import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / "results" / "batch_summary.json"
DEFAULT_OUTPUT = HERE / "baselines" / "v1.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _timing_summary(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": round(values[0], 3),
        "median": round(statistics.median(values), 3),
        "max": round(values[-1], 3),
    }


def _quality_scores(evaluation):
    return {
        name: value.get("score")
        for name, value in (evaluation.get("quality") or {}).items()
        if isinstance(value, dict) and "score" in value
    }


def _compliance_states(evaluation):
    return {
        name: value.get("passed")
        for name, value in (evaluation.get("compliance") or {}).items()
        if isinstance(value, dict) and "passed" in value
    }


def _call_snapshot(result_path, evaluation):
    metadata = evaluation.get("metadata") or {}
    escalation = evaluation.get("escalation") or {}
    hybrid = escalation.get("hybrid") or {}
    workflow = evaluation.get("workflow") or {}
    steps = workflow.get("expected_steps") or []
    pipeline = evaluation.get("_pipeline") or {}

    return {
        "domain": metadata.get("domain"),
        "accent": metadata.get("accent"),
        "duration_seconds": metadata.get("duration_seconds"),
        "result_sha256": _sha256(result_path),
        "rubric_version": evaluation.get("rubric_version"),
        "decision": {
            "attention_required": None,
            "status": "not_defined_in_v1",
            "exposed_risk_level": escalation.get("risk_level"),
        },
        "escalation": {
            "text_risk": hybrid.get("text_risk"),
            "acoustic_risk": hybrid.get("acoustic_risk"),
            "method": hybrid.get("method"),
            "late_mean": hybrid.get("late_mean_escalation"),
            "peak": hybrid.get("peak_escalation"),
            "red_flags": escalation.get("red_flags") or [],
        },
        "quality_scores": _quality_scores(evaluation),
        "compliance": _compliance_states(evaluation),
        "workflow": {
            "subject": workflow.get("subject"),
            "total": len(steps),
            "met": sum(step.get("met") is True for step in steps),
            "missed": sum(step.get("met") is False for step in steps),
            "not_applicable": sum(step.get("met") is None for step in steps),
        },
        "evidence": evaluation.get("_anchor_stats") or {},
        "pipeline": {
            "wall_clock_seconds": pipeline.get("wall_clock"),
            "anchor_passes": pipeline.get("anchor_passes"),
            "nodes": pipeline.get("nodes") or {},
        },
    }


def build_snapshot(summary_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_dir = summary_path.parent
    calls = {}
    missing = {}
    risk_counts = Counter()
    domain_counts = Counter()
    wall_clock = []
    node_times = defaultdict(list)
    anchored = 0
    evidence_total = 0

    for call_id in sorted((summary.get("calls") or {}).keys()):
        result_path = result_dir / f"{call_id}_graph.json"
        if not result_path.exists():
            missing[call_id] = f"missing {result_path.name}"
            continue

        evaluation = json.loads(result_path.read_text(encoding="utf-8"))
        call = _call_snapshot(result_path, evaluation)
        calls[call_id] = call
        risk_counts[call["decision"]["exposed_risk_level"] or "unknown"] += 1
        domain_counts[call["domain"] or "unknown"] += 1

        timing = call["pipeline"]["wall_clock_seconds"]
        if isinstance(timing, (int, float)):
            wall_clock.append(timing)
        for node, values in call["pipeline"]["nodes"].items():
            duration = values.get("duration") if isinstance(values, dict) else None
            if isinstance(duration, (int, float)):
                node_times[node].append(duration)

        evidence = call["evidence"]
        anchored += evidence.get("anchored", 0) or 0
        evidence_total += evidence.get("total", 0) or 0

    failures = dict(summary.get("failures") or {})
    failures.update(missing)
    return {
        "schema_version": 1,
        "evaluator_version": "v1",
        "rubric_version": summary.get("rubric_version"),
        "source_commit": _source_commit(),
        "source_summary": str(summary_path.relative_to(HERE)),
        "decision_contract": {
            "attention_required": "not_defined_in_v1",
            "comparison_signal": "escalation.risk_level",
        },
        "aggregate": {
            "expected_calls": len(summary.get("calls") or {}),
            "captured_calls": len(calls),
            "failed_calls": len(failures),
            "risk_counts": dict(sorted(risk_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "evidence": {
                "anchored": anchored,
                "total": evidence_total,
                "anchor_rate": (
                    round(anchored / evidence_total, 4)
                    if evidence_total
                    else None
                ),
            },
            "wall_clock_seconds": _timing_summary(wall_clock),
            "node_seconds": {
                node: _timing_summary(values)
                for node, values in sorted(node_times.items())
            },
        },
        "calls": calls,
        "failures": failures,
    }


def verify_snapshot(snapshot_path):
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    result_dir = HERE / "results"
    changed = []

    for call_id, expected in (snapshot.get("calls") or {}).items():
        result_path = result_dir / f"{call_id}_graph.json"
        if not result_path.exists():
            changed.append(f"{call_id}: missing result")
        elif _sha256(result_path) != expected.get("result_sha256"):
            changed.append(f"{call_id}: hash changed")

    if changed:
        print("\n".join(changed))
        return 1
    print(f"Baseline verified: {len(snapshot.get('calls') or {})} result files")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(verify_snapshot(args.output.resolve()))

    snapshot = build_snapshot(args.summary.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(
        f"Captured {snapshot['aggregate']['captured_calls']} calls "
        f"with {snapshot['aggregate']['failed_calls']} failures -> {output}"
    )


if __name__ == "__main__":
    main()
