from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import RUNTIME_VERSION, safe_run_shadow_evaluation

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRANSCRIPT_ROOT = REPO_ROOT / "data" / "sentence_segments" / "banking"
LEGACY_ROOT = REPO_ROOT / "ml-services" / "evaluation" / "results"
DEFAULT_OUTPUT = REPO_ROOT / "frontend" / "public" / "evaluation-v2"
DEFAULT_SUMMARY = HERE / "research" / "shadow_rollout_0_1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    decision_statuses: dict[str, int] = {}
    comparison_counts = {
        "comparable": 0,
        "not_comparable": 0,
        "attention_agreement": 0,
        "attention_disagreement": 0,
    }
    legacy_attention = 0
    v2_attention = 0
    rows = []
    for transcript_path in sorted(TRANSCRIPT_ROOT.glob("*.json")):
        call_id = transcript_path.stem
        legacy_path = LEGACY_ROOT / f"{call_id}_graph.json"
        run = safe_run_shadow_evaluation(
            transcript=_load(transcript_path),
            legacy_evaluation=(
                _load(legacy_path) if legacy_path.exists() else None
            ),
            transcript_source=str(transcript_path.relative_to(REPO_ROOT)),
        )
        output = args.output_dir / f"{call_id}.json"
        output.write_text(
            json.dumps(run.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        counts[run.status.value] = counts.get(run.status.value, 0) + 1
        if run.legacy_proxy and run.legacy_proxy.attention_required:
            legacy_attention += 1
        if run.decision:
            status = run.decision.decision_status.value
            decision_statuses[status] = decision_statuses.get(status, 0) + 1
            v2_attention += int(run.decision.attention_required)
        if run.comparison:
            if run.comparison.comparable:
                comparison_counts["comparable"] += 1
                key = (
                    "attention_agreement"
                    if run.comparison.attention_agreement
                    else "attention_disagreement"
                )
                comparison_counts[key] += 1
            else:
                comparison_counts["not_comparable"] += 1
        rows.append({
            "call_id": call_id,
            "status": run.status.value,
            "decision_status": (
                run.decision.decision_status.value
                if run.decision
                else None
            ),
            "legacy_attention_proxy": (
                run.legacy_proxy.attention_required
                if run.legacy_proxy
                else None
            ),
            "v2_attention": (
                run.decision.attention_required
                if run.decision
                else None
            ),
            "comparable": (
                run.comparison.comparable
                if run.comparison
                else False
            ),
            "limitations": run.limitations,
        })

    summary = {
        "schema_version": "1.0",
        "runtime_version": RUNTIME_VERSION,
        "population": "ten_banking_calls",
        "call_count": len(rows),
        "run_statuses": dict(sorted(counts.items())),
        "decision_statuses": dict(sorted(decision_statuses.items())),
        "legacy_attention_proxy_count": legacy_attention,
        "v2_attention_count": v2_attention,
        "comparison": comparison_counts,
        "conclusion": (
            "No attention disagreement is reported because every v2 run "
            "has incomplete semantic requirement coverage."
        ),
        "calls": rows,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {sum(counts.values())} shadow runs: "
        + ", ".join(
            f"{status}={count}"
            for status, count in sorted(counts.items())
        )
    )
    print(f"Summary -> {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
