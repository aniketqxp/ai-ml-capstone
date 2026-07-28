"""Run evaluator-v2 attention and action policy over banking fixtures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .decisions import (
    DECISION_POLICY_ID,
    DECISION_POLICY_VERSION,
    build_call_decision,
)
from .run_findings_batch import (
    REPO_ROOT,
    batch_source_metadata,
    iter_batch_artifacts,
)
from .validation import validate_decision_references


def build_batch_summary(
    *,
    profile_ref: str,
    transcript_ref: str,
    sentiment_ref: str,
    assessment_dir: Path | None,
) -> dict[str, Any]:
    calls = []
    for artifacts in iter_batch_artifacts(
        profile_ref=profile_ref,
        transcript_ref=transcript_ref,
        sentiment_ref=sentiment_ref,
        assessment_dir=assessment_dir,
    ):
        decision = build_call_decision(
            artifacts.bundle,
            artifacts.derivation,
        )
        validate_decision_references(
            decision,
            artifacts.bundle,
        )
        calls.append(
            {
                "call_id": artifacts.call_id,
                "attention_required": (
                    decision.attention_required
                ),
                "decision_status": decision.decision_status.value,
                "triggered_finding_count": len(
                    decision.triggered_findings
                ),
                "positive_finding_count": len(
                    decision.positive_findings
                ),
                "uncertainty_count": len(
                    decision.uncertainties
                ),
                "controlling_finding_ids": (
                    decision.decision_trace
                    .controlling_finding_ids
                ),
                "recovery_effect": (
                    decision.decision_trace
                    .recovery_effect.value
                ),
                "action_type": (
                    decision.recommended_action.action_type.value
                ),
                "action_execution": (
                    decision.recommended_action.execution.value
                ),
                "signal_bundle_sha256": (
                    decision.signal_bundle_sha256
                ),
                "supplied_assessment_count": (
                    artifacts.supplied_assessment_count
                ),
            }
        )

    action_types = Counter(
        call["action_type"] for call in calls
    )
    action_execution = Counter(
        call["action_execution"] for call in calls
    )
    decision_statuses = Counter(
        call["decision_status"] for call in calls
    )
    return {
        "schema_version": "1.0",
        "decision_policy_id": DECISION_POLICY_ID,
        "decision_policy_version": DECISION_POLICY_VERSION,
        **batch_source_metadata(
            profile_ref=profile_ref,
            transcript_ref=transcript_ref,
            sentiment_ref=sentiment_ref,
            assessment_dir=assessment_dir,
        ),
        "aggregate": {
            "call_count": len(calls),
            "attention_required_count": sum(
                call["attention_required"] for call in calls
            ),
            "decision_status_counts": dict(
                sorted(decision_statuses.items())
            ),
            "triggered_finding_count": sum(
                call["triggered_finding_count"]
                for call in calls
            ),
            "positive_finding_count": sum(
                call["positive_finding_count"]
                for call in calls
            ),
            "uncertainty_count": sum(
                call["uncertainty_count"] for call in calls
            ),
            "action_type_counts": dict(
                sorted(action_types.items())
            ),
            "action_execution_counts": dict(
                sorted(action_execution.items())
            ),
        },
        "calls": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-ref", default="HEAD")
    parser.add_argument("--transcript-ref", default="HEAD")
    parser.add_argument(
        "--sentiment-ref",
        default="origin/integration/full-pipeline-test",
    )
    parser.add_argument("--assessment-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    assessment_dir = args.assessment_dir
    if assessment_dir and not assessment_dir.is_absolute():
        assessment_dir = REPO_ROOT / assessment_dir
    summary = build_batch_summary(
        profile_ref=args.profile_ref,
        transcript_ref=args.transcript_ref,
        sentiment_ref=args.sentiment_ref,
        assessment_dir=assessment_dir,
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        output = (
            args.output
            if args.output.is_absolute()
            else REPO_ROOT / args.output
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
