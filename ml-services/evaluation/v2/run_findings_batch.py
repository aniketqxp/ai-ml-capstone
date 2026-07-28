"""Run evaluator-v2 finding derivation over the banking fixture batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .acoustic_dynamics import derive_acoustic_dynamics
from .adapters import build_signal_bundle
from .domain_profiles import (
    DomainProfile,
    ProfileSelection,
    resolve_domain_plan,
)
from .findings import (
    RequirementAssessmentBatch,
    derive_findings,
)
from .schemas import Modality

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = "ml-services/evaluation/v2/profiles/banking_v1.json"
SELECTIONS_PATH = (
    "ml-services/evaluation/v2/profiles/"
    "banking_call_selections.json"
)
TRANSCRIPT_PREFIX = "data/sentence_segments/banking"
SENTIMENT_PREFIX = (
    "ml-services/outputs/backend/"
    "sentiment_calls_with_features/banking"
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    )


def _git_json(ref: str, path: str) -> dict[str, Any]:
    return json.loads(_git("show", f"{ref}:{path}"))


def _assessment(
    assessment_dir: Path | None,
    call_id: str,
) -> tuple[RequirementAssessmentBatch | None, str | None]:
    if assessment_dir is None:
        return None, None
    path = assessment_dir / f"{call_id}.json"
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return (
        RequirementAssessmentBatch.model_validate_json(raw),
        hashlib.sha256(raw).hexdigest(),
    )


def _has_acoustic_support(finding) -> bool:
    return any(
        evidence.modality == Modality.ACOUSTIC
        for evidence in finding.evidence
    )


def build_batch_summary(
    *,
    profile_ref: str,
    transcript_ref: str,
    sentiment_ref: str,
    assessment_dir: Path | None,
) -> dict[str, Any]:
    profile = DomainProfile.model_validate(
        _git_json(profile_ref, PROFILE_PATH)
    )
    selection_payload = _git_json(profile_ref, SELECTIONS_PATH)
    calls = []
    for raw_selection in selection_payload["calls"]:
        selection = ProfileSelection.model_validate(raw_selection)
        call_id = selection.call_id
        transcript_path = f"{TRANSCRIPT_PREFIX}/{call_id}.json"
        sentiment_path = (
            f"{SENTIMENT_PREFIX}/"
            f"{call_id}_backend_sentiment_with_features.json"
        )
        transcript = _git_json(transcript_ref, transcript_path)
        sentiment = _git_json(sentiment_ref, sentiment_path)
        bundle = derive_acoustic_dynamics(
            build_signal_bundle(
                transcript,
                sentiment,
                transcript_source=(
                    f"{transcript_ref}:{transcript_path}"
                ),
                sentiment_source=(
                    f"{sentiment_ref}:{sentiment_path}"
                ),
            )
        )
        plan = resolve_domain_plan(profile, selection)
        assessments, assessment_sha256 = _assessment(
            assessment_dir,
            call_id,
        )
        derivation = derive_findings(
            bundle,
            plan,
            assessments,
        )
        calls.append(
            {
                "call_id": call_id,
                "applicable_requirement_count": len(
                    plan.requirements
                ),
                "supplied_assessment_count": (
                    len(assessments.assessments)
                    if assessments
                    else 0
                ),
                "requirement_uncertainty_count": len(
                    derivation.uncertainties
                ),
                "triggered_finding_types": [
                    finding.finding_type
                    for finding in derivation.triggered_findings
                ],
                "positive_finding_types": [
                    finding.finding_type
                    for finding in derivation.positive_findings
                ],
                "acoustically_supported_finding_count": sum(
                    _has_acoustic_support(finding)
                    for finding in (
                        derivation.triggered_findings
                        + derivation.positive_findings
                    )
                ),
                "suppressed_duplicates": (
                    derivation.suppressed_duplicates
                ),
                "assessment_sha256": assessment_sha256,
            }
        )

    triggered = Counter(
        finding_type
        for call in calls
        for finding_type in call["triggered_finding_types"]
    )
    positive = Counter(
        finding_type
        for call in calls
        for finding_type in call["positive_finding_types"]
    )
    return {
        "schema_version": "1.0",
        "profile_ref": profile_ref,
        "profile_commit": _git("rev-parse", profile_ref).strip(),
        "transcript_ref": transcript_ref,
        "transcript_commit": _git(
            "rev-parse",
            transcript_ref,
        ).strip(),
        "sentiment_ref": sentiment_ref,
        "sentiment_commit": _git(
            "rev-parse",
            sentiment_ref,
        ).strip(),
        "assessment_directory": (
            str(assessment_dir)
            if assessment_dir
            else None
        ),
        "aggregate": {
            "call_count": len(calls),
            "applicable_requirement_count": sum(
                call["applicable_requirement_count"]
                for call in calls
            ),
            "supplied_assessment_count": sum(
                call["supplied_assessment_count"]
                for call in calls
            ),
            "requirement_uncertainty_count": sum(
                call["requirement_uncertainty_count"]
                for call in calls
            ),
            "triggered_finding_type_counts": dict(
                sorted(triggered.items())
            ),
            "positive_finding_type_counts": dict(
                sorted(positive.items())
            ),
            "acoustically_supported_finding_count": sum(
                call["acoustically_supported_finding_count"]
                for call in calls
            ),
            "suppressed_duplicate_count": sum(
                len(call["suppressed_duplicates"])
                for call in calls
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
    parser.add_argument(
        "--assessment-dir",
        type=Path,
        help=(
            "Optional directory containing one grounded assessment "
            "batch per call."
        ),
    )
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
