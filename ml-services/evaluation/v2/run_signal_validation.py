"""Execute Phase 8 evaluator-v2 modality ablations."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .acoustic_dynamics import derive_acoustic_dynamics
from .adapters import build_signal_bundle
from .domain_profiles import (
    DomainProfile,
    ProfileSelection,
    resolve_domain_plan,
)
from .evaluation_data import load_evaluation_dataset
from .signal_validation import (
    AblationVariant,
    ClaimLevel,
    SignalValidationReport,
    ValidationPopulation,
    apply_controlled_acoustics,
    build_report,
    challenge_transcript,
    controlled_assessments,
    evaluate_bundle,
    make_audio_only_bundle,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
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
DATASET_PATH = HERE / "research" / "evaluation_data_0_1.json"
DEFAULT_OUTPUT = HERE / "research" / "signal_validation_0_1.json"
DEFAULT_SUMMARY = (
    HERE / "research" / "signal_validation_summary_0_1.json"
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


def _existing_results(
    *,
    dataset,
    profile: DomainProfile,
    profile_ref: str,
    transcript_ref: str,
    sentiment_ref: str,
):
    selections = {
        item["call_id"]: ProfileSelection.model_validate(item)
        for item in _git_json(
            profile_ref,
            SELECTIONS_PATH,
        )["calls"]
    }
    output = []
    for record in dataset.existing_calls:
        call_id = record.call_id
        transcript_path = f"{TRANSCRIPT_PREFIX}/{call_id}.json"
        sentiment_path = (
            f"{SENTIMENT_PREFIX}/"
            f"{call_id}_backend_sentiment_with_features.json"
        )
        transcript = _git_json(transcript_ref, transcript_path)
        sentiment = _git_json(sentiment_ref, sentiment_path)
        text_bundle = build_signal_bundle(
            transcript,
            transcript_source=f"{transcript_ref}:{transcript_path}",
        )
        multimodal_bundle = derive_acoustic_dynamics(
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
        audio_bundle = make_audio_only_bundle(multimodal_bundle)
        plan = resolve_domain_plan(
            profile,
            selections[call_id],
        )
        bundles = {
            AblationVariant.TEXT_ONLY: text_bundle,
            AblationVariant.AUDIO_ONLY: audio_bundle,
            AblationVariant.MULTIMODAL: multimodal_bundle,
        }
        for variant, bundle in bundles.items():
            limitations = [
                "researcher_seed_label_not_human_adjudicated",
                "no_grounded_requirement_assessments",
            ]
            if variant == AblationVariant.AUDIO_ONLY:
                limitations.append("transcript_semantics_removed")
            result, _ = evaluate_bundle(
                annotation=record.annotation,
                bundle=bundle,
                plan=plan,
                assessments=None,
                population=ValidationPopulation.EXISTING_CALLS,
                claim_level=ClaimLevel.DETECTOR_ONLY,
                variant=variant,
                limitations=limitations,
            )
            output.append(result)
    return output


def _challenge_results(dataset, profile: DomainProfile):
    output = []
    for call in dataset.challenge_calls:
        transcript = challenge_transcript(call)
        text_bundle = build_signal_bundle(
            transcript,
            transcript_source=(
                f"{dataset.dataset_id}:{call.call_id}"
            ),
        )
        multimodal_bundle = apply_controlled_acoustics(
            text_bundle,
            call,
        )
        audio_bundle = make_audio_only_bundle(multimodal_bundle)
        selection = ProfileSelection(
            call_id=call.call_id,
            intent_ids=call.intent_ids,
            facts=call.facts,
            selection_method="controlled_challenge",
        )
        plan = resolve_domain_plan(profile, selection)
        assessments = controlled_assessments(
            text_bundle,
            plan,
            call.annotation,
        )
        variants = {
            AblationVariant.TEXT_ONLY: (
                text_bundle,
                assessments,
                [
                    "semantic_assessments_are_controlled_known_answers"
                ],
            ),
            AblationVariant.AUDIO_ONLY: (
                audio_bundle,
                None,
                [
                    "transcript_semantics_removed",
                    "no_semantic_requirement_assessments",
                    "acoustic_condition_is_authored_not_raw_audio",
                ],
            ),
            AblationVariant.MULTIMODAL: (
                multimodal_bundle,
                assessments,
                [
                    "semantic_assessments_are_controlled_known_answers",
                    "acoustic_condition_is_authored_not_raw_audio",
                ],
            ),
        }
        for variant, (
            bundle,
            selected_assessments,
            limitations,
        ) in variants.items():
            result, _ = evaluate_bundle(
                annotation=call.annotation,
                bundle=bundle,
                plan=plan,
                assessments=selected_assessments,
                population=(
                    ValidationPopulation.CONTROLLED_CHALLENGES
                ),
                claim_level=(
                    ClaimLevel.KNOWN_ANSWER_POLICY_VALIDATION
                ),
                variant=variant,
                limitations=limitations,
            )
            output.append(result)
    return output


def build_signal_validation_report(
    *,
    profile_ref: str = "HEAD",
    transcript_ref: str = "HEAD",
    sentiment_ref: str = "origin/integration/full-pipeline-test",
) -> SignalValidationReport:
    dataset = load_evaluation_dataset(DATASET_PATH)
    profile = DomainProfile.model_validate(
        _git_json(profile_ref, PROFILE_PATH)
    )
    call_results = [
        *_existing_results(
            dataset=dataset,
            profile=profile,
            profile_ref=profile_ref,
            transcript_ref=transcript_ref,
            sentiment_ref=sentiment_ref,
        ),
        *_challenge_results(dataset, profile),
    ]
    return build_report(
        dataset=dataset,
        profile=profile,
        source_revisions={
            "profile_ref": profile_ref,
            "profile_commit": _git(
                "rev-parse",
                profile_ref,
            ).strip(),
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
        },
        call_results=call_results,
    )


def compact_summary(
    report: SignalValidationReport,
) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "validation_version": report.validation_version,
        "dataset_id": report.dataset_id,
        "dataset_version": report.dataset_version,
        "profile_id": report.profile_id,
        "source_revisions": report.source_revisions,
        "variant_metrics": [
            item.model_dump(mode="json")
            for item in report.variant_metrics
        ],
        "pair_metrics": [
            item.model_dump(mode="json")
            for item in report.pair_metrics
        ],
        "human_agreement": report.human_agreement.model_dump(
            mode="json"
        ),
        "threshold_assessment": (
            report.threshold_assessment.model_dump(mode="json")
        ),
        "acoustic_value": report.acoustic_value.model_dump(
            mode="json"
        ),
        "conclusions": report.conclusions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-ref", default="HEAD")
    parser.add_argument("--transcript-ref", default="HEAD")
    parser.add_argument(
        "--sentiment-ref",
        default="origin/integration/full-pipeline-test",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    args = parser.parse_args()

    report = build_signal_validation_report(
        profile_ref=args.profile_ref,
        transcript_ref=args.transcript_ref,
        sentiment_ref=args.sentiment_ref,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(compact_summary(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(report.call_results)} call ablations and "
        f"{len(report.pair_results)} pair checks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
