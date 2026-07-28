from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import build_signal_bundle
from .decisions import build_call_decision
from .domain_profiles import (
    DomainProfile,
    ProfileSelection,
    resolve_domain_plan,
)
from .evaluation_data import load_evaluation_dataset
from .findings import derive_findings
from .presentation import (
    OUTPUT_INVENTORY,
    PRESENTATION_VERSION,
    PresentationValidationReport,
    project_call_evaluation,
    scenario_result,
)
from .signal_validation import (
    apply_controlled_acoustics,
    challenge_transcript,
    controlled_assessments,
)

HERE = Path(__file__).resolve().parent
PROFILE_PATH = HERE / "profiles" / "banking_v1.json"
DATASET_PATH = HERE / "research" / "evaluation_data_0_1.json"
DEFAULT_OUTPUT = HERE / "research" / "presentation_validation_0_1.json"

HUMAN_TEST_QUESTIONS = [
    "Does this call need attention?",
    "What is the main reason?",
    "Where in the call is the supporting evidence?",
    "What did the agent handle well?",
    "What action follows from this result?",
    "Is the evaluation complete?",
]


def build_presentation_validation_report() -> PresentationValidationReport:
    dataset = load_evaluation_dataset(DATASET_PATH)
    profile = DomainProfile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )
    scenarios = []
    for call in dataset.challenge_calls:
        transcript = challenge_transcript(call)
        text_bundle = build_signal_bundle(
            transcript,
            transcript_source=(
                f"{dataset.dataset_id}:{call.call_id}"
            ),
        )
        bundle = apply_controlled_acoustics(text_bundle, call)
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
        derivation = derive_findings(bundle, plan, assessments)
        decision = build_call_decision(bundle, derivation)
        view = project_call_evaluation(decision)
        scenarios.append(scenario_result(view))

    passed = sum(item.check.passed for item in scenarios)
    return PresentationValidationReport(
        presentation_version=PRESENTATION_VERSION,
        scenario_source=(
            f"{dataset.dataset_id}:{dataset.dataset_version}:"
            "controlled_challenges"
        ),
        scenario_count=len(scenarios),
        passed_scenario_count=passed,
        structural_checks_passed=passed == len(scenarios),
        human_test_questions=HUMAN_TEST_QUESTIONS,
        inventory=list(OUTPUT_INVENTORY),
        scenarios=scenarios,
        limitations=[
            (
                "Structural checks verify that each scenario exposes one "
                "literal answer path; they do not measure human "
                "comprehension."
            ),
            (
                "No unfamiliar-user sessions were run in this phase. The "
                "six-question protocol must be run after the Phase 11 page "
                "is interactive."
            ),
            (
                "Controlled challenge calls use authored known answers and "
                "do not establish real-world manager agreement."
            ),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_presentation_validation_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {report.passed_scenario_count}/"
        f"{report.scenario_count} passing presentation scenarios."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
