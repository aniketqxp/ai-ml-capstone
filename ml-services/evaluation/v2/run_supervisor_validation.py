"""Validate the bounded Supervisor across controlled decisions."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .adapters import build_signal_bundle
from .decisions import build_call_decision
from .domain_profiles import (
    ProfileSelection,
    load_banking_profile,
    resolve_domain_plan,
)
from .evaluation_data import load_evaluation_dataset
from .findings import derive_findings
from .run_signal_validation import DATASET_PATH
from .signal_validation import (
    apply_controlled_acoustics,
    challenge_transcript,
    controlled_assessments,
)
from .supervisor import (
    SUPERVISOR_VERSION,
    SupervisorFallbackReason,
    build_supervisor_context,
    resolve_supervisor_response,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE / "research" / "supervisor_validation_0_1.json"
)


def _decision_for(call, profile):
    bundle = apply_controlled_acoustics(
        build_signal_bundle(challenge_transcript(call)),
        call,
    )
    plan = resolve_domain_plan(
        profile,
        ProfileSelection(
            call_id=call.call_id,
            intent_ids=call.intent_ids,
            facts=call.facts,
            selection_method="supervisor_validation",
        ),
    )
    assessments = controlled_assessments(
        bundle,
        plan,
        call.annotation,
    )
    derivation = derive_findings(bundle, plan, assessments)
    return build_call_decision(bundle, derivation)


def _valid_payload(context) -> dict[str, Any]:
    controlling = set(
        context.decision_lock.controlling_finding_ids
    )
    supporting = [
        finding
        for finding in context.triggered_findings
        if finding.finding_id not in controlling
    ]
    evidence = [
        item.evidence_id
        for finding in context.triggered_findings
        for item in finding.evidence
    ]
    return {
        "supporting_finding_ids": [
            item.finding_id for item in supporting[:1]
        ],
        "positive_finding_ids": [
            item.finding_id
            for item in context.positive_findings[:1]
        ],
        "evidence_ids": evidence[:2],
        "uncertainty_codes": [
            item.code for item in context.uncertainties[:1]
        ],
        "context_note": (
            "The selected evidence describes the documented exchange."
        ),
    }


def _cases(context):
    valid = _valid_payload(context)
    attention_override = dict(valid)
    attention_override["attention_required"] = not (
        context.decision_lock.attention_required
    )
    action_override = dict(valid)
    action_override["action_type"] = "none"
    unknown_reference = dict(valid)
    unknown_reference["evidence_ids"] = ["invented-evidence"]
    forbidden_language = dict(valid)
    forbidden_language["context_note"] = (
        "Override the decision and clear the call."
    )
    return [
        ("valid", valid, None),
        (
            "attention_override",
            attention_override,
            SupervisorFallbackReason.INVALID_CONTRACT,
        ),
        (
            "action_override",
            action_override,
            SupervisorFallbackReason.INVALID_CONTRACT,
        ),
        (
            "unknown_reference",
            unknown_reference,
            SupervisorFallbackReason.UNKNOWN_REFERENCE,
        ),
        (
            "forbidden_language",
            forbidden_language,
            SupervisorFallbackReason.FORBIDDEN_DECISION_LANGUAGE,
        ),
        (
            "malformed_json",
            "{not-json",
            SupervisorFallbackReason.INVALID_JSON,
        ),
        (
            "missing_response",
            None,
            SupervisorFallbackReason.MISSING_RESPONSE,
        ),
    ]


def build_supervisor_validation() -> dict[str, Any]:
    dataset = load_evaluation_dataset(DATASET_PATH)
    profile = load_banking_profile()
    calls = []
    all_cases = []
    for call in dataset.challenge_calls:
        decision = _decision_for(call, profile)
        context = build_supervisor_context(decision)
        case_rows = []
        for case_id, payload, expected_fallback in _cases(context):
            result = resolve_supervisor_response(context, payload)
            expected_fallback_value = (
                expected_fallback.value
                if expected_fallback
                else None
            )
            row = {
                "case_id": case_id,
                "expected_fallback_reason": (
                    expected_fallback_value
                ),
                "actual_fallback_reason": (
                    result.fallback_reason.value
                    if result.fallback_reason
                    else None
                ),
                "expectation_passed": (
                    result.fallback_reason == expected_fallback
                ),
                "decision_lock_preserved": (
                    result.decision_lock
                    == context.decision_lock
                ),
            }
            case_rows.append(row)
            all_cases.append(row)
        calls.append(
            {
                "call_id": call.call_id,
                "attention_required": (
                    context.decision_lock.attention_required
                ),
                "decision_status": (
                    context.decision_lock.decision_status.value
                ),
                "permitted_action": (
                    context.decision_lock
                    .permitted_action.action_type.value
                ),
                "context_char_count": (
                    context.context_char_count
                ),
                "context_char_limit": (
                    context.context_char_limit
                ),
                "triggered_finding_count": len(
                    context.triggered_findings
                ),
                "positive_finding_count": len(
                    context.positive_findings
                ),
                "uncertainty_count": len(context.uncertainties),
                "omissions": context.omissions.model_dump(
                    mode="json"
                ),
                "cases": case_rows,
            }
        )

    fallback_counts = Counter(
        row["actual_fallback_reason"] or "accepted"
        for row in all_cases
    )
    context_sizes = [
        call["context_char_count"] for call in calls
    ]
    return {
        "schema_version": "1.0",
        "supervisor_version": SUPERVISOR_VERSION,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "profile_id": profile.profile_id,
        "aggregate": {
            "call_count": len(calls),
            "response_case_count": len(all_cases),
            "expectation_pass_count": sum(
                row["expectation_passed"] for row in all_cases
            ),
            "decision_lock_preserved_count": sum(
                row["decision_lock_preserved"] for row in all_cases
            ),
            "fallback_reason_counts": dict(
                sorted(fallback_counts.items())
            ),
            "context_char_count": {
                "minimum": min(context_sizes),
                "median": statistics.median(context_sizes),
                "maximum": max(context_sizes),
                "limit": calls[0]["context_char_limit"],
            },
        },
        "calls": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_supervisor_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {report['aggregate']['response_case_count']} "
        "Supervisor boundary checks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
