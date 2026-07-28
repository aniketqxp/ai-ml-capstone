"""Deterministic attention and action policy for evaluator v2."""
from __future__ import annotations

import hashlib
import json

from .findings import FindingDerivation
from .schemas import (
    ActionExecution,
    ActionType,
    CallDecision,
    DecisionPolicyTrace,
    DecisionStatus,
    Finding,
    FindingCategory,
    FindingQualification,
    FindingSeverity,
    PresentationSelection,
    QualificationReason,
    RecommendedAction,
    RecoveryEffect,
    ReliabilityStatus,
    SignalBundle,
    SourceProvenance,
    Visibility,
)

DECISION_POLICY_ID = "evaluator-v2-attention"
DECISION_POLICY_VERSION = "0.1.0"
ATTENTION_AGGREGATION = "any_qualifying_finding"
RECOVERY_FINDING_TYPE = "escalation.recovery_observed"
FOLLOW_UP_FINDING_TYPES = {
    "process.repeat_contact_unresolved",
}


_PRECEDENCE_ORDER = (
    "critical.escalation",
    "critical.required_control",
    "critical.outcome",
    "critical.process",
    "critical.agent_behavior",
    "critical.data_quality",
    "review.outcome",
    "review.required_control",
    "review.escalation",
    "review.process",
    "review.agent_behavior",
    "review.data_quality",
)

_PRECEDENCE_INDEX = {
    group: index
    for index, group in enumerate(_PRECEDENCE_ORDER)
}


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.decisions",
        producer_version=DECISION_POLICY_VERSION,
        method=(
            "any_qualifying_finding_with_named_precedence_and_"
            "constrained_actions"
        ),
    )


def signal_bundle_sha256(bundle: SignalBundle) -> str:
    payload = json.dumps(
        bundle.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _precedence_group(finding: Finding) -> str:
    return f"{finding.severity.value}.{finding.category.value}"


def _qualification(finding: Finding) -> FindingQualification:
    if finding.visibility == Visibility.INTERNAL:
        return FindingQualification(
            finding_id=finding.finding_id,
            qualifies_for_attention=False,
            reason=QualificationReason.EXCLUDED_INTERNAL,
        )
    if finding.reliability.status in (
        ReliabilityStatus.UNUSABLE,
        ReliabilityStatus.UNAVAILABLE,
    ):
        return FindingQualification(
            finding_id=finding.finding_id,
            qualifies_for_attention=False,
            reason=QualificationReason.EXCLUDED_UNRELIABLE,
        )
    if finding.severity == FindingSeverity.INFO:
        return FindingQualification(
            finding_id=finding.finding_id,
            qualifies_for_attention=False,
            reason=QualificationReason.EXCLUDED_INFORMATIONAL,
        )

    if finding.reliability.status == ReliabilityStatus.LIMITED:
        reason = QualificationReason.QUALIFIES_LIMITED_REVIEW
    elif finding.severity == FindingSeverity.CRITICAL:
        reason = QualificationReason.QUALIFIES_CRITICAL
    else:
        reason = QualificationReason.QUALIFIES_REVIEW
    group = _precedence_group(finding)
    if group not in _PRECEDENCE_INDEX:
        raise ValueError(
            f"attention policy has no precedence for {group!r}"
        )
    return FindingQualification(
        finding_id=finding.finding_id,
        qualifies_for_attention=True,
        reason=reason,
        precedence_group=group,
    )


def _controlling_ids(
    qualifications: list[FindingQualification],
) -> list[str]:
    qualifying = [
        item
        for item in qualifications
        if item.qualifies_for_attention
        and item.precedence_group is not None
    ]
    if not qualifying:
        return []
    controlling_group = min(
        (item.precedence_group for item in qualifying),
        key=lambda group: _PRECEDENCE_INDEX[group],
    )
    return sorted(
        item.finding_id
        for item in qualifying
        if item.precedence_group == controlling_group
    )


def _controlling_findings(
    derivation: FindingDerivation,
    controlling_ids: list[str],
) -> list[Finding]:
    by_id = {
        finding.finding_id: finding
        for finding in derivation.triggered_findings
    }
    return [by_id[finding_id] for finding_id in controlling_ids]


def _reason(findings: list[Finding]) -> str:
    titles = "; ".join(finding.title for finding in findings)
    return f"Controlling findings: {titles}."


def _automatic_review_case(
    findings: list[Finding],
) -> RecommendedAction:
    return RecommendedAction(
        action_type=ActionType.CREATE_REVIEW_CASE,
        execution=ActionExecution.AUTOMATIC,
        label="Create review case",
        reason=_reason(findings),
        finding_ids=[
            finding.finding_id for finding in findings
        ],
        automation_allowed=True,
        requires_human_approval=False,
    )


def _approval_action(
    findings: list[Finding],
    action_type: ActionType,
    label: str,
) -> RecommendedAction:
    return RecommendedAction(
        action_type=action_type,
        execution=ActionExecution.REQUIRES_APPROVAL,
        label=label,
        reason=_reason(findings),
        finding_ids=[
            finding.finding_id for finding in findings
        ],
        automation_allowed=False,
        requires_human_approval=True,
    )


def _recommended_action(
    controlling: list[Finding],
    decision_status: DecisionStatus,
) -> RecommendedAction:
    if not controlling:
        reason = (
            "No action because applicable requirements remain "
            "unassessed; this call has not been cleared."
            if decision_status
            == DecisionStatus.INSUFFICIENT_EVIDENCE
            else "No negative finding qualified for attention."
        )
        return RecommendedAction(
            action_type=ActionType.NONE,
            execution=ActionExecution.NO_ACTION,
            label="No automated action",
            reason=reason,
        )

    if any(
        finding.severity == FindingSeverity.CRITICAL
        for finding in controlling
    ):
        return _automatic_review_case(controlling)

    categories = {finding.category for finding in controlling}
    if (
        FindingCategory.OUTCOME in categories
        or any(
            finding.finding_type in FOLLOW_UP_FINDING_TYPES
            for finding in controlling
        )
    ):
        return _approval_action(
            controlling,
            ActionType.REQUEST_CUSTOMER_FOLLOW_UP,
            "Request customer follow-up",
        )
    if categories.intersection(
        {
            FindingCategory.PROCESS,
            FindingCategory.AGENT_BEHAVIOR,
        }
    ):
        return _approval_action(
            controlling,
            ActionType.RECOMMEND_COACHING,
            "Recommend coaching",
        )
    return _automatic_review_case(controlling)


def _decision_trace(
    derivation: FindingDerivation,
) -> DecisionPolicyTrace:
    qualifications = [
        _qualification(finding)
        for finding in derivation.triggered_findings
    ]
    recovery_ids = sorted(
        finding.finding_id
        for finding in derivation.positive_findings
        if finding.finding_type == RECOVERY_FINDING_TYPE
    )
    return DecisionPolicyTrace(
        policy_id=DECISION_POLICY_ID,
        policy_version=DECISION_POLICY_VERSION,
        aggregation=ATTENTION_AGGREGATION,
        qualifications=qualifications,
        controlling_finding_ids=_controlling_ids(qualifications),
        recovery_finding_ids=recovery_ids,
        recovery_effect=(
            RecoveryEffect.CONTEXT_ONLY
            if recovery_ids
            else RecoveryEffect.NONE
        ),
    )


def _decision_status(
    derivation: FindingDerivation,
) -> DecisionStatus:
    requirement_uncertainty = any(
        uncertainty.code.startswith("requirement.")
        for uncertainty in derivation.uncertainties
    )
    if not requirement_uncertainty:
        return DecisionStatus.COMPLETE
    if (
        derivation.triggered_findings
        or derivation.positive_findings
    ):
        return DecisionStatus.PARTIAL
    return DecisionStatus.INSUFFICIENT_EVIDENCE


def build_call_decision(
    bundle: SignalBundle,
    derivation: FindingDerivation,
) -> CallDecision:
    """Build a decision without averaging, scoring, or LLM arbitration."""
    if bundle.call_id != derivation.call_id:
        raise ValueError(
            "finding derivation and signal bundle call_id values must match"
        )
    trace = _decision_trace(derivation)
    controlling = _controlling_findings(
        derivation,
        trace.controlling_finding_ids,
    )
    decision_status = _decision_status(derivation)
    return CallDecision(
        call_id=bundle.call_id,
        evaluator_version=(
            f"v2-policy-{DECISION_POLICY_VERSION}"
        ),
        domain_profile_id=derivation.profile_id,
        signal_bundle_sha256=signal_bundle_sha256(bundle),
        decision_status=decision_status,
        attention_required=bool(trace.controlling_finding_ids),
        triggered_findings=derivation.triggered_findings,
        positive_findings=derivation.positive_findings,
        recommended_action=_recommended_action(
            controlling,
            decision_status,
        ),
        uncertainties=derivation.uncertainties,
        decision_trace=trace,
        presentation=PresentationSelection(),
        provenance=_source(),
    )
