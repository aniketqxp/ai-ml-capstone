"""Behavioral tests for deterministic attention and action policy."""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from v2.decisions import (
    build_call_decision,
    signal_bundle_sha256,
)
from v2.findings import FindingDerivation
from v2.schemas import (
    ActionExecution,
    ActionType,
    Applicability,
    CallDecision,
    DecisionStatus,
    DecisionUncertainty,
    DetectionRule,
    EvidenceRef,
    Finding,
    FindingCategory,
    FindingPolarity,
    FindingSeverity,
    Modality,
    QualificationReason,
    RecoveryEffect,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SourceProvenance,
    Visibility,
)
from v2.validation import validate_decision_references


def _source(producer: str = "test") -> SourceProvenance:
    return SourceProvenance(
        producer=producer,
        producer_version="1",
        method="synthetic_test_fixture",
    )


def _bundle(call_id: str = "test-call") -> SignalBundle:
    return SignalBundle(
        call_id=call_id,
        domain="banking",
        duration_seconds=0.0,
        segments=[],
        sources=[_source()],
    )


def _finding(
    finding_type: str,
    *,
    category: FindingCategory,
    severity: FindingSeverity,
    polarity: FindingPolarity = FindingPolarity.NEGATIVE,
    reliability: ReliabilityStatus = ReliabilityStatus.USABLE,
    visibility: Visibility = Visibility.DETAILS,
) -> Finding:
    finding_id = (
        f"finding-{polarity.value}-"
        f"{finding_type.replace('.', '-')}"
    )
    return Finding(
        finding_id=finding_id,
        finding_type=finding_type,
        category=category,
        polarity=polarity,
        severity=severity,
        title=f"Synthetic {finding_type}",
        summary="Synthetic finding for decision-policy tests.",
        business_definition="Synthetic business definition.",
        applicability=Applicability(
            applies=True,
            profile_id="banking-demo-v1",
            rule_id=finding_type,
            reason="Synthetic applicable rule.",
        ),
        detection_rule=DetectionRule(
            rule_id=finding_type,
            rule_version="1",
            detector="synthetic",
        ),
        evidence=[
            EvidenceRef(
                evidence_id=f"evidence-{finding_id}",
                modality=Modality.TEXT,
                observation="Synthetic supporting evidence.",
            )
        ],
        counter_evidence=[
            EvidenceRef(
                evidence_id=f"counter-{finding_id}",
                modality=Modality.TEXT,
                observation="Synthetic counter-evidence.",
            )
        ],
        reliability=ReliabilityAssessment(status=reliability),
        visibility=visibility,
    )


def _derivation(
    negative: list[Finding] | None = None,
    positive: list[Finding] | None = None,
    *,
    call_id: str = "test-call",
    uncertainties: list[DecisionUncertainty] | None = None,
) -> FindingDerivation:
    return FindingDerivation(
        call_id=call_id,
        profile_id="banking-demo-v1",
        findings_version="test",
        triggered_findings=negative or [],
        positive_findings=positive or [],
        uncertainties=uncertainties or [],
        provenance=_source("test_findings"),
    )


class AttentionPolicyTests(unittest.TestCase):
    def test_no_findings_means_no_attention_and_no_action(self):
        decision = build_call_decision(
            _bundle(),
            _derivation(),
        )

        self.assertFalse(decision.attention_required)
        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.NONE,
        )
        self.assertEqual(
            decision.decision_trace.aggregation,
            "any_qualifying_finding",
        )
        self.assertEqual(
            decision.decision_status,
            DecisionStatus.COMPLETE,
        )

    def test_assessment_gaps_do_not_masquerade_as_a_clear_call(self):
        uncertainty = DecisionUncertainty(
            code="requirement.not_assessed.transfer.amount",
            message="Transfer amount was not assessed.",
            modality=Modality.TEXT,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation(uncertainties=[uncertainty]),
        )

        self.assertFalse(decision.attention_required)
        self.assertEqual(
            decision.decision_status,
            DecisionStatus.INSUFFICIENT_EVIDENCE,
        )
        self.assertIn(
            "not been cleared",
            decision.recommended_action.reason,
        )

    def test_finding_with_assessment_gap_is_partial(self):
        finding = _finding(
            "outcome.next_steps_unclear",
            category=FindingCategory.OUTCOME,
            severity=FindingSeverity.REVIEW,
        )
        uncertainty = DecisionUncertainty(
            code="requirement.not_assessed.transfer.amount",
            message="Transfer amount was not assessed.",
            modality=Modality.TEXT,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation(
                [finding],
                uncertainties=[uncertainty],
            ),
        )

        self.assertTrue(decision.attention_required)
        self.assertEqual(
            decision.decision_status,
            DecisionStatus.PARTIAL,
        )

    def test_critical_control_emails_manager_review(self):
        finding = _finding(
            "control.identity_verification_missing",
            category=FindingCategory.REQUIRED_CONTROL,
            severity=FindingSeverity.CRITICAL,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([finding]),
        )

        self.assertTrue(decision.attention_required)
        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.MANAGER_REVIEW,
        )
        self.assertEqual(
            decision.recommended_action.execution,
            ActionExecution.AUTOMATIC,
        )
        self.assertTrue(
            decision.recommended_action.automation_allowed
        )

    def test_review_outcome_requires_follow_up_approval(self):
        finding = _finding(
            "outcome.next_steps_unclear",
            category=FindingCategory.OUTCOME,
            severity=FindingSeverity.REVIEW,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([finding]),
        )

        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.CUSTOMER_FOLLOW_UP,
        )
        self.assertEqual(
            decision.recommended_action.execution,
            ActionExecution.REQUIRES_APPROVAL,
        )
        self.assertTrue(
            decision.recommended_action.requires_human_approval
        )

    def test_review_process_requires_coaching_approval(self):
        finding = _finding(
            "process.intent_not_confirmed",
            category=FindingCategory.PROCESS,
            severity=FindingSeverity.REVIEW,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([finding]),
        )

        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.AGENT_COACHING,
        )
        self.assertEqual(
            decision.recommended_action.execution,
            ActionExecution.REQUIRES_APPROVAL,
        )

    def test_repeat_contact_requires_follow_up_approval(self):
        finding = _finding(
            "process.repeat_contact_unresolved",
            category=FindingCategory.PROCESS,
            severity=FindingSeverity.REVIEW,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([finding]),
        )

        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.CUSTOMER_FOLLOW_UP,
        )
        self.assertEqual(
            decision.recommended_action.execution,
            ActionExecution.REQUIRES_APPROVAL,
        )

    def test_review_escalation_emails_manager_review(self):
        finding = _finding(
            "escalation.explicit_dissatisfaction",
            category=FindingCategory.ESCALATION,
            severity=FindingSeverity.REVIEW,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([finding]),
        )

        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.MANAGER_REVIEW,
        )
        self.assertEqual(
            decision.recommended_action.execution,
            ActionExecution.AUTOMATIC,
        )

    def test_informational_internal_and_unreliable_findings_are_excluded(self):
        findings = [
            _finding(
                "process.informational",
                category=FindingCategory.PROCESS,
                severity=FindingSeverity.INFO,
            ),
            _finding(
                "process.internal",
                category=FindingCategory.PROCESS,
                severity=FindingSeverity.CRITICAL,
                visibility=Visibility.INTERNAL,
            ),
            _finding(
                "process.unreliable",
                category=FindingCategory.PROCESS,
                severity=FindingSeverity.CRITICAL,
                reliability=ReliabilityStatus.UNAVAILABLE,
            ),
        ]

        decision = build_call_decision(
            _bundle(),
            _derivation(findings),
        )
        reasons = {
            item.finding_id: item.reason
            for item in decision.decision_trace.qualifications
        }

        self.assertFalse(decision.attention_required)
        self.assertEqual(
            reasons[findings[0].finding_id],
            QualificationReason.EXCLUDED_INFORMATIONAL,
        )
        self.assertEqual(
            reasons[findings[1].finding_id],
            QualificationReason.EXCLUDED_INTERNAL,
        )
        self.assertEqual(
            reasons[findings[2].finding_id],
            QualificationReason.EXCLUDED_UNRELIABLE,
        )

    def test_one_qualifying_finding_is_not_averaged_away(self):
        qualifying = _finding(
            "outcome.next_steps_unclear",
            category=FindingCategory.OUTCOME,
            severity=FindingSeverity.REVIEW,
        )
        informational = [
            _finding(
                f"process.information_{index}",
                category=FindingCategory.PROCESS,
                severity=FindingSeverity.INFO,
            )
            for index in range(10)
        ]
        positives = [
            _finding(
                f"control.positive_{index}",
                category=FindingCategory.REQUIRED_CONTROL,
                severity=FindingSeverity.INFO,
                polarity=FindingPolarity.POSITIVE,
            )
            for index in range(10)
        ]

        decision = build_call_decision(
            _bundle(),
            _derivation(
                [qualifying, *informational],
                positives,
            ),
        )

        self.assertTrue(decision.attention_required)
        self.assertEqual(
            decision.decision_trace.controlling_finding_ids,
            [qualifying.finding_id],
        )

    def test_critical_escalation_controls_review_outcome(self):
        outcome = _finding(
            "outcome.next_steps_unclear",
            category=FindingCategory.OUTCOME,
            severity=FindingSeverity.REVIEW,
        )
        escalation = _finding(
            "escalation.manager_requested",
            category=FindingCategory.ESCALATION,
            severity=FindingSeverity.CRITICAL,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([outcome, escalation]),
        )

        self.assertEqual(
            decision.decision_trace.controlling_finding_ids,
            [escalation.finding_id],
        )
        self.assertEqual(
            decision.recommended_action.finding_ids,
            [escalation.finding_id],
        )
        self.assertEqual(
            decision.recommended_action.action_type,
            ActionType.MANAGER_REVIEW,
        )

    def test_recovery_is_context_and_does_not_cancel_attention(self):
        escalation = _finding(
            "escalation.manager_requested",
            category=FindingCategory.ESCALATION,
            severity=FindingSeverity.CRITICAL,
        )
        recovery = _finding(
            "escalation.recovery_observed",
            category=FindingCategory.ESCALATION,
            severity=FindingSeverity.INFO,
            polarity=FindingPolarity.POSITIVE,
            visibility=Visibility.INTERNAL,
            reliability=ReliabilityStatus.LIMITED,
        )

        decision = build_call_decision(
            _bundle(),
            _derivation([escalation], [recovery]),
        )

        self.assertTrue(decision.attention_required)
        self.assertEqual(
            decision.decision_trace.recovery_effect,
            RecoveryEffect.CONTEXT_ONLY,
        )
        self.assertEqual(
            decision.decision_trace.recovery_finding_ids,
            [recovery.finding_id],
        )


class DecisionIntegrityTests(unittest.TestCase):
    def test_bundle_hash_is_deterministic_and_recorded(self):
        bundle = _bundle()

        first = build_call_decision(bundle, _derivation())
        second = build_call_decision(bundle, _derivation())

        self.assertEqual(
            first.signal_bundle_sha256,
            signal_bundle_sha256(bundle),
        )
        self.assertEqual(
            first.signal_bundle_sha256,
            second.signal_bundle_sha256,
        )

    def test_mismatched_call_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "call_id"):
            build_call_decision(
                _bundle("bundle-call"),
                _derivation(call_id="finding-call"),
            )

    def test_attention_cannot_contradict_policy_trace(self):
        finding = _finding(
            "outcome.next_steps_unclear",
            category=FindingCategory.OUTCOME,
            severity=FindingSeverity.REVIEW,
        )
        decision = build_call_decision(
            _bundle(),
            _derivation([finding]),
        )
        payload = decision.model_dump()
        payload["attention_required"] = False

        with self.assertRaisesRegex(
            ValidationError,
            "attention must equal",
        ):
            CallDecision.model_validate(payload)

    def test_action_cannot_cite_noncontrolling_finding(self):
        outcome = _finding(
            "outcome.next_steps_unclear",
            category=FindingCategory.OUTCOME,
            severity=FindingSeverity.REVIEW,
        )
        escalation = _finding(
            "escalation.manager_requested",
            category=FindingCategory.ESCALATION,
            severity=FindingSeverity.CRITICAL,
        )
        decision = build_call_decision(
            _bundle(),
            _derivation([outcome, escalation]),
        )
        payload = decision.model_dump()
        payload["recommended_action"]["finding_ids"] = [
            outcome.finding_id
        ]

        with self.assertRaisesRegex(
            ValidationError,
            "controlling findings",
        ):
            CallDecision.model_validate(payload)

    def test_cross_artifact_validation_checks_bundle_hash(self):
        bundle = _bundle()
        decision = build_call_decision(bundle, _derivation())
        changed = bundle.model_copy(deep=True)
        changed.duration_seconds = 1.0

        with self.assertRaisesRegex(ValueError, "hash"):
            validate_decision_references(decision, changed)


if __name__ == "__main__":
    unittest.main()
