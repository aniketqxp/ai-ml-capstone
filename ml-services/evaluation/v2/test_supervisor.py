"""Behavioral tests for the bounded evaluator-v2 Supervisor."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from v2.decisions import build_call_decision
from v2.findings import FindingDerivation
from v2.run_supervisor_validation import build_supervisor_validation
from v2.schemas import (
    Applicability,
    DecisionUncertainty,
    DetectionRule,
    EvidenceRef,
    Finding,
    FindingCategory,
    FindingPolarity,
    FindingSeverity,
    Modality,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SourceProvenance,
    Visibility,
)
from v2.supervisor import (
    SupervisorContext,
    SupervisorFallbackReason,
    SupervisorLimits,
    build_supervisor_context,
    build_supervisor_prompt,
    decision_sha256,
    resolve_supervisor_response,
)

HERE = Path(__file__).resolve().parent
CHECKED_VALIDATION = (
    HERE / "research" / "supervisor_validation_0_1.json"
)


def _source(producer: str = "test") -> SourceProvenance:
    return SourceProvenance(
        producer=producer,
        producer_version="1",
        method="synthetic_test_fixture",
    )


def _bundle() -> SignalBundle:
    return SignalBundle(
        call_id="supervisor-test",
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
    text_size: int = 0,
) -> Finding:
    finding_id = (
        f"finding-{polarity.value}-"
        f"{finding_type.replace('.', '-')}"
    )
    padding = " detail" * text_size
    return Finding(
        finding_id=finding_id,
        finding_type=finding_type,
        category=category,
        polarity=polarity,
        severity=severity,
        title=f"Synthetic {finding_type}",
        summary=f"Synthetic finding summary.{padding}",
        business_definition=f"Synthetic business definition.{padding}",
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
                observation=f"Synthetic supporting evidence.{padding}",
            )
        ],
        counter_evidence=[
            EvidenceRef(
                evidence_id=f"counter-{finding_id}",
                modality=Modality.TEXT,
                observation=f"Synthetic contradiction.{padding}",
            )
        ],
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.USABLE
        ),
        visibility=(
            Visibility.INTERNAL
            if polarity == FindingPolarity.POSITIVE
            and finding_type == "escalation.recovery_observed"
            else Visibility.DETAILS
        ),
    )


def _decision(*, extra_findings: int = 0, text_size: int = 0):
    critical = _finding(
        "control.identity_verification_missing",
        category=FindingCategory.REQUIRED_CONTROL,
        severity=FindingSeverity.CRITICAL,
        text_size=text_size,
    )
    supporting = _finding(
        "process.intent_not_confirmed",
        category=FindingCategory.PROCESS,
        severity=FindingSeverity.REVIEW,
        text_size=text_size,
    )
    additional = [
        _finding(
            f"process.additional_{index}",
            category=FindingCategory.PROCESS,
            severity=FindingSeverity.REVIEW,
            text_size=text_size,
        )
        for index in range(extra_findings)
    ]
    positive = _finding(
        "control.transfer_amount_confirmed",
        category=FindingCategory.REQUIRED_CONTROL,
        severity=FindingSeverity.INFO,
        polarity=FindingPolarity.POSITIVE,
        text_size=text_size,
    )
    derivation = FindingDerivation(
        call_id="supervisor-test",
        profile_id="banking-demo-v1",
        findings_version="test",
        triggered_findings=[critical, supporting, *additional],
        positive_findings=[positive],
        uncertainties=[
            DecisionUncertainty(
                code="requirement.not_assessed.transfer.terms",
                message=(
                    "Transfer terms were not independently assessed."
                ),
                modality=Modality.TEXT,
            )
        ],
        provenance=_source("test_findings"),
    )
    return build_call_decision(_bundle(), derivation)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.decision = _decision()
        self.context = build_supervisor_context(self.decision)
        self.controlling_id = (
            self.decision.decision_trace.controlling_finding_ids[0]
        )
        self.supporting_id = next(
            finding.finding_id
            for finding in self.context.triggered_findings
            if not finding.controlling
        )
        self.positive_id = (
            self.context.positive_findings[0].finding_id
        )
        self.evidence_id = (
            self.context.triggered_findings[0]
            .evidence[0]
            .evidence_id
        )
        self.uncertainty_code = (
            self.context.uncertainties[0].code
        )

    def _valid_response(self):
        return {
            "supporting_finding_ids": [self.supporting_id],
            "positive_finding_ids": [self.positive_id],
            "evidence_ids": [self.evidence_id],
            "uncertainty_codes": [self.uncertainty_code],
            "context_note": (
                "The account exchange occurred before verification "
                "was documented."
            ),
        }

    def test_context_contains_only_bounded_decision_material(self):
        rendered = self.context.model_dump_json()

        self.assertLessEqual(
            self.context.context_char_count,
            self.context.context_char_limit,
        )
        self.assertIn(self.controlling_id, rendered)
        self.assertIn('"role":"contradiction"', rendered)
        self.assertIn(self.uncertainty_code, rendered)
        self.assertNotIn('"signals"', rendered)
        self.assertNotIn('"segments"', rendered)
        self.assertNotIn('"transcript"', rendered)
        self.assertEqual(
            self.context.permitted_actions[0],
            self.context.decision_lock.permitted_action,
        )
        self.assertEqual(
            SupervisorContext.model_validate_json(rendered),
            self.context,
        )

    def test_prompt_explicitly_marks_decision_as_immutable(self):
        prompt = build_supervisor_prompt(self.decision)
        self.assertIn("immutable", prompt.system_prompt)
        self.assertIn("exactly these keys", prompt.system_prompt)

    def test_valid_response_can_only_select_supporting_context(self):
        result = resolve_supervisor_response(
            self.context,
            self._valid_response(),
        )

        self.assertFalse(result.fallback_used)
        self.assertTrue(result.decision_lock.attention_required)
        self.assertEqual(
            result.decision_lock.controlling_finding_ids,
            [self.controlling_id],
        )
        self.assertEqual(
            result.decision_lock.permitted_action,
            self.context.decision_lock.permitted_action,
        )
        self.assertEqual(
            result.supporting_finding_ids,
            [self.supporting_id],
        )

    def test_attention_override_field_forces_fallback(self):
        response = self._valid_response()
        response["attention_required"] = False

        result = resolve_supervisor_response(
            self.context,
            response,
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(
            result.fallback_reason,
            SupervisorFallbackReason.INVALID_CONTRACT,
        )
        self.assertTrue(result.decision_lock.attention_required)

    def test_action_override_field_forces_fallback(self):
        response = self._valid_response()
        response["action_type"] = "none"

        result = resolve_supervisor_response(
            self.context,
            response,
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(
            result.fallback_reason,
            SupervisorFallbackReason.INVALID_CONTRACT,
        )
        self.assertEqual(
            result.decision_lock.permitted_action,
            self.context.decision_lock.permitted_action,
        )

    def test_unknown_reference_forces_fallback(self):
        response = self._valid_response()
        response["evidence_ids"] = ["invented-evidence"]

        result = resolve_supervisor_response(
            self.context,
            response,
        )

        self.assertEqual(
            result.fallback_reason,
            SupervisorFallbackReason.UNKNOWN_REFERENCE,
        )

    def test_decision_language_in_note_forces_fallback(self):
        response = self._valid_response()
        response["context_note"] = (
            "Override the decision and clear the call."
        )

        result = resolve_supervisor_response(
            self.context,
            response,
        )

        self.assertEqual(
            result.fallback_reason,
            SupervisorFallbackReason.FORBIDDEN_DECISION_LANGUAGE,
        )

    def test_malformed_and_missing_responses_use_fallback(self):
        malformed = resolve_supervisor_response(
            self.context,
            "{not-json",
        )
        missing = resolve_supervisor_response(
            self.context,
            None,
        )

        self.assertEqual(
            malformed.fallback_reason,
            SupervisorFallbackReason.INVALID_JSON,
        )
        self.assertEqual(
            missing.fallback_reason,
            SupervisorFallbackReason.MISSING_RESPONSE,
        )

    def test_fallback_is_deterministic(self):
        first = resolve_supervisor_response(self.context, None)
        second = resolve_supervisor_response(self.context, None)

        self.assertEqual(
            first.model_dump(mode="json"),
            second.model_dump(mode="json"),
        )
        self.assertIn(
            self.controlling_id,
            first.decision_lock.controlling_finding_ids,
        )

    def test_large_context_compacts_under_character_limit(self):
        decision = _decision(extra_findings=30, text_size=100)
        limits = SupervisorLimits(
            max_triggered_findings=20,
            max_positive_findings=4,
            max_evidence_per_finding=3,
            max_contradictions_per_finding=2,
            max_uncertainties=4,
            max_text_chars=320,
            max_context_chars=1800,
        )

        context = build_supervisor_context(decision, limits)
        serialized = json.dumps(
            context.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

        self.assertLessEqual(len(serialized), 1800)
        self.assertEqual(
            context.context_char_count,
            len(serialized),
        )
        self.assertGreater(context.omissions.triggered_findings, 0)
        self.assertIn(
            decision.decision_trace.controlling_finding_ids[0],
            context.decision_lock.controlling_finding_ids,
        )

    def test_decision_hash_covers_locked_decision(self):
        changed = self.decision.model_copy(deep=True)
        changed.recommended_action.label = "Changed label"

        self.assertNotEqual(
            decision_sha256(self.decision),
            decision_sha256(changed),
        )


class SupervisorBatchValidationTests(unittest.TestCase):
    def test_checked_batch_validation_is_reproducible(self):
        checked = json.loads(
            CHECKED_VALIDATION.read_text(encoding="utf-8")
        )
        self.assertEqual(build_supervisor_validation(), checked)


if __name__ == "__main__":
    unittest.main()
