from __future__ import annotations

import unittest

from v2.decisions import build_call_decision
from v2.findings import FindingDerivation
from v2.presentation import (
    MAX_EVIDENCE_ITEMS,
    OUTPUT_INVENTORY,
    PageState,
    Representation,
    check_presentation,
    project_call_evaluation,
)
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
    Speaker,
    Visibility,
)
from v2.supervisor import (
    build_supervisor_context,
    resolve_supervisor_response,
)


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="presentation_test",
        producer_version="1",
    )


def _finding(
    finding_id: str,
    *,
    polarity: FindingPolarity,
    category: FindingCategory,
    severity: FindingSeverity,
    priority: int,
    visibility: Visibility = Visibility.DETAILS,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        finding_type=f"test.{finding_id.replace('-', '_')}",
        category=category,
        polarity=polarity,
        severity=severity,
        title=f"Title {finding_id}",
        summary=f"Summary {finding_id}.",
        business_definition="Test business definition.",
        applicability=Applicability(
            applies=True,
            profile_id="banking-demo-v1",
            rule_id="test-rule",
            reason="Applicable test case.",
        ),
        detection_rule=DetectionRule(
            rule_id="test-rule",
            rule_version="1",
            detector="test",
        ),
        evidence=[
            EvidenceRef(
                evidence_id=f"{finding_id}-text",
                modality=Modality.TRANSCRIPT,
                speaker=Speaker.CUSTOMER,
                start_seconds=float(priority),
                end_seconds=float(priority + 1),
                quote=f"Evidence for {finding_id}.",
            ),
            EvidenceRef(
                evidence_id=f"{finding_id}-audio",
                modality=Modality.ACOUSTIC,
                speaker=Speaker.CUSTOMER,
                start_seconds=float(priority),
                end_seconds=float(priority + 1),
                observation="Audio supports the transcript finding.",
            ),
        ],
        counter_evidence=[
            EvidenceRef(
                evidence_id=f"{finding_id}-counter",
                modality=Modality.TRANSCRIPT,
                speaker=Speaker.AGENT,
                start_seconds=float(priority + 1),
                end_seconds=float(priority + 2),
                quote=f"Counter-evidence for {finding_id}.",
            )
        ],
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.USABLE
        ),
        visibility=visibility,
        display_priority=priority,
    )


def _decision(
    *,
    negative: list[Finding] | None = None,
    positive: list[Finding] | None = None,
    uncertainties: list[DecisionUncertainty] | None = None,
):
    bundle = SignalBundle(
        call_id="presentation-test",
        domain="banking",
        duration_seconds=20,
        segments=[],
        sources=[_source()],
    )
    derivation = FindingDerivation(
        call_id=bundle.call_id,
        profile_id="banking-demo-v1",
        findings_version="test",
        triggered_findings=negative or [],
        positive_findings=positive or [],
        uncertainties=uncertainties or [],
        provenance=_source(),
    )
    return build_call_decision(bundle, derivation)


class PresentationProjectionTests(unittest.TestCase):
    def test_complete_call_avoids_pass_language_and_action(self):
        view = project_call_evaluation(_decision())

        self.assertEqual(view.state, PageState.NO_ATTENTION_FINDING)
        self.assertEqual(view.headline, "No review finding identified")
        self.assertNotIn("pass", view.headline.lower())
        self.assertIsNone(view.recommended_action)
        self.assertIsNone(view.completeness_notice)
        self.assertTrue(check_presentation(view).passed)

    def test_incomplete_call_is_not_cleared(self):
        uncertainty = DecisionUncertainty(
            code="requirement.not_assessed.identity",
            message="Identity verification was not assessed.",
            modality=Modality.TEXT,
            visibility=Visibility.INTERNAL,
        )
        view = project_call_evaluation(
            _decision(uncertainties=[uncertainty])
        )

        self.assertEqual(view.state, PageState.EVALUATION_INCOMPLETE)
        self.assertIn("not been cleared", view.completeness_notice.message)
        self.assertIsNone(view.recommended_action)
        self.assertTrue(check_presentation(view).passed)

    def test_partial_attention_keeps_finding_and_incomplete_notice(self):
        finding = _finding(
            "negative-partial",
            polarity=FindingPolarity.NEGATIVE,
            category=FindingCategory.PROCESS,
            severity=FindingSeverity.REVIEW,
            priority=1,
        )
        uncertainty = DecisionUncertainty(
            code="requirement.assessment_uncertain.authorization",
            message="Authorization could not be determined.",
            modality=Modality.TEXT,
            visibility=Visibility.DETAILS,
        )
        view = project_call_evaluation(
            _decision(
                negative=[finding],
                uncertainties=[uncertainty],
            )
        )

        self.assertEqual(view.state, PageState.NEEDS_ATTENTION)
        self.assertIsNotNone(view.recommended_action)
        self.assertIsNotNone(view.completeness_notice)
        self.assertTrue(check_presentation(view).passed)

    def test_projection_bounds_and_deduplicates_content(self):
        negative = [
            _finding(
                f"negative-{index}",
                polarity=FindingPolarity.NEGATIVE,
                category=FindingCategory.REQUIRED_CONTROL,
                severity=FindingSeverity.CRITICAL,
                priority=index,
            )
            for index in range(4)
        ]
        positive = [
            _finding(
                f"positive-{index}",
                polarity=FindingPolarity.POSITIVE,
                category=FindingCategory.OUTCOME,
                severity=FindingSeverity.INFO,
                priority=20 + index,
            )
            for index in range(4)
        ]
        view = project_call_evaluation(
            _decision(negative=negative, positive=positive)
        )

        self.assertEqual(view.state, PageState.NEEDS_ATTENTION)
        self.assertEqual(len(view.primary_reasons), 2)
        self.assertEqual(view.additional_reason_count, 2)
        self.assertEqual(len(view.positive_highlights), 2)
        self.assertEqual(view.additional_positive_count, 2)
        self.assertLessEqual(len(view.evidence), MAX_EVIDENCE_ITEMS)
        self.assertIsNotNone(view.recommended_action)
        self.assertTrue(check_presentation(view).passed)

    def test_audio_is_marked_as_supporting_only(self):
        finding = _finding(
            "negative-audio",
            polarity=FindingPolarity.NEGATIVE,
            category=FindingCategory.ESCALATION,
            severity=FindingSeverity.REVIEW,
            priority=1,
        )
        view = project_call_evaluation(_decision(negative=[finding]))

        audio = [
            item for item in view.evidence
            if item.kind.value == "audio_support"
        ]
        self.assertEqual(len(audio), 1)
        self.assertTrue(audio[0].supporting_only)

    def test_supervisor_can_order_context_but_not_decision(self):
        negative = [
            _finding(
                "negative-control",
                polarity=FindingPolarity.NEGATIVE,
                category=FindingCategory.REQUIRED_CONTROL,
                severity=FindingSeverity.CRITICAL,
                priority=1,
            ),
            _finding(
                "negative-support",
                polarity=FindingPolarity.NEGATIVE,
                category=FindingCategory.PROCESS,
                severity=FindingSeverity.REVIEW,
                priority=2,
            ),
        ]
        positive = [
            _finding(
                "positive-handling",
                polarity=FindingPolarity.POSITIVE,
                category=FindingCategory.AGENT_BEHAVIOR,
                severity=FindingSeverity.INFO,
                priority=3,
            )
        ]
        decision = _decision(negative=negative, positive=positive)
        context = build_supervisor_context(decision)
        result = resolve_supervisor_response(
            context,
            {
                "supporting_finding_ids": ["negative-support"],
                "positive_finding_ids": ["positive-handling"],
                "evidence_ids": ["negative-control-text"],
                "uncertainty_codes": [],
                "context_note": "The customer repeated the request.",
            },
        )
        view = project_call_evaluation(decision, result)

        self.assertTrue(view.attention_required)
        self.assertEqual(
            view.recommended_action.action_type,
            decision.recommended_action.action_type,
        )
        self.assertEqual(
            view.details.supervisor_context_note,
            "The customer repeated the request.",
        )

        changed = decision.model_copy(deep=True)
        changed.call_id = "different-call"
        with self.assertRaisesRegex(ValueError, "does not match"):
            project_call_evaluation(changed, result)

    def test_inventory_excludes_composite_scores(self):
        composite = next(
            item
            for item in OUTPUT_INVENTORY
            if item.output_id == "scores.composites"
        )
        self.assertEqual(composite.visibility, Visibility.INTERNAL)
        self.assertEqual(
            composite.representation,
            Representation.NOT_RENDERED,
        )


if __name__ == "__main__":
    unittest.main()
