from __future__ import annotations

import unittest
from unittest.mock import patch

from v2.findings import (
    RequirementAssessment,
    RequirementAssessmentBatch,
    RequirementVerdict,
)
from v2.runtime import (
    ShadowRunStatus,
    legacy_attention_proxy,
    run_shadow_evaluation,
    safe_run_shadow_evaluation,
)
from v2.schemas import (
    EvidenceRef,
    Modality,
    ReliabilityAssessment,
    ReliabilityStatus,
    SourceProvenance,
)


def _transcript(
    call_id: str = "en_CA_Banking_1586889",
    domain: str = "banking",
) -> dict:
    return {
        "call_id": call_id,
        "domain": domain,
        "accent": "en-CA",
        "model": "test",
        "sentences": [
            {
                "id": "AGENT_001",
                "seq_id": 1,
                "speaker": "AGENT",
                "start": 0.0,
                "end": 2.0,
                "text": "Thank you for calling. My name is Emily.",
            },
            {
                "id": "CUSTOMER_001",
                "seq_id": 1,
                "speaker": "CUSTOMER",
                "start": 2.1,
                "end": 4.0,
                "text": "Please set up a monthly transfer.",
            },
        ],
    }


def _assessments(bundle, plan):
    segment = bundle.segments[0]
    provenance = SourceProvenance(
        producer="test",
        producer_version="1",
        method="runtime_test",
    )
    return RequirementAssessmentBatch(
        call_id=bundle.call_id,
        profile_id=plan.profile_id,
        assessments=[
            RequirementAssessment(
                assessment_id=f"assessment:{requirement.requirement_id}",
                requirement_id=requirement.requirement_id,
                verdict=RequirementVerdict.MET,
                rationale="The requirement is demonstrated in the test.",
                evidence=[
                    EvidenceRef(
                        evidence_id=(
                            f"evidence:{requirement.requirement_id}"
                        ),
                        modality=Modality.TRANSCRIPT,
                        segment_ids=[segment.segment_id],
                        speaker=segment.speaker,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        quote=segment.text,
                    )
                ],
                reliability=ReliabilityAssessment(
                    status=ReliabilityStatus.LIMITED,
                    coverage_ratio=1.0,
                    reasons=["test"],
                ),
                provenance=provenance,
            )
            for requirement in plan.requirements
        ],
        provenance=provenance,
    )


class ShadowRuntimeTests(unittest.TestCase):
    @patch("v2.runtime.assess_requirements", side_effect=_assessments)
    def test_known_banking_call_runs_semantic_assessments(self, assessor):
        stages = []
        run = run_shadow_evaluation(
            transcript=_transcript(),
            progress=stages.append,
        )

        self.assertEqual(run.status, ShadowRunStatus.SUCCEEDED)
        self.assertIsNotNone(run.decision)
        self.assertIsNotNone(run.presentation)
        self.assertEqual(
            run.decision.decision_status.value,
            "complete",
        )
        self.assertNotIn(
            "semantic_requirement_assessments_missing",
            run.limitations,
        )
        assessor.assert_called_once()
        self.assertEqual(
            stages,
            [
                "evaluating_v2_prepare",
                "evaluating_v2_signals",
                "evaluating_v2_requirements",
                "evaluating_v2_findings",
                "evaluating_v2_decision",
                "evaluating_v2_presentation",
            ],
        )

    def test_semantic_assessment_can_be_explicitly_disabled(self):
        run = run_shadow_evaluation(
            transcript=_transcript(),
            run_semantic_assessor=False,
        )

        self.assertEqual(
            run.decision.decision_status.value,
            "insufficient_evidence",
        )
        self.assertIn(
            "semantic_requirement_assessments_missing",
            run.limitations,
        )

    def test_unsupported_domain_is_recorded(self):
        run = run_shadow_evaluation(
            transcript=_transcript(domain="health")
        )

        self.assertEqual(
            run.status,
            ShadowRunStatus.UNSUPPORTED_DOMAIN,
        )
        self.assertIsNone(run.decision)

    def test_unknown_banking_call_requires_profile_selection(self):
        run = run_shadow_evaluation(
            transcript=_transcript(call_id="new-banking-call")
        )

        self.assertEqual(
            run.status,
            ShadowRunStatus.PROFILE_SELECTION_UNAVAILABLE,
        )

    def test_incomplete_v2_is_not_compared_to_legacy_attention(self):
        legacy = {
            "rubric_version": "0.4.1",
            "compliance": {
                "recording": {"passed": False},
            },
            "escalation": {"risk_level": "none"},
        }
        run = run_shadow_evaluation(
            transcript=_transcript(),
            legacy_evaluation=legacy,
            run_semantic_assessor=False,
        )

        self.assertTrue(run.legacy_proxy.attention_required)
        self.assertFalse(run.comparison.comparable)
        self.assertIsNone(run.comparison.attention_agreement)

    def test_legacy_proxy_names_its_non_equivalent_policy(self):
        proxy = legacy_attention_proxy(
            {
                "compliance": {"one": {"passed": True}},
                "escalation": {"risk_level": "review"},
            }
        )

        self.assertTrue(proxy.attention_required)
        self.assertEqual(
            proxy.basis,
            "failed_compliance_or_escalation",
        )

    def test_safe_runner_records_failures(self):
        run = safe_run_shadow_evaluation(transcript={})

        self.assertEqual(run.status, ShadowRunStatus.FAILED)
        self.assertIn("call_id is required", run.failure_reason)


if __name__ == "__main__":
    unittest.main()
