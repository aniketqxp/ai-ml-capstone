from __future__ import annotations

import unittest

from v2.runtime import (
    ShadowRunStatus,
    legacy_attention_proxy,
    run_shadow_evaluation,
    safe_run_shadow_evaluation,
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


class ShadowRuntimeTests(unittest.TestCase):
    def test_known_banking_call_runs_without_fabricating_assessments(self):
        run = run_shadow_evaluation(transcript=_transcript())

        self.assertEqual(run.status, ShadowRunStatus.SUCCEEDED)
        self.assertIsNotNone(run.decision)
        self.assertIsNotNone(run.presentation)
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
