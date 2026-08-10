import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.routers.calls import _evaluation_summary, _pipeline_progress


def _job(status, stage):
    return SimpleNamespace(status=status, stage=stage)


class PipelineProgressTests(unittest.TestCase):
    def test_active_stage_completes_every_prior_enabled_step(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_ACOUSTIC": "1",
                "EVALUATOR_V2_SHADOW": "1",
            },
        ):
            progress = _pipeline_progress(
                _job("processing", "evaluating_v2_requirements")
            )

        states = {
            stage["id"]: stage["state"]
            for stage in progress["stages"]
        }
        self.assertEqual(states["evaluation"], "completed")
        self.assertEqual(states["evidence"], "completed")
        self.assertEqual(states["requirements"], "active")
        self.assertEqual(states["findings"], "pending")
        self.assertEqual(states["publish"], "pending")
        self.assertEqual(
            progress["current_stage_label"],
            "Assess requirements",
        )

    def test_disabled_optional_stages_are_marked_skipped(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_ACOUSTIC": "0",
                "EVALUATOR_V2_SHADOW": "0",
            },
        ):
            progress = _pipeline_progress(
                _job("processing", "evaluating")
            )

        states = {
            stage["id"]: stage["state"]
            for stage in progress["stages"]
        }
        self.assertEqual(states["acoustic"], "completed")
        self.assertEqual(states["evidence"], "skipped")
        self.assertEqual(states["requirements"], "skipped")
        self.assertEqual(states["findings"], "skipped")
        self.assertEqual(states["decision"], "skipped")
        self.assertEqual(states["presentation"], "skipped")
        self.assertEqual(states["evaluation"], "active")

    def test_success_marks_enabled_pipeline_complete(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_ACOUSTIC": "1",
                "EVALUATOR_V2_SHADOW": "0",
            },
        ):
            progress = _pipeline_progress(_job("succeeded", "done"))

        enabled_states = [
            stage["state"]
            for stage in progress["stages"]
            if stage["state"] != "skipped"
        ]
        self.assertEqual(set(enabled_states), {"completed"})
        self.assertEqual(progress["percent"], 100)

    def test_failure_marks_current_stage(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_ACOUSTIC": "1",
                "EVALUATOR_V2_SHADOW": "0",
            },
        ):
            progress = _pipeline_progress(
                _job("failed", "segmenting")
            )

        current = next(
            stage
            for stage in progress["stages"]
            if stage["id"] == "segments"
        )
        self.assertEqual(current["state"], "failed")


class EvaluationSummaryTests(unittest.TestCase):
    def test_success_uses_evaluator_presentation_not_legacy_scores(self):
        run = SimpleNamespace(
            status="succeeded",
            created_at=None,
            payload={
                "status": "succeeded",
                "evaluator_version": "v2-policy-test",
                "decision": {
                    "decision_status": "complete",
                    "attention_required": True,
                },
                "presentation": {
                    "state": "needs_attention",
                    "evaluation_status": "complete",
                    "attention_required": True,
                    "manager_questions": [
                        {
                            "question_id": "call.request",
                            "answer": "yes",
                            "summary": "Request confirmed.",
                            "evidence_ids": ["evidence-request"],
                        },
                        {
                            "question_id": "call.process",
                            "answer": "partly",
                            "summary": "One process concern.",
                            "evidence_ids": ["evidence-process"],
                        },
                        {
                            "question_id": "call.experience",
                            "answer": "yes",
                            "summary": "Experience handled.",
                            "evidence_ids": [],
                        },
                        {
                            "question_id": "call.outcome",
                            "answer": "yes",
                            "summary": "Outcome confirmed.",
                            "evidence_ids": ["evidence-outcome"],
                        },
                    ],
                    "primary_reasons": [{"finding_id": "finding-a"}],
                    "additional_reason_count": 1,
                    "checklist": [
                        {"status": "demonstrated"},
                        {"status": "incorrect"},
                        {"status": "not_demonstrated"},
                    ],
                    "acoustic_context": {
                        "status": "limited",
                        "coverage_label": "Audio support on 2 of 3 segments",
                    },
                },
            },
        )

        summary = _evaluation_summary("call-a", "banking", run)

        self.assertTrue(summary["evaluation_available"])
        self.assertTrue(summary["attention_required"])
        self.assertEqual(summary["evaluation_state"], "needs_attention")
        self.assertEqual(summary["result"], "review")
        self.assertEqual(summary["aspects"]["request"]["state"], "ok")
        self.assertEqual(summary["aspects"]["process"]["state"], "concern")
        self.assertEqual(summary["concern_count"], 2)
        self.assertEqual(summary["checklist_counts"]["demonstrated"], 1)
        self.assertEqual(summary["checklist_counts"]["incorrect"], 1)
        self.assertEqual(
            summary["checklist_counts"]["not_demonstrated"],
            1,
        )
        self.assertEqual(summary["acoustic_status"], "limited")

    def test_domain_without_profile_is_not_reported_as_evaluated(self):
        summary = _evaluation_summary(
            "call-b",
            "health",
            SimpleNamespace(
                status="unsupported_domain",
                created_at=None,
                payload={"status": "unsupported_domain"},
            ),
        )

        self.assertFalse(summary["evaluation_available"])
        self.assertTrue(summary["evaluation_supported"])
        self.assertEqual(summary["evaluation_state"], "unsupported")


if __name__ == "__main__":
    unittest.main()
