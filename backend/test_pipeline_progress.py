import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.routers.calls import _pipeline_progress


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
                _job("processing", "evaluating_v2_shadow")
            )

        states = {
            stage["id"]: stage["state"]
            for stage in progress["stages"]
        }
        self.assertEqual(states["evaluation"], "completed")
        self.assertEqual(states["evaluation_v2"], "active")
        self.assertEqual(states["publish"], "pending")
        self.assertEqual(progress["current_stage_label"], "V2 shadow")

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
        self.assertEqual(states["acoustic"], "skipped")
        self.assertEqual(states["evaluation_v2"], "skipped")
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


if __name__ == "__main__":
    unittest.main()
