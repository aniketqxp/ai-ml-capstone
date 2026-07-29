import unittest

from app.routers.calls import EvaluationFeedbackRequest
from pydantic import ValidationError

HASH = "a" * 64


class EvaluationFeedbackRequestTests(unittest.TestCase):
    def test_finding_feedback_requires_finding_id(self):
        with self.assertRaises(ValidationError):
            EvaluationFeedbackRequest(
                feedback_type="dismiss_finding",
                decision_sha256=HASH,
            )

    def test_action_feedback_requires_action_type(self):
        with self.assertRaises(ValidationError):
            EvaluationFeedbackRequest(
                feedback_type="approve_action",
                decision_sha256=HASH,
            )

    def test_decision_dismissal_needs_no_secondary_target(self):
        payload = EvaluationFeedbackRequest(
            feedback_type="dismiss_decision",
            decision_sha256=HASH,
            note="Not relevant to this call.",
        )

        self.assertEqual(payload.feedback_type, "dismiss_decision")


if __name__ == "__main__":
    unittest.main()
