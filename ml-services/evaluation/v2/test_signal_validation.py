import unittest
from pathlib import Path

from v2.adapters import build_signal_bundle
from v2.domain_profiles import (
    ProfileSelection,
    load_banking_profile,
    resolve_domain_plan,
)
from v2.evaluation_data import load_evaluation_dataset
from v2.findings import RequirementVerdict
from v2.run_signal_validation import (
    DATASET_PATH,
    build_signal_validation_report,
)
from v2.signal_validation import (
    AblationVariant,
    AcousticPolicy,
    AgreementStatus,
    SignalValidationReport,
    ThresholdFitStatus,
    apply_controlled_acoustics,
    challenge_transcript,
    controlled_assessments,
    make_audio_only_bundle,
)

HERE = Path(__file__).resolve().parent
CHECKED_REPORT = (
    HERE / "research" / "signal_validation_0_1.json"
)


class SignalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checked = SignalValidationReport.model_validate_json(
            CHECKED_REPORT.read_text(encoding="utf-8")
        )
        revisions = cls.checked.source_revisions
        cls.rebuilt = build_signal_validation_report(
            profile_ref=revisions["profile_ref"],
            transcript_ref=revisions["transcript_ref"],
            sentiment_ref=revisions["sentiment_ref"],
        )

    def test_checked_report_is_reproducible(self):
        self.assertEqual(
            self.rebuilt.model_dump(mode="json"),
            self.checked.model_dump(mode="json"),
        )

    def test_report_has_all_call_and_pair_ablations(self):
        self.assertEqual(len(self.checked.call_results), 72)
        self.assertEqual(len(self.checked.pair_results), 21)

    def test_controlled_multimodal_passes_known_answer_checks(self):
        metrics = next(
            item
            for item in self.checked.variant_metrics
            if (
                item.population.value == "controlled_challenges"
                and item.variant == AblationVariant.MULTIMODAL
            )
        )
        self.assertEqual(metrics.call_count, 14)
        self.assertEqual(metrics.sensitivity, 1.0)
        self.assertEqual(metrics.false_positive_rate, 0.0)
        self.assertEqual(metrics.reason_recall, 1.0)
        self.assertEqual(metrics.evidence_localization_rate, 1.0)
        pairs = next(
            item
            for item in self.checked.pair_metrics
            if item.variant == AblationVariant.MULTIMODAL
        )
        self.assertEqual(pairs.passed_count, 7)

    def test_audio_only_is_not_an_attention_detector(self):
        metrics = next(
            item
            for item in self.checked.variant_metrics
            if (
                item.population.value == "controlled_challenges"
                and item.variant == AblationVariant.AUDIO_ONLY
            )
        )
        self.assertEqual(metrics.sensitivity, 0.0)
        self.assertEqual(metrics.predicted_attention_count, 0)
        self.assertEqual(metrics.false_positive_rate, 0.0)

    def test_real_calls_have_no_audio_decision_value_yet(self):
        assessment = self.checked.acoustic_value
        self.assertEqual(
            assessment.selected_policy,
            AcousticPolicy.SUPPORT_ONLY,
        )
        self.assertEqual(
            assessment.attention_changes_from_text_only,
            0,
        )
        self.assertEqual(
            assessment.real_call_supported_finding_count,
            0,
        )
        self.assertEqual(assessment.real_call_recovery_count, 0)

    def test_human_agreement_and_threshold_fit_are_withheld(self):
        self.assertEqual(
            self.checked.human_agreement.status,
            AgreementStatus.UNAVAILABLE,
        )
        self.assertEqual(
            self.checked.human_agreement.human_adjudicated_call_count,
            0,
        )
        self.assertIsNone(
            self.checked.human_agreement.human_attention_agreement
        )
        self.assertEqual(
            self.checked.threshold_assessment.status,
            ThresholdFitStatus.NOT_ATTEMPTED,
        )
        self.assertEqual(
            self.checked.threshold_assessment.fitted_threshold_count,
            0,
        )

    def test_audio_only_bundle_removes_words_not_acoustic_structure(self):
        dataset = load_evaluation_dataset(DATASET_PATH)
        call = next(
            item
            for item in dataset.challenge_calls
            if item.call_id == "challenge-delivery-elevated"
        )
        bundle = build_signal_bundle(challenge_transcript(call))
        bundle = apply_controlled_acoustics(bundle, call)
        audio_only = make_audio_only_bundle(bundle)
        self.assertTrue(audio_only.episodes)
        self.assertTrue(audio_only.signals)
        self.assertTrue(
            all(not segment.text for segment in audio_only.segments)
        )

    def test_controlled_failure_becomes_missed_assessment(self):
        dataset = load_evaluation_dataset(DATASET_PATH)
        call = next(
            item
            for item in dataset.challenge_calls
            if (
                item.call_id
                == "challenge-control-authorization-missing"
            )
        )
        profile = load_banking_profile()
        plan = resolve_domain_plan(
            profile,
            ProfileSelection(
                call_id=call.call_id,
                intent_ids=call.intent_ids,
                facts=call.facts,
                selection_method="test",
            ),
        )
        bundle = build_signal_bundle(challenge_transcript(call))
        assessments = controlled_assessments(
            bundle,
            plan,
            call.annotation,
        )
        authorization = next(
            item
            for item in assessments.assessments
            if (
                item.requirement_id
                == "transfer.authorization_obtained"
            )
        )
        self.assertEqual(
            authorization.verdict,
            RequirementVerdict.MISSED,
        )


if __name__ == "__main__":
    unittest.main()
