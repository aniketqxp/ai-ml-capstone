import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from v2.build_evaluation_data import (
    REPO_ROOT,
    build_dataset,
    build_summary,
)
from v2.evaluation_data import (
    AnnotationStatus,
    AnnotatorType,
    CounterfactualAxis,
    EvaluationDataset,
    ExpectedAttention,
    load_evaluation_dataset,
    validate_evaluation_dataset,
)

HERE = Path(__file__).resolve().parent
CHECKED_DATASET = HERE / "research" / "evaluation_data_0_1.json"
CHECKED_SUMMARY = (
    HERE / "research" / "evaluation_data_summary_0_1.json"
)


class EvaluationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset()

    def test_checked_dataset_matches_reproducible_builder(self):
        checked = load_evaluation_dataset(CHECKED_DATASET)
        self.assertEqual(
            checked.model_dump(mode="json"),
            self.dataset.model_dump(mode="json"),
        )
        checked_summary = json.loads(
            CHECKED_SUMMARY.read_text(encoding="utf-8")
        )
        self.assertEqual(
            checked_summary,
            build_summary(self.dataset),
        )

    def test_existing_annotations_are_draft_seeds_not_ground_truth(self):
        self.assertEqual(len(self.dataset.existing_calls), 10)
        for record in self.dataset.existing_calls:
            provenance = record.annotation.provenance
            self.assertEqual(
                provenance.annotator_type,
                AnnotatorType.MODEL_ASSISTED_RESEARCHER_SEED,
            )
            self.assertEqual(
                provenance.status,
                AnnotationStatus.DRAFT,
            )
            self.assertTrue(provenance.requires_human_review)
            self.assertIsNone(provenance.ground_truth_basis)
        self.assertEqual(
            sum(
                item.annotation.expected_attention
                == ExpectedAttention.REQUIRED
                for item in self.dataset.existing_calls
            ),
            4,
        )

    def test_model_assisted_seed_cannot_claim_adjudication(self):
        payload = self.dataset.existing_calls[
            0
        ].annotation.model_dump(mode="json")
        payload["provenance"]["status"] = "adjudicated"
        with self.assertRaises(ValidationError):
            self.dataset.existing_calls[
                0
            ].annotation.__class__.model_validate(payload)

    def test_all_source_evidence_and_hashes_validate(self):
        validate_evaluation_dataset(self.dataset, REPO_ROOT)

    def test_changed_evidence_quote_is_rejected(self):
        changed = self.dataset.model_copy(deep=True)
        changed.existing_calls[
            0
        ].annotation.evidence[0].quote = "Altered quote"
        with self.assertRaisesRegex(
            ValueError,
            "does not exactly match",
        ):
            validate_evaluation_dataset(changed, REPO_ROOT)

    def test_challenges_cover_every_axis_with_known_answers(self):
        self.assertEqual(len(self.dataset.challenge_calls), 14)
        self.assertEqual(len(self.dataset.counterfactual_pairs), 7)
        self.assertEqual(
            {
                pair.axis
                for pair in self.dataset.counterfactual_pairs
            },
            set(CounterfactualAxis),
        )
        for call in self.dataset.challenge_calls:
            provenance = call.annotation.provenance
            self.assertEqual(
                provenance.annotator_type,
                AnnotatorType.SYNTHETIC_AUTHOR,
            )
            self.assertEqual(
                provenance.status,
                AnnotationStatus.ADJUDICATED,
            )
            self.assertFalse(provenance.requires_human_review)
            self.assertTrue(provenance.ground_truth_basis)

    def test_declared_transcript_invariant_is_enforced(self):
        changed = self.dataset.model_copy(deep=True)
        pair = next(
            item
            for item in changed.counterfactual_pairs
            if item.axis == CounterfactualAxis.DELIVERY
        )
        variant = next(
            item
            for item in changed.challenge_calls
            if item.call_id == pair.variant_call_id
        )
        variant.segments[0].text = "Changed transcript"
        with self.assertRaisesRegex(
            ValueError,
            "transcript invariant failed",
        ):
            validate_evaluation_dataset(changed, REPO_ROOT)

    def test_unknown_pair_reference_is_rejected_by_contract(self):
        payload = self.dataset.model_dump(mode="json")
        payload["counterfactual_pairs"][0][
            "variant_call_id"
        ] = "missing-call"
        with self.assertRaises(ValidationError):
            EvaluationDataset.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
