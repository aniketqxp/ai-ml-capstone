import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from v2.adapters import build_signal_bundle
from v2.schemas import (
    ActionExecution,
    ActionType,
    Applicability,
    CallDecision,
    DecisionPolicyTrace,
    DecisionStatus,
    DetectionRule,
    EvidenceRef,
    Finding,
    FindingCategory,
    FindingPolarity,
    FindingQualification,
    FindingSeverity,
    Modality,
    PresentationSelection,
    QualificationReason,
    RecommendedAction,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalRecord,
    SignalScope,
    SourceProvenance,
    Visibility,
)
from v2.validation import validate_decision_references

REPO_ROOT = Path(__file__).resolve().parents[3]


def _transcript_fixture():
    return {
        "call_id": "banking-call-1",
        "domain": "banking",
        "accent": "en-CA",
        "model": "faster-whisper small.en/int8",
        "sentences": [
            {
                "id": "CUSTOMER_001",
                "seq_id": 7,
                "speaker": "CUSTOMER",
                "start": 1.0,
                "end": 3.0,
                "text": "I still cannot access the transfer.",
            },
            {
                "id": "CUSTOMER_002",
                "seq_id": 7,
                "speaker": "CUSTOMER",
                "start": 4.0,
                "end": 4.4,
                "text": "Hello?",
            },
        ],
    }


def _enriched_sentiment_fixture():
    return {
        "call_id": "banking-call-1",
        "domain": "banking",
        "model_version": "wav2vec2-test",
        "has_audio_features": True,
        "segments": [
            {
                "segment_index": 1,
                "seq_id": 7,
                "speaker": "CUSTOMER",
                "sentiment": "Negative",
                "dominant_emotion": "anger",
                "escalation_score": 0.72,
                "prediction_confidence": 0.83,
                "negative_emotion_probability": 0.78,
                "processing_status": "success",
                "audio_features": {
                    "duration_seconds": 2.0,
                    "pitch_mean_hz": 188.0,
                    "pitch_std_hz": 31.0,
                    "rms_energy_mean": 0.08,
                    "volume_db_mean": -11.0,
                    "pause_ratio": 0.1,
                    "pause_count": 1,
                    "speech_rate_words_per_minute": 150.0,
                    "audio_quality_flags": {
                        "short_segment": False,
                        "missing_pitch": False,
                    },
                },
            },
            {
                "segment_index": 2,
                "seq_id": 7,
                "speaker": "CUSTOMER",
                "sentiment": None,
                "dominant_emotion": None,
                "escalation_score": None,
                "processing_status": "skipped_too_short",
                "audio_features": {
                    "duration_seconds": 0.4,
                    "pitch_mean_hz": 205.0,
                    "speech_rate_words_per_minute": 300.0,
                    "audio_quality_flags": {
                        "very_short_segment": True,
                        "unrealistic_speech_rate": True,
                    },
                },
            },
        ],
        "temporal_emotion_trajectory": {
            "trajectory_direction": "worsened",
            "trajectory_delta": 0.21,
            "start_escalation": 0.32,
            "end_escalation": 0.53,
            "peak_escalation": 0.72,
            "deescalation_detected": False,
            "unresolved_end_risk": True,
        },
        "agent_empathy_tone_alignment": {
            "empathy_score": 92,
        },
        "multi_signal_escalation_intelligence": {
            "resolution_effectiveness_score": 96,
        },
    }


def _finding(polarity=FindingPolarity.NEGATIVE):
    return Finding(
        finding_id=(
            "finding-unresolved"
            if polarity == FindingPolarity.NEGATIVE
            else "finding-next-step"
        ),
        finding_type=(
            "outcome.unresolved"
            if polarity == FindingPolarity.NEGATIVE
            else "outcome.next_step_confirmed"
        ),
        category=FindingCategory.OUTCOME,
        polarity=polarity,
        severity=(
            FindingSeverity.REVIEW
            if polarity == FindingPolarity.NEGATIVE
            else FindingSeverity.INFO
        ),
        title=(
            "Outcome remained unresolved"
            if polarity == FindingPolarity.NEGATIVE
            else "Next step was confirmed"
        ),
        summary="The call contains direct transcript evidence.",
        business_definition="Whether the customer leaves with a resolved outcome.",
        applicability=Applicability(
            applies=True,
            profile_id="banking-v1",
            rule_id="outcome-resolution",
            reason="The customer requested an account transaction.",
        ),
        detection_rule=DetectionRule(
            rule_id="outcome-resolution",
            rule_version="1",
            detector="text_evidence",
        ),
        evidence=[
            EvidenceRef(
                evidence_id="evidence-1",
                modality=Modality.TRANSCRIPT,
                segment_ids=["CUSTOMER_001"],
                start_seconds=1.0,
                end_seconds=3.0,
                quote="I still cannot access the transfer.",
            )
        ],
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.USABLE,
            coverage_ratio=1.0,
        ),
        visibility=Visibility.PRIMARY,
    )


def _decision_trace(finding):
    return DecisionPolicyTrace(
        policy_id="test-policy",
        policy_version="1",
        aggregation="any_qualifying_finding",
        qualifications=[
            FindingQualification(
                finding_id=finding.finding_id,
                qualifies_for_attention=True,
                reason=QualificationReason.QUALIFIES_REVIEW,
                precedence_group="review.outcome",
            )
        ],
        controlling_finding_ids=[finding.finding_id],
    )


class SignalBundleTests(unittest.TestCase):
    def test_current_sentence_segments_map_without_sentiment(self):
        path = (
            REPO_ROOT
            / "data"
            / "sentence_segments"
            / "banking"
            / "en_CA_Banking_1586889.json"
        )
        transcript = json.loads(path.read_text(encoding="utf-8"))
        bundle = build_signal_bundle(transcript)

        self.assertEqual(bundle.call_id, "en_CA_Banking_1586889")
        self.assertEqual(len(bundle.segments), 152)
        self.assertEqual(bundle.signals, [])
        self.assertEqual(bundle.coverage[0].coverage_ratio, 1.0)

    def test_enriched_payload_maps_raw_features_and_candidate_trajectory(self):
        bundle = build_signal_bundle(
            _transcript_fixture(),
            _enriched_sentiment_fixture(),
        )
        names = {signal.name for signal in bundle.signals}

        self.assertIn("acoustic.escalation.score", names)
        self.assertIn("acoustic.pitch.mean", names)
        self.assertIn("candidate.trajectory.delta", names)
        self.assertNotIn("candidate.empathy_score", names)
        self.assertNotIn("candidate.resolution_effectiveness_score", names)
        self.assertEqual(
            [segment.seq_id for segment in bundle.segments],
            [7, 7],
        )
        emotion_coverage = next(
            item
            for item in bundle.coverage
            if item.source == "emotion_model"
        )
        self.assertEqual(emotion_coverage.coverage_ratio, 0.5)

    def test_raw_signal_cannot_be_primary_content(self):
        with self.assertRaises(ValidationError):
            SignalRecord(
                signal_id="signal-1",
                name="acoustic.pitch.mean",
                modality=Modality.ACOUSTIC,
                scope=SignalScope.CALL,
                value=180.0,
                unit="hz",
                reliability=ReliabilityAssessment(
                    status=ReliabilityStatus.USABLE
                ),
                provenance=SourceProvenance(
                    producer="test",
                    producer_version="1",
                ),
                visibility=Visibility.PRIMARY,
            )


class DecisionContractTests(unittest.TestCase):
    def test_attention_decision_requires_evidence_backed_finding(self):
        finding = _finding()
        decision = CallDecision(
            call_id="banking-call-1",
            evaluator_version="v2",
            domain_profile_id="banking-v1",
            decision_status=DecisionStatus.COMPLETE,
            attention_required=True,
            triggered_findings=[finding],
            recommended_action=RecommendedAction(
                action_type=ActionType.MANAGER_REVIEW,
                execution=ActionExecution.AUTOMATIC,
                label="Email manager review summary",
                reason="An unresolved outcome requires review.",
                finding_ids=[finding.finding_id],
                automation_allowed=True,
                requires_human_approval=False,
            ),
            decision_trace=_decision_trace(finding),
            presentation=PresentationSelection(
                primary_finding_ids=[finding.finding_id],
            ),
            provenance=SourceProvenance(
                producer="evaluator-v2",
                producer_version="2.0",
            ),
        )

        self.assertTrue(decision.attention_required)
        self.assertEqual(
            decision.presentation.primary_finding_ids,
            ["finding-unresolved"],
        )

    def test_attention_without_findings_is_rejected(self):
        with self.assertRaises(ValidationError):
            CallDecision(
                call_id="banking-call-1",
                evaluator_version="v2",
                domain_profile_id="banking-v1",
                decision_status=DecisionStatus.COMPLETE,
                attention_required=True,
                recommended_action=RecommendedAction(
                    action_type=ActionType.MANAGER_REVIEW,
                    execution=ActionExecution.AUTOMATIC,
                    label="Email manager review summary",
                    reason="Review required.",
                    automation_allowed=True,
                ),
                decision_trace=DecisionPolicyTrace(
                    policy_id="test-policy",
                    policy_version="1",
                    aggregation="any_qualifying_finding",
                ),
                presentation=PresentationSelection(),
                provenance=SourceProvenance(
                    producer="evaluator-v2",
                    producer_version="2.0",
                ),
            )

    def test_consequential_action_cannot_be_automatic(self):
        with self.assertRaises(ValidationError):
            RecommendedAction(
                action_type=ActionType.CUSTOMER_FOLLOW_UP,
                execution=ActionExecution.AUTOMATIC,
                label="Contact customer",
                reason="The issue may be unresolved.",
                automation_allowed=True,
            )

    def test_none_action_cannot_cite_a_finding(self):
        with self.assertRaises(ValidationError):
            RecommendedAction(
                action_type=ActionType.NONE,
                execution=ActionExecution.NO_ACTION,
                label="No action",
                reason="No action is required.",
                finding_ids=["finding-unresolved"],
            )

    def test_schema_generation_succeeds(self):
        schema = CallDecision.model_json_schema()
        self.assertIn("$defs", schema)
        self.assertIn("attention_required", schema["properties"])

    def test_decision_evidence_must_resolve_in_signal_bundle(self):
        bundle = build_signal_bundle(
            _transcript_fixture(),
            _enriched_sentiment_fixture(),
        )
        finding = _finding()
        decision = CallDecision(
            call_id="banking-call-1",
            evaluator_version="v2",
            domain_profile_id="banking-v1",
            decision_status=DecisionStatus.COMPLETE,
            attention_required=True,
            triggered_findings=[finding],
            recommended_action=RecommendedAction(
                action_type=ActionType.MANAGER_REVIEW,
                execution=ActionExecution.AUTOMATIC,
                label="Email manager review summary",
                reason="An unresolved outcome requires review.",
                finding_ids=[finding.finding_id],
                automation_allowed=True,
            ),
            decision_trace=_decision_trace(finding),
            presentation=PresentationSelection(
                primary_finding_ids=[finding.finding_id],
            ),
            provenance=SourceProvenance(
                producer="evaluator-v2",
                producer_version="2.0",
            ),
        )
        self.assertIs(
            validate_decision_references(decision, bundle),
            decision,
        )

        broken = decision.model_copy(deep=True)
        broken.triggered_findings[0].evidence[0].segment_ids = ["missing"]
        with self.assertRaises(ValueError):
            validate_decision_references(broken, bundle)


if __name__ == "__main__":
    unittest.main()
