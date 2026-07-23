"""
Emotion inference pipeline for the audio sentiment analysis module.

This module loads the trained Wav2Vec2 emotion classifier and returns the
standard AudioSentimentResult schema used by the backend and dashboard.

Run test from ml-services:

    python -m src.test_emotion_predictor
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from src.inference.escalation_score import calculate_audio_escalation_score

from src.data.audio_dataset import DEFAULT_SAMPLE_RATE, load_audio_file, resolve_audio_path
from src.features.inference_audio_features import (
    extract_audio_feature_summary,
    map_raw_features_to_summary,
)
from src.sentiment_config import IntensityLevel, SentimentShift
from src.sentiment_schema import (
    AudioSentimentResult,
    EmotionProbabilities,
    PeakEmotion,
    infer_confidence_level,
    infer_overall_sentiment,
    infer_risk_level,
    is_uncertain_prediction,
    seconds_to_timestamp,
)
from src.inference.sentiment_timeline import (
    TimelineConfig,
    build_sentiment_timeline,
    calculate_audio_sentiment_shift,
    calculate_emotional_volatility,
    find_peak_emotion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

SELECTED_EMOTION_MODEL_VERSION = "model_v5_cremad_ravdess_freeze6_epochs5_lr1e5"

SELECTED_EMOTION_MODEL_DIR = (
    ML_SERVICES_ROOT
    / "outputs"
    / "wav2vec2"
    / SELECTED_EMOTION_MODEL_VERSION
    / "best_model"
)

DEFAULT_MODEL_DIR = SELECTED_EMOTION_MODEL_DIR


class EmotionPredictor:
    """
    Loads the trained Wav2Vec2 emotion classifier and predicts emotion
    probabilities for one audio file.
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_seconds: Optional[float] = 6.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Selected emotion model not found at: {self.model_dir}\n"
                f"Expected selected model version: {SELECTED_EMOTION_MODEL_VERSION}"
            )

        self.device = self._get_device()

        self.processor = Wav2Vec2Processor.from_pretrained(str(self.model_dir))
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(str(self.model_dir))

        self.model.to(self.device)
        self.model.eval()

        self.id_to_label = {
            int(index): label for index, label in self.model.config.id2label.items()
        }

    @staticmethod
    def _get_device() -> torch.device:
        """Select the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def predict_probabilities(self, audio_path: Path) -> Dict[str, float]:
        """
        Predict emotion probabilities for one audio file.

        Args:
            audio_path: Path to WAV audio file. Can be relative to ml-services or absolute.

        Returns:
            Dictionary of emotion label to probability.
        """
        resolved_path = resolve_audio_path(str(audio_path))

        waveform, _ = load_audio_file(
            audio_path=resolved_path,
            target_sample_rate=self.sample_rate,
            max_duration_seconds=self.max_duration_seconds,
        )

        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        emotion_probabilities: Dict[str, float] = {}

        for index, probability in enumerate(probabilities):
            label = self.id_to_label[index]
            emotion_probabilities[label] = float(probability)

        return emotion_probabilities

    @staticmethod
    def _build_probability_schema(
        probabilities: Dict[str, float],
    ) -> EmotionProbabilities:
        """Convert raw model probabilities into the standardized probability schema."""
        return EmotionProbabilities(
            anger=probabilities.get("anger", 0.0),
            disgust=probabilities.get("disgust", 0.0),
            fear=probabilities.get("fear", 0.0),
            happy=probabilities.get("happy", 0.0),
            neutral=probabilities.get("neutral", 0.0),
            sadness=probabilities.get("sadness", 0.0),
        )
    @staticmethod
    def _calculate_confidence_values(
        probabilities: Dict[str, float],
    ) -> tuple[float, float, bool]:
        """
        Calculate confidence, margin, and uncertainty flag.

        Returns:
            prediction_confidence:
                Highest emotion probability.
            top_emotion_margin:
                Difference between top emotion and second emotion.
            uncertain_prediction:
                True if confidence is low or top classes are close.
        """
        if not probabilities:
            return 0.0, 0.0, True

        sorted_probabilities = sorted(
            probabilities.values(),
            reverse=True,
        )

        prediction_confidence = float(sorted_probabilities[0])

        if len(sorted_probabilities) >= 2:
            top_emotion_margin = float(
                sorted_probabilities[0] - sorted_probabilities[1]
            )
        else:
            top_emotion_margin = prediction_confidence

        uncertain = is_uncertain_prediction(
            confidence=prediction_confidence,
            top_emotion_margin=top_emotion_margin,
        )

        return prediction_confidence, top_emotion_margin, uncertain

    @staticmethod
    def _calculate_basic_escalation_score(
        probabilities: EmotionProbabilities,
    ) -> float:
        """
        Calculate first-version escalation score from emotion probabilities only.

        Later steps will add loudness, pitch variability, speech rate, pauses,
        silence duration, and overlap rate.
        """
        score = (
            0.35 * probabilities.anger
            + 0.25 * probabilities.stress_probability()
            + 0.25 * probabilities.negative_probability()
            + 0.15 * probabilities.fear
        )

        return float(np.clip(score, 0.0, 1.0))

    def analyze_audio(
        self,
        audio_path: Path,
        call_id: str = "CALL_TEST_001",
        build_timeline: bool = True,
        raw_audio_features=None,
    ) -> AudioSentimentResult:
        """
        Analyze one audio file and return the official sentiment result.
        """
        raw_probabilities = self.predict_probabilities(audio_path)
        probabilities = self._build_probability_schema(raw_probabilities)
        (
            prediction_confidence,
            top_emotion_margin,
            uncertain_prediction,
        ) = self._calculate_confidence_values(raw_probabilities)

        confidence_level = infer_confidence_level(prediction_confidence)

        dominant_emotion = probabilities.dominant_emotion()
        overall_sentiment = infer_overall_sentiment(probabilities)

        audio_feature_summary = (
            map_raw_features_to_summary(raw_audio_features)
            if raw_audio_features is not None
            else extract_audio_feature_summary(audio_path)
        )

        sentiment_timeline = []
        emotional_volatility = IntensityLevel.LOW
        audio_sentiment_shift = SentimentShift.UNCHANGED
        peak_emotion = PeakEmotion(
            time_seconds=0.0,
            timestamp=seconds_to_timestamp(0.0),
            emotion=dominant_emotion,
            score=self._calculate_basic_escalation_score(probabilities),
        )

        if build_timeline:
            sentiment_timeline = build_sentiment_timeline(
                audio_path=audio_path,
                probability_predictor=self.predict_probabilities,
                config=TimelineConfig(
                    segment_duration_seconds=5.0,
                    min_segment_duration_seconds=1.0,
                    max_duration_seconds=self.max_duration_seconds,
                ),
                single_window_probabilities=raw_probabilities,
            )

            emotional_volatility = calculate_emotional_volatility(sentiment_timeline)
            audio_sentiment_shift = calculate_audio_sentiment_shift(sentiment_timeline)
            peak_emotion = find_peak_emotion(sentiment_timeline)

        escalation_breakdown = calculate_audio_escalation_score(
            probabilities=probabilities,
            audio_features=audio_feature_summary,
            timeline=sentiment_timeline,
            emotional_volatility=emotional_volatility,
            audio_sentiment_shift=audio_sentiment_shift,
            prediction_confidence=prediction_confidence,
            uncertain_prediction=uncertain_prediction,
        )

        escalation_score = escalation_breakdown.final_score
        risk_level = infer_risk_level(escalation_score)

        return AudioSentimentResult(
            call_id=call_id,
            overall_audio_sentiment=overall_sentiment,
            dominant_emotion=dominant_emotion,
            negative_emotion_probability=probabilities.negative_probability(),
            anger_probability=probabilities.anger,
            stress_probability=probabilities.stress_probability(),
            sadness_probability=probabilities.sadness,
            anxiety_probability=probabilities.fear,
            calm_probability=probabilities.calm_probability(),
            audio_features=audio_feature_summary,
            emotional_volatility=emotional_volatility,
            audio_sentiment_shift=audio_sentiment_shift,
            audio_escalation_score=escalation_score,
            risk_level=risk_level,
            escalation_score_breakdown={
                "emotion_risk": escalation_breakdown.emotion_risk,
                "voice_risk": escalation_breakdown.voice_risk,
                "timeline_risk": escalation_breakdown.timeline_risk,
                "uncertainty_adjustment": escalation_breakdown.uncertainty_adjustment,
                "final_score": escalation_breakdown.final_score,
            },
            prediction_confidence=prediction_confidence,
            confidence_level=confidence_level,
            uncertain_prediction=uncertain_prediction,
            top_emotion_margin=top_emotion_margin,
            peak_emotion=peak_emotion,
            sentiment_timeline=sentiment_timeline,
            model_name="Wav2Vec2 emotion classifier - selected V5",
            model_version=SELECTED_EMOTION_MODEL_VERSION,
            processing_status="success",
            warnings=[
                "For short audio clips, the timeline may contain only one segment. For long call-center audio, the same model is applied across multiple segments to track emotion changes over time.",
                "If uncertain_prediction is true, the top emotion probabilities are close or the prediction confidence is low.",
            ],
        )
