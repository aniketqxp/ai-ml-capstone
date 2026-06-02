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

from src.data.audio_dataset import DEFAULT_SAMPLE_RATE, load_audio_file, resolve_audio_path
from src.features.inference_audio_features import extract_audio_feature_summary
from src.sentiment_config import IntensityLevel, SentimentShift
from src.sentiment_schema import (
    AudioFeatureSummary,
    AudioSentimentResult,
    EmotionProbabilities,
    PeakEmotion,
    infer_overall_sentiment,
    infer_risk_level,
    seconds_to_timestamp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_MODEL_DIR = ML_SERVICES_ROOT / "outputs" / "wav2vec2" / "best_model"


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
                f"Trained Wav2Vec2 model not found at: {self.model_dir}\n"
                "Run training first or make sure outputs/wav2vec2/best_model exists."
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
    ) -> AudioSentimentResult:
        """
        Analyze one audio file and return the official sentiment result.
        """
        raw_probabilities = self.predict_probabilities(audio_path)
        probabilities = self._build_probability_schema(raw_probabilities)

        dominant_emotion = probabilities.dominant_emotion()
        overall_sentiment = infer_overall_sentiment(probabilities)

        audio_feature_summary = extract_audio_feature_summary(audio_path)

        escalation_score = self._calculate_basic_escalation_score(probabilities)
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
            emotional_volatility=IntensityLevel.UNKNOWN,
            audio_sentiment_shift=SentimentShift.UNKNOWN,
            audio_escalation_score=escalation_score,
            risk_level=risk_level,
            peak_emotion=PeakEmotion(
                time_seconds=0.0,
                timestamp=seconds_to_timestamp(0.0),
                emotion=dominant_emotion,
                score=escalation_score,
            ),
            sentiment_timeline=[],
            model_name="Emotion-pretrained Wav2Vec2 classifier",
            model_version="cremad-emotion-pretrained-v1",
            processing_status="success",
                        warnings=[
                "This result uses emotion probabilities and single-clip audio features. Timeline analysis, sentiment shift, and peak timestamp detection will be added in later steps."
            ],
        )