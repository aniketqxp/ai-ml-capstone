"""
Label mapping utilities for the sentiment analysis module.

This file translates dataset-specific labels into the unified emotion and
sentiment labels used by the capstone ML pipeline.
"""

from enum import Enum
from typing import Dict

try:
    from src.sentiment_config import EmotionLabel, OverallSentiment
except ModuleNotFoundError:
    from sentiment_config import EmotionLabel, OverallSentiment


class DatasetName(str, Enum):
    """Supported datasets for the sentiment module."""

    CREMAD = "cremad"


CREMAD_EMOTION_CODE_TO_LABEL: Dict[str, EmotionLabel] = {
    "ANG": EmotionLabel.ANGER,
    "DIS": EmotionLabel.DISGUST,
    "FEA": EmotionLabel.FEAR,
    "HAP": EmotionLabel.HAPPY,
    "NEU": EmotionLabel.NEUTRAL,
    "SAD": EmotionLabel.SADNESS,
}


CREMAD_INTENSITY_CODE_TO_LABEL: Dict[str, str] = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified",
}


EMOTION_TO_SENTIMENT: Dict[EmotionLabel, OverallSentiment] = {
    EmotionLabel.ANGER: OverallSentiment.NEGATIVE,
    EmotionLabel.DISGUST: OverallSentiment.NEGATIVE,
    EmotionLabel.FEAR: OverallSentiment.NEGATIVE,
    EmotionLabel.SADNESS: OverallSentiment.NEGATIVE,
    EmotionLabel.HAPPY: OverallSentiment.POSITIVE,
    EmotionLabel.NEUTRAL: OverallSentiment.NEUTRAL,
}


def map_cremad_emotion_code(emotion_code: str) -> EmotionLabel:
    """
    Convert a CREMA-D emotion code into the standard project emotion label.

    Example:
        ANG -> anger
        SAD -> sadness
        FEA -> fear
    """
    normalized_code = emotion_code.strip().upper()

    if normalized_code not in CREMAD_EMOTION_CODE_TO_LABEL:
        raise ValueError(f"Unknown CREMA-D emotion code: {emotion_code}")

    return CREMAD_EMOTION_CODE_TO_LABEL[normalized_code]


def map_cremad_intensity_code(intensity_code: str) -> str:
    """
    Convert a CREMA-D intensity code into a readable label.

    Example:
        HI -> high
        MD -> medium
        LO -> low
        XX -> unspecified
    """
    normalized_code = intensity_code.strip().upper()
    return CREMAD_INTENSITY_CODE_TO_LABEL.get(normalized_code, "unknown")


def map_emotion_to_sentiment(emotion: EmotionLabel) -> OverallSentiment:
    """
    Convert an emotion label into a high-level sentiment label.
    """
    return EMOTION_TO_SENTIMENT.get(emotion, OverallSentiment.UNKNOWN)


def is_negative_emotion(emotion: EmotionLabel) -> bool:
    """
    Return True if the emotion is considered negative for call-center risk analysis.
    """
    return emotion in {
        EmotionLabel.ANGER,
        EmotionLabel.DISGUST,
        EmotionLabel.FEAR,
        EmotionLabel.SADNESS,
    }