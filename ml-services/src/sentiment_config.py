"""
Configuration constants for the audio-only sentiment analysis module.

This file keeps labels, thresholds, and scoring weights in one place so the
classifier, feature extraction code, backend, and dashboard all use consistent
definitions.
"""

from enum import Enum


class EmotionLabel(str, Enum):
    """Supported emotion labels used by the sentiment module."""

    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    DISGUST = "disgust"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class OverallSentiment(str, Enum):
    """High-level audio sentiment categories."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"


class IntensityLevel(str, Enum):
    """Categorical level used for audio features."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


class SentimentShift(str, Enum):
    """Direction of emotional change across the call."""

    IMPROVED = "Improved"
    WORSENED = "Worsened"
    UNCHANGED = "Unchanged"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"


class RiskLevel(str, Enum):
    """Risk level derived from the audio escalation score."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"

class ConfidenceLevel(str, Enum):
    """Confidence level for model predictions."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


NEGATIVE_EMOTIONS = {
    EmotionLabel.ANGER,
    EmotionLabel.SADNESS,
    EmotionLabel.FEAR,
    EmotionLabel.DISGUST,
}

POSITIVE_EMOTIONS = {
    EmotionLabel.HAPPY,
}

CALM_EMOTIONS = {
    EmotionLabel.NEUTRAL,
}


# CREMA-D filename emotion codes.
# Example filename:
# 1001_DFA_ANG_XX.wav
CREMAD_EMOTION_MAP = {
    "ANG": EmotionLabel.ANGER,
    "SAD": EmotionLabel.SADNESS,
    "FEA": EmotionLabel.FEAR,
    "DIS": EmotionLabel.DISGUST,
    "HAP": EmotionLabel.HAPPY,
    "NEU": EmotionLabel.NEUTRAL,
}


# These weights will be used later when calculating audio_escalation_score.
# Defining them now makes the scoring method explainable and consistent.
ESCALATION_SCORE_WEIGHTS = {
    "anger_probability": 0.25,
    "stress_probability": 0.20,
    "negative_emotion_probability": 0.20,
    "vocal_intensity_score": 0.10,
    "pitch_variability_score": 0.10,
    "speech_rate_score": 0.05,
    "pause_score": 0.05,
    "overlap_score": 0.05,
}