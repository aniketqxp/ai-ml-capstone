"""
Audio escalation scoring for call-center sentiment analysis.

This module combines emotion probabilities, audio features, timeline behavior,
and prediction confidence into one audio escalation score.

The score is designed for dashboard triage:
    - Low: emotionally calm or low-risk
    - Medium: some negative/emotional signal
    - High: strong anger/stress or worsening emotional pattern
    - Critical: intense negative emotional state with strong voice/timeline risk

The score is not meant to replace human QA. It prioritizes calls that need review.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.sentiment_config import IntensityLevel, SentimentShift
from src.sentiment_schema import AudioFeatureSummary, EmotionProbabilities, SentimentSegment


@dataclass(frozen=True)
class EscalationScoreBreakdown:
    """
    Detailed score components used for debugging/reporting.

    This is useful later if we want to show why a call was marked high risk.
    """

    emotion_risk: float
    voice_risk: float
    timeline_risk: float
    uncertainty_adjustment: float
    final_score: float


def level_to_score(level: IntensityLevel) -> float:
    """
    Convert Low / Medium / High feature levels into numeric risk values.
    """
    if level == IntensityLevel.HIGH:
        return 1.0

    if level == IntensityLevel.MEDIUM:
        return 0.5

    if level == IntensityLevel.LOW:
        return 0.0

    return 0.0


def sentiment_shift_to_score(shift: SentimentShift) -> float:
    """
    Convert sentiment shift into numeric risk.

    Worsened = higher risk.
    Improved = lower risk.
    Mixed = medium/high because the call was emotionally unstable.
    """
    if shift == SentimentShift.WORSENED:
        return 1.0

    if shift == SentimentShift.MIXED:
        return 0.7

    if shift == SentimentShift.UNCHANGED:
        return 0.3

    if shift == SentimentShift.IMPROVED:
        return 0.0

    return 0.0


def calculate_emotion_risk(probabilities: EmotionProbabilities) -> float:
    """
    Calculate emotion-only risk.

    Anger and stress/fear are weighted strongly because they are important
    escalation indicators in call-center conversations.
    """
    emotion_risk = (
        0.35 * probabilities.anger
        + 0.25 * probabilities.stress_probability()
        + 0.25 * probabilities.negative_probability()
        + 0.15 * probabilities.fear
    )

    return float(np.clip(emotion_risk, 0.0, 1.0))


def calculate_voice_risk(audio_features: AudioFeatureSummary) -> float:
    """
    Calculate risk from interpretable audio features.

    High loudness, high pitch variability, fast speech, frequent pauses, and
    long silence can all indicate emotional escalation or poor call handling.
    """
    vocal_intensity_score = level_to_score(audio_features.vocal_intensity)
    pitch_variability_score = level_to_score(audio_features.pitch_variability)
    speech_rate_score = level_to_score(audio_features.speech_rate)
    pause_frequency_score = level_to_score(audio_features.pause_frequency)
    long_silence_score = 1.0 if audio_features.long_silence_detected else 0.0

    voice_risk = (
        0.30 * vocal_intensity_score
        + 0.25 * pitch_variability_score
        + 0.20 * speech_rate_score
        + 0.15 * pause_frequency_score
        + 0.10 * long_silence_score
    )

    return float(np.clip(voice_risk, 0.0, 1.0))


def calculate_timeline_risk(
    timeline: list[SentimentSegment],
    emotional_volatility: IntensityLevel,
    audio_sentiment_shift: SentimentShift,
) -> float:
    """
    Calculate risk from the sentiment timeline.

    For short clips, this will usually be close to the single segment risk.
    For long calls, it captures peak emotion, average emotion, volatility, and
    whether the emotional state improved or worsened.
    """
    if not timeline:
        average_segment_risk = 0.0
        peak_segment_risk = 0.0
    else:
        segment_risks = [segment.risk_score for segment in timeline]
        average_segment_risk = float(np.mean(segment_risks))
        peak_segment_risk = float(np.max(segment_risks))

    volatility_score = level_to_score(emotional_volatility)
    shift_score = sentiment_shift_to_score(audio_sentiment_shift)

    timeline_risk = (
        0.35 * average_segment_risk
        + 0.30 * peak_segment_risk
        + 0.20 * volatility_score
        + 0.15 * shift_score
    )

    return float(np.clip(timeline_risk, 0.0, 1.0))


def calculate_uncertainty_adjustment(
    uncertain_prediction: bool,
    prediction_confidence: float,
) -> float:
    """
    Calculate score adjustment for uncertain predictions.

    If prediction is uncertain, we slightly reduce the final automated score
    because the model is not confident. The separate uncertain_prediction flag
    should still tell the dashboard to review the result manually.
    """
    if not uncertain_prediction:
        return 1.0

    if prediction_confidence < 0.50:
        return 0.85

    return 0.90


def calculate_audio_escalation_score(
    probabilities: EmotionProbabilities,
    audio_features: AudioFeatureSummary,
    timeline: Optional[list[SentimentSegment]] = None,
    emotional_volatility: IntensityLevel = IntensityLevel.UNKNOWN,
    audio_sentiment_shift: SentimentShift = SentimentShift.UNKNOWN,
    prediction_confidence: float = 1.0,
    uncertain_prediction: bool = False,
) -> EscalationScoreBreakdown:
    """
    Calculate final audio escalation score.

    The final score combines:
        - emotion probabilities
        - voice feature risk
        - timeline risk
        - uncertainty adjustment

    Returns:
        EscalationScoreBreakdown with component values and final score.
    """
    if timeline is None:
        timeline = []

    emotion_risk = calculate_emotion_risk(probabilities)
    voice_risk = calculate_voice_risk(audio_features)
    timeline_risk = calculate_timeline_risk(
        timeline=timeline,
        emotional_volatility=emotional_volatility,
        audio_sentiment_shift=audio_sentiment_shift,
    )

    uncertainty_adjustment = calculate_uncertainty_adjustment(
        uncertain_prediction=uncertain_prediction,
        prediction_confidence=prediction_confidence,
    )

    # Emotion is still the main signal, but voice/timeline features make the
    # score more useful for call-center triage.
    combined_score = (
        0.50 * emotion_risk
        + 0.25 * voice_risk
        + 0.25 * timeline_risk
    )

    final_score = combined_score * uncertainty_adjustment

    return EscalationScoreBreakdown(
        emotion_risk=float(np.clip(emotion_risk, 0.0, 1.0)),
        voice_risk=float(np.clip(voice_risk, 0.0, 1.0)),
        timeline_risk=float(np.clip(timeline_risk, 0.0, 1.0)),
        uncertainty_adjustment=float(np.clip(uncertainty_adjustment, 0.0, 1.0)),
        final_score=float(np.clip(final_score, 0.0, 1.0)),
    )