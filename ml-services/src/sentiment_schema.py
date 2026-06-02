"""
Pydantic schemas for the audio-only sentiment analysis module.

These schemas define the exact JSON contract returned by the ML service.
The backend and dashboard can depend on this structure without knowing how the
model works internally.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from sentiment_config import (
    EmotionLabel,
    IntensityLevel,
    OverallSentiment,
    RiskLevel,
    SentimentShift,
)


class EmotionProbabilities(BaseModel):
    """
    Probability distribution over supported speech emotion classes.

    These probabilities will later come from the speech emotion recognition model.
    """

    anger: float = Field(default=0.0, ge=0.0, le=1.0)
    sadness: float = Field(default=0.0, ge=0.0, le=1.0)
    fear: float = Field(default=0.0, ge=0.0, le=1.0)
    disgust: float = Field(default=0.0, ge=0.0, le=1.0)
    happy: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral: float = Field(default=0.0, ge=0.0, le=1.0)

    def as_dict(self) -> Dict[str, float]:
        """Return probabilities as a plain dictionary."""
        return self.model_dump()

    def dominant_emotion(self) -> EmotionLabel:
        """Return the emotion label with the highest probability."""
        probabilities = self.as_dict()
        dominant_label = max(probabilities, key=probabilities.get)
        return EmotionLabel(dominant_label)

    def negative_probability(self) -> float:
        """Calculate total negative emotion probability."""
        return min(
            self.anger + self.sadness + self.fear + self.disgust,
            1.0,
        )

    def calm_probability(self) -> float:
        """Return calm/neutral probability."""
        return self.neutral

    def stress_probability(self) -> float:
        """
        Estimate stress/frustration from available emotion probabilities.

        CREMA-D does not directly contain a frustration label, so for the first
        version we estimate stress from anger and fear.
        """
        return min((0.60 * self.anger) + (0.40 * self.fear), 1.0)


class AudioFeatureSummary(BaseModel):
    """
    Call-level audio features extracted from the waveform.

    These are audio-only features, not transcript-based features.
    """

    vocal_intensity: IntensityLevel = IntensityLevel.UNKNOWN
    pitch_level: IntensityLevel = IntensityLevel.UNKNOWN
    pitch_variability: IntensityLevel = IntensityLevel.UNKNOWN
    speech_rate: IntensityLevel = IntensityLevel.UNKNOWN
    pause_frequency: IntensityLevel = IntensityLevel.UNKNOWN
    long_silence_detected: bool = False
    total_silence_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    overlap_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SentimentSegment(BaseModel):
    """
    Segment-level sentiment output for the dashboard timeline.

    Long call audio will be split into smaller windows, and each window will get
    its own emotion probabilities and risk score.
    """

    segment_id: int = Field(ge=0)
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float = Field(ge=0.0)
    dominant_emotion: EmotionLabel = EmotionLabel.UNKNOWN
    overall_audio_sentiment: OverallSentiment = OverallSentiment.UNKNOWN
    emotion_probabilities: EmotionProbabilities
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_segment_times(self):
        """Ensure the segment end time is not before the start time."""
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError(
                "end_time_seconds must be greater than or equal to start_time_seconds"
            )
        return self


class PeakEmotion(BaseModel):
    """Most emotionally intense moment detected in the call."""

    time_seconds: float = Field(default=0.0, ge=0.0)
    timestamp: str = "00:00:00"
    emotion: EmotionLabel = EmotionLabel.UNKNOWN
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class AudioSentimentResult(BaseModel):
    """
    Final result returned by the audio-only sentiment module.

    This is the main object that will be saved by the backend and displayed on
    the dashboard.
    """

    call_id: str = Field(min_length=1)

    overall_audio_sentiment: OverallSentiment = OverallSentiment.UNKNOWN
    dominant_emotion: EmotionLabel = EmotionLabel.UNKNOWN

    negative_emotion_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    anger_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    stress_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    sadness_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    anxiety_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    calm_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    audio_features: AudioFeatureSummary = Field(default_factory=AudioFeatureSummary)

    emotional_volatility: IntensityLevel = IntensityLevel.UNKNOWN
    audio_sentiment_shift: SentimentShift = SentimentShift.UNKNOWN

    audio_escalation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = RiskLevel.UNKNOWN

    peak_emotion: PeakEmotion = Field(default_factory=PeakEmotion)
    sentiment_timeline: List[SentimentSegment] = Field(default_factory=list)

    model_name: Optional[str] = None
    model_version: Optional[str] = None
    processing_status: str = "success"
    warnings: List[str] = Field(default_factory=list)

    @field_validator("call_id")
    @classmethod
    def clean_call_id(cls, value: str) -> str:
        """Remove unnecessary spaces from the call ID."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("call_id cannot be empty")
        return cleaned

    def to_api_response(self) -> Dict:
        """Return a clean JSON-ready dictionary."""
        return self.model_dump(mode="json", exclude_none=True)


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.

    Example:
        192 seconds -> 00:03:12
    """
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def infer_overall_sentiment(
    probabilities: EmotionProbabilities,
) -> OverallSentiment:
    """
    Infer high-level audio sentiment from emotion probabilities.

    This will become more advanced later when we add call-level sentiment shift.
    """
    negative_score = probabilities.negative_probability()
    positive_score = probabilities.happy
    neutral_score = probabilities.neutral

    if negative_score >= 0.55:
        return OverallSentiment.NEGATIVE

    if positive_score >= 0.50 and positive_score > negative_score:
        return OverallSentiment.POSITIVE

    if neutral_score >= 0.50:
        return OverallSentiment.NEUTRAL

    if abs(negative_score - positive_score) < 0.15:
        return OverallSentiment.MIXED

    return OverallSentiment.UNKNOWN


def infer_risk_level(escalation_score: float) -> RiskLevel:
    """Convert numeric escalation score into a dashboard risk level."""
    if escalation_score >= 0.80:
        return RiskLevel.CRITICAL

    if escalation_score >= 0.60:
        return RiskLevel.HIGH

    if escalation_score >= 0.30:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW