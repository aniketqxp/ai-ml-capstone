"""
Sentiment timeline utilities for long audio analysis.

This module splits longer audio files into smaller segments, runs emotion
prediction on each segment, and calculates:
    - Segment-level sentiment timeline
    - Emotional volatility
    - Audio sentiment shift
    - Peak emotional timestamp

Why this matters:
    CREMA-D contains short labelled clips, so the model learns emotion at the
    speech-segment level. For call-center audio, we apply the same model to
    short windows across the full call to understand how emotion changes over time.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

from src.data.audio_dataset import DEFAULT_SAMPLE_RATE, load_audio_file, resolve_audio_path
from src.sentiment_config import EmotionLabel, IntensityLevel, SentimentShift
from src.sentiment_schema import (
    EmotionProbabilities,
    PeakEmotion,
    SentimentSegment,
    infer_overall_sentiment,
    seconds_to_timestamp,
)


@dataclass(frozen=True)
class TimelineConfig:
    """
    Configuration for segmenting long audio into timeline windows.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    segment_duration_seconds: float = 5.0
    min_segment_duration_seconds: float = 1.0
    max_duration_seconds: Optional[float] = None


@dataclass(frozen=True)
class AudioSegmentWindow:
    """
    Represents one audio segment window.
    """

    segment_id: int
    start_time_seconds: float
    end_time_seconds: float
    waveform: np.ndarray


def split_audio_into_segments(
    audio_path: Path,
    config: Optional[TimelineConfig] = None,
) -> List[AudioSegmentWindow]:
    """
    Split an audio file into fixed-length segments.

    Args:
        audio_path:
            Path to audio file. Can be relative to ml-services or absolute.
        config:
            Timeline segmentation configuration.

    Returns:
        List of AudioSegmentWindow objects.
    """
    if config is None:
        config = TimelineConfig()

    resolved_path = resolve_audio_path(str(audio_path))

    waveform, _ = load_audio_file(
        audio_path=resolved_path,
        target_sample_rate=config.sample_rate,
        max_duration_seconds=config.max_duration_seconds,
    )

    total_samples = len(waveform)
    segment_samples = int(config.segment_duration_seconds * config.sample_rate)
    min_segment_samples = int(config.min_segment_duration_seconds * config.sample_rate)

    if total_samples == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    segments: List[AudioSegmentWindow] = []

    segment_id = 0
    for start_sample in range(0, total_samples, segment_samples):
        end_sample = min(start_sample + segment_samples, total_samples)
        segment_waveform = waveform[start_sample:end_sample]

        if len(segment_waveform) < min_segment_samples:
            continue

        start_time = start_sample / config.sample_rate
        end_time = end_sample / config.sample_rate

        segments.append(
            AudioSegmentWindow(
                segment_id=segment_id,
                start_time_seconds=float(start_time),
                end_time_seconds=float(end_time),
                waveform=segment_waveform,
            )
        )

        segment_id += 1

    if not segments:
        segments.append(
            AudioSegmentWindow(
                segment_id=0,
                start_time_seconds=0.0,
                end_time_seconds=total_samples / config.sample_rate,
                waveform=waveform,
            )
        )

    return segments


def save_segment_to_temp_wav(
    segment: AudioSegmentWindow,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """
    Save a segment waveform to a temporary WAV file.

    We do this so the existing EmotionPredictor probability function can reuse
    the same audio loading path safely.
    """
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    temp_path = Path(temp_file.name)
    temp_file.close()

    sf.write(
        file=str(temp_path),
        data=segment.waveform,
        samplerate=sample_rate,
    )

    return temp_path


def build_emotion_probabilities_schema(
    probabilities: Dict[str, float],
) -> EmotionProbabilities:
    """
    Convert raw model probabilities into EmotionProbabilities schema.
    """
    return EmotionProbabilities(
        anger=probabilities.get("anger", 0.0),
        disgust=probabilities.get("disgust", 0.0),
        fear=probabilities.get("fear", 0.0),
        happy=probabilities.get("happy", 0.0),
        neutral=probabilities.get("neutral", 0.0),
        sadness=probabilities.get("sadness", 0.0),
    )


def calculate_segment_risk_score(
    probabilities: EmotionProbabilities,
) -> float:
    """
    Calculate risk score for a segment using emotion probabilities.

    This matches the first-version escalation logic used for full-clip inference.
    """
    score = (
        0.35 * probabilities.anger
        + 0.25 * probabilities.stress_probability()
        + 0.25 * probabilities.negative_probability()
        + 0.15 * probabilities.fear
    )

    return float(np.clip(score, 0.0, 1.0))


def build_sentiment_timeline(
    audio_path: Path,
    probability_predictor: Callable[[Path], Dict[str, float]],
    config: Optional[TimelineConfig] = None,
    single_window_probabilities: Optional[Dict[str, float]] = None,
) -> List[SentimentSegment]:
    """
    Build a segment-level sentiment timeline for an audio file.

    Args:
        audio_path:
            Path to the full audio file.
        probability_predictor:
            Function that accepts an audio path and returns emotion probabilities.
        config:
            Segmenting configuration.

    Returns:
        List of SentimentSegment objects.
    """
    if config is None:
        config = TimelineConfig()

    audio_segments = split_audio_into_segments(audio_path, config)
    timeline: List[SentimentSegment] = []

    for segment in audio_segments:
        # A clip that fits in one timeline window is the same waveform already
        # classified by EmotionPredictor. Reusing that result avoids a second
        # neural inference without changing the timeline or its scores.
        if len(audio_segments) == 1 and single_window_probabilities is not None:
            raw_probabilities = single_window_probabilities
            probabilities = build_emotion_probabilities_schema(raw_probabilities)
            dominant_emotion = probabilities.dominant_emotion()
            overall_sentiment = infer_overall_sentiment(probabilities)
            risk_score = calculate_segment_risk_score(probabilities)
            timeline.append(SentimentSegment(
                segment_id=segment.segment_id,
                start_time_seconds=round(segment.start_time_seconds, 3),
                end_time_seconds=round(segment.end_time_seconds, 3),
                dominant_emotion=dominant_emotion,
                overall_audio_sentiment=overall_sentiment,
                emotion_probabilities=probabilities,
                risk_score=risk_score,
            ))
            continue
        temp_path = save_segment_to_temp_wav(
            segment=segment,
            sample_rate=config.sample_rate,
        )

        try:
            raw_probabilities = probability_predictor(temp_path)
            probabilities = build_emotion_probabilities_schema(raw_probabilities)

            dominant_emotion = probabilities.dominant_emotion()
            overall_sentiment = infer_overall_sentiment(probabilities)
            risk_score = calculate_segment_risk_score(probabilities)

            timeline.append(
                SentimentSegment(
                    segment_id=segment.segment_id,
                    start_time_seconds=round(segment.start_time_seconds, 3),
                    end_time_seconds=round(segment.end_time_seconds, 3),
                    dominant_emotion=dominant_emotion,
                    overall_audio_sentiment=overall_sentiment,
                    emotion_probabilities=probabilities,
                    risk_score=risk_score,
                )
            )

        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return timeline


def calculate_emotional_volatility(
    timeline: List[SentimentSegment],
) -> IntensityLevel:
    """
    Calculate how much emotion changes across the call.

    For one short CREMA-D clip, volatility will usually be Low. For long calls,
    frequent emotion/risk changes can become Medium or High.
    """
    if len(timeline) <= 1:
        return IntensityLevel.LOW

    emotion_changes = 0
    risk_changes: List[float] = []

    for previous_segment, current_segment in zip(timeline[:-1], timeline[1:]):
        if previous_segment.dominant_emotion != current_segment.dominant_emotion:
            emotion_changes += 1

        risk_changes.append(
            abs(current_segment.risk_score - previous_segment.risk_score)
        )

    emotion_change_rate = emotion_changes / max(len(timeline) - 1, 1)
    average_risk_change = float(np.mean(risk_changes)) if risk_changes else 0.0

    volatility_score = (0.60 * emotion_change_rate) + (0.40 * average_risk_change)

    if volatility_score >= 0.55:
        return IntensityLevel.HIGH

    if volatility_score >= 0.25:
        return IntensityLevel.MEDIUM

    return IntensityLevel.LOW


def calculate_audio_sentiment_shift(
    timeline: List[SentimentSegment],
) -> SentimentShift:
    """
    Calculate whether emotion improved, worsened, stayed unchanged, or was mixed.

    Uses risk score from the first and last meaningful segments.
    """
    if len(timeline) <= 1:
        return SentimentShift.UNCHANGED

    first_risk = timeline[0].risk_score
    last_risk = timeline[-1].risk_score

    risk_delta = last_risk - first_risk

    risk_values = [segment.risk_score for segment in timeline]
    risk_range = max(risk_values) - min(risk_values)

    if risk_range >= 0.45 and abs(risk_delta) < 0.20:
        return SentimentShift.MIXED

    if risk_delta <= -0.20:
        return SentimentShift.IMPROVED

    if risk_delta >= 0.20:
        return SentimentShift.WORSENED

    return SentimentShift.UNCHANGED


def find_peak_emotion(
    timeline: List[SentimentSegment],
) -> PeakEmotion:
    """
    Find the segment with the highest emotional risk score.
    """
    if not timeline:
        return PeakEmotion()

    peak_segment = max(timeline, key=lambda segment: segment.risk_score)
    peak_time = (
        peak_segment.start_time_seconds + peak_segment.end_time_seconds
    ) / 2.0

    return PeakEmotion(
        time_seconds=round(float(peak_time), 3),
        timestamp=seconds_to_timestamp(peak_time),
        emotion=peak_segment.dominant_emotion,
        score=peak_segment.risk_score,
    )


def summarize_timeline_risk(
    timeline: List[SentimentSegment],
) -> float:
    """
    Calculate a call-level risk score from the timeline.

    Uses both average risk and peak risk so that one very emotional segment is
    not ignored.
    """
    if not timeline:
        return 0.0

    risk_values = [segment.risk_score for segment in timeline]

    average_risk = float(np.mean(risk_values))
    peak_risk = float(np.max(risk_values))

    call_level_score = (0.60 * average_risk) + (0.40 * peak_risk)

    return float(np.clip(call_level_score, 0.0, 1.0))
