"""
Audio-only feature extraction for sentiment inference.

This module extracts interpretable call-center audio features from waveform audio.
These features are used by the sentiment pipeline together with emotion
probabilities from the Wav2Vec2 model.

Extracted features:
    - Vocal intensity / loudness
    - Pitch level
    - Pitch variability
    - Speech rate approximation
    - Pause frequency
    - Total silence duration
    - Long silence detection
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from src.data.audio_dataset import DEFAULT_SAMPLE_RATE, load_audio_file, resolve_audio_path
from src.sentiment_config import IntensityLevel
from src.sentiment_schema import AudioFeatureSummary


@dataclass(frozen=True)
class InferenceAudioFeatureConfig:
    """
    Configuration for audio feature extraction during inference.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_length: int = 1024
    hop_length: int = 512
    max_duration_seconds: Optional[float] = 30.0

    # Silence detection threshold in decibels.
    # Higher means stricter silence detection.
    silence_top_db: int = 30

    # Pause settings.
    min_pause_duration_seconds: float = 0.30
    long_silence_threshold_seconds: float = 1.50


@dataclass(frozen=True)
class RawAudioFeatureValues:
    """
    Raw numeric audio feature values before categorical mapping.
    """

    duration_seconds: float
    rms_mean: float
    rms_std: float
    pitch_mean: float
    pitch_std: float
    voiced_ratio: float
    speech_rate_proxy: float
    pause_count: int
    pause_frequency_per_minute: float
    total_silence_duration_seconds: float
    longest_silence_seconds: float
    long_silence_detected: bool


def _level_from_value(
    value: float,
    low_threshold: float,
    high_threshold: float,
) -> IntensityLevel:
    """
    Convert a numeric value into Low / Medium / High.
    """
    if value < low_threshold:
        return IntensityLevel.LOW

    if value >= high_threshold:
        return IntensityLevel.HIGH

    return IntensityLevel.MEDIUM


def _calculate_rms_features(
    waveform: np.ndarray,
    config: InferenceAudioFeatureConfig,
) -> tuple[float, float]:
    """
    Calculate RMS loudness statistics.
    """
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    ).flatten()

    if rms.size == 0:
        return 0.0, 0.0

    return float(np.mean(rms)), float(np.std(rms))


def _calculate_pitch_features(
    waveform: np.ndarray,
    config: InferenceAudioFeatureConfig,
) -> tuple[float, float, float]:
    """
    Calculate pitch mean, pitch standard deviation, and voiced ratio.

    Uses librosa.pyin to estimate fundamental frequency.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=config.sample_rate,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )

        if f0 is None or voiced_flag is None:
            return 0.0, 0.0, 0.0

        voiced_pitch = f0[voiced_flag]

        if voiced_pitch.size == 0:
            return 0.0, 0.0, float(np.mean(voiced_flag))

        voiced_pitch = np.nan_to_num(voiced_pitch, nan=0.0)

        return (
            float(np.mean(voiced_pitch)),
            float(np.std(voiced_pitch)),
            float(np.mean(voiced_flag)),
        )

    except Exception:
        return 0.0, 0.0, 0.0


def _calculate_silence_features(
    waveform: np.ndarray,
    config: InferenceAudioFeatureConfig,
) -> tuple[int, float, float, float, bool]:
    """
    Calculate pause and silence-related features.

    Returns:
        pause_count
        pause_frequency_per_minute
        total_silence_duration_seconds
        longest_silence_seconds
        long_silence_detected
    """
    duration_seconds = len(waveform) / config.sample_rate

    non_silent_intervals = librosa.effects.split(
        waveform,
        top_db=config.silence_top_db,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )

    if len(non_silent_intervals) == 0:
        return (
            1,
            60.0 / max(duration_seconds, 1e-6),
            duration_seconds,
            duration_seconds,
            duration_seconds >= config.long_silence_threshold_seconds,
        )

    silence_durations = []

    # Silence before first speech segment.
    first_start = non_silent_intervals[0][0]
    if first_start > 0:
        silence_durations.append(first_start / config.sample_rate)

    # Silence gaps between non-silent segments.
    for previous_interval, current_interval in zip(
        non_silent_intervals[:-1],
        non_silent_intervals[1:],
    ):
        previous_end = previous_interval[1]
        current_start = current_interval[0]
        gap_duration = max(0.0, (current_start - previous_end) / config.sample_rate)

        if gap_duration > 0:
            silence_durations.append(gap_duration)

    # Silence after last speech segment.
    last_end = non_silent_intervals[-1][1]
    total_samples = len(waveform)
    if last_end < total_samples:
        silence_durations.append((total_samples - last_end) / config.sample_rate)

    meaningful_pauses = [
        duration
        for duration in silence_durations
        if duration >= config.min_pause_duration_seconds
    ]

    pause_count = len(meaningful_pauses)
    total_silence_duration = float(sum(silence_durations))
    longest_silence = float(max(silence_durations)) if silence_durations else 0.0

    pause_frequency_per_minute = (
        pause_count / max(duration_seconds / 60.0, 1e-6)
    )

    long_silence_detected = longest_silence >= config.long_silence_threshold_seconds

    return (
        pause_count,
        float(pause_frequency_per_minute),
        total_silence_duration,
        longest_silence,
        bool(long_silence_detected),
    )


def _calculate_speech_rate_proxy(
    waveform: np.ndarray,
    config: InferenceAudioFeatureConfig,
) -> float:
    """
    Estimate speech activity rate from onset strength.

    This is not a transcript-based words-per-minute value. It is an audio-only
    rhythm/activity proxy, useful for detecting fast or urgent speech patterns.
    """
    duration_seconds = len(waveform) / config.sample_rate

    if duration_seconds <= 0:
        return 0.0

    onset_envelope = librosa.onset.onset_strength(
        y=waveform,
        sr=config.sample_rate,
        hop_length=config.hop_length,
    )

    if onset_envelope.size == 0:
        return 0.0

    onset_threshold = np.mean(onset_envelope) + 0.5 * np.std(onset_envelope)
    active_onsets = int(np.sum(onset_envelope > onset_threshold))

    return float(active_onsets / max(duration_seconds, 1e-6))


def extract_raw_audio_features(
    audio_path: Path,
    config: Optional[InferenceAudioFeatureConfig] = None,
) -> RawAudioFeatureValues:
    """
    Extract raw numeric audio features from one audio file.

    Args:
        audio_path:
            Path to the audio file. Can be relative to ml-services or absolute.
        config:
            Optional feature extraction configuration.

    Returns:
        RawAudioFeatureValues.
    """
    if config is None:
        config = InferenceAudioFeatureConfig()

    resolved_path = resolve_audio_path(str(audio_path))

    waveform, _ = load_audio_file(
        audio_path=resolved_path,
        target_sample_rate=config.sample_rate,
        max_duration_seconds=config.max_duration_seconds,
    )

    duration_seconds = len(waveform) / config.sample_rate

    rms_mean, rms_std = _calculate_rms_features(waveform, config)
    pitch_mean, pitch_std, voiced_ratio = _calculate_pitch_features(waveform, config)
    speech_rate_proxy = _calculate_speech_rate_proxy(waveform, config)

    (
        pause_count,
        pause_frequency_per_minute,
        total_silence_duration_seconds,
        longest_silence_seconds,
        long_silence_detected,
    ) = _calculate_silence_features(waveform, config)

    return RawAudioFeatureValues(
        duration_seconds=float(duration_seconds),
        rms_mean=rms_mean,
        rms_std=rms_std,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        voiced_ratio=voiced_ratio,
        speech_rate_proxy=speech_rate_proxy,
        pause_count=pause_count,
        pause_frequency_per_minute=pause_frequency_per_minute,
        total_silence_duration_seconds=total_silence_duration_seconds,
        longest_silence_seconds=longest_silence_seconds,
        long_silence_detected=long_silence_detected,
    )


def map_raw_features_to_summary(
    raw_features: RawAudioFeatureValues,
) -> AudioFeatureSummary:
    """
    Convert raw numeric audio features into dashboard-friendly levels.

    These thresholds are practical starting points and can be tuned after testing
    on more call-center audio.
    """
    vocal_intensity = _level_from_value(
        raw_features.rms_mean,
        low_threshold=0.015,
        high_threshold=0.055,
    )

    pitch_level = _level_from_value(
        raw_features.pitch_mean,
        low_threshold=140.0,
        high_threshold=230.0,
    )

    pitch_variability = _level_from_value(
        raw_features.pitch_std,
        low_threshold=25.0,
        high_threshold=65.0,
    )

    speech_rate = _level_from_value(
        raw_features.speech_rate_proxy,
        low_threshold=2.0,
        high_threshold=5.0,
    )

    pause_frequency = _level_from_value(
        raw_features.pause_frequency_per_minute,
        low_threshold=4.0,
        high_threshold=10.0,
    )

    return AudioFeatureSummary(
        vocal_intensity=vocal_intensity,
        pitch_level=pitch_level,
        pitch_variability=pitch_variability,
        speech_rate=speech_rate,
        pause_frequency=pause_frequency,
        long_silence_detected=raw_features.long_silence_detected,
        total_silence_duration_seconds=round(
            raw_features.total_silence_duration_seconds,
            3,
        ),
        overlap_rate=None,
    )


def extract_audio_feature_summary(
    audio_path: Path,
    config: Optional[InferenceAudioFeatureConfig] = None,
) -> AudioFeatureSummary:
    """
    Extract dashboard-ready audio feature summary for one audio file.
    """
    raw_features = extract_raw_audio_features(audio_path=audio_path, config=config)
    return map_raw_features_to_summary(raw_features)