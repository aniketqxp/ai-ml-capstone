"""
Audio feature extraction utilities for baseline speech emotion recognition.

This module extracts traditional acoustic features from waveform audio. These
features are used by the baseline machine learning model before moving to a
deep learning model such as Wav2Vec2.

Feature groups:
    - MFCC statistics
    - Chroma statistics
    - Spectral contrast
    - Zero crossing rate
    - RMS energy / loudness
    - Pitch statistics
    - Tempo-like speech rhythm approximation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd


DEFAULT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioFeatureConfig:
    """
    Configuration for baseline audio feature extraction.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_mfcc: int = 20
    n_fft: int = 1024
    hop_length: int = 512
    max_duration_seconds: Optional[float] = 6.0


def safe_stat_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Calculate stable summary statistics for a 1D or 2D feature array.

    For 2D arrays shaped [features, frames], statistics are calculated for each
    feature dimension across time.
    """
    features: Dict[str, float] = {}

    if values.size == 0:
        features[f"{prefix}_mean"] = 0.0
        features[f"{prefix}_std"] = 0.0
        features[f"{prefix}_min"] = 0.0
        features[f"{prefix}_max"] = 0.0
        return features

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if values.ndim == 1:
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_std"] = float(np.std(values))
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))
        return features

    for index in range(values.shape[0]):
        row = values[index]
        features[f"{prefix}_{index + 1}_mean"] = float(np.mean(row))
        features[f"{prefix}_{index + 1}_std"] = float(np.std(row))
        features[f"{prefix}_{index + 1}_min"] = float(np.min(row))
        features[f"{prefix}_{index + 1}_max"] = float(np.max(row))

    return features


def load_audio_for_features(
    audio_path: Path,
    config: AudioFeatureConfig,
) -> np.ndarray:
    """
    Load audio for feature extraction.

    Args:
        audio_path: Path to audio file.
        config: Feature extraction configuration.

    Returns:
        Mono waveform at target sample rate.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, _ = librosa.load(
        audio_path,
        sr=config.sample_rate,
        mono=True,
        duration=config.max_duration_seconds,
    )

    if waveform.size == 0:
        raise ValueError(f"Loaded empty audio file: {audio_path}")

    return waveform.astype(np.float32)


def extract_pitch_features(
    waveform: np.ndarray,
    config: AudioFeatureConfig,
) -> Dict[str, float]:
    """
    Extract pitch statistics using librosa.pyin.

    Pitch is useful for emotion detection because angry, fearful, or stressed
    speech often has higher or more unstable pitch.
    """
    features: Dict[str, float] = {}

    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=config.sample_rate,
            frame_length=config.n_fft,
            hop_length=config.hop_length,
        )

        voiced_pitch = f0[voiced_flag] if f0 is not None and voiced_flag is not None else []

        if len(voiced_pitch) == 0:
            features.update(
                {
                    "pitch_mean": 0.0,
                    "pitch_std": 0.0,
                    "pitch_min": 0.0,
                    "pitch_max": 0.0,
                    "voiced_ratio": 0.0,
                }
            )
            return features

        voiced_pitch = np.nan_to_num(voiced_pitch, nan=0.0)
        features["pitch_mean"] = float(np.mean(voiced_pitch))
        features["pitch_std"] = float(np.std(voiced_pitch))
        features["pitch_min"] = float(np.min(voiced_pitch))
        features["pitch_max"] = float(np.max(voiced_pitch))
        features["voiced_ratio"] = float(np.mean(voiced_flag))

    except Exception:
        features.update(
            {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_min": 0.0,
                "pitch_max": 0.0,
                "voiced_ratio": 0.0,
            }
        )

    return features


def extract_baseline_audio_features(
    audio_path: Path,
    config: Optional[AudioFeatureConfig] = None,
) -> Dict[str, float]:
    """
    Extract a complete baseline feature vector from one audio file.

    Args:
        audio_path: Path to audio file.
        config: Optional feature extraction config.

    Returns:
        Dictionary of feature_name -> value.
    """
    if config is None:
        config = AudioFeatureConfig()

    waveform = load_audio_for_features(audio_path, config)

    features: Dict[str, float] = {}

    # Basic duration
    duration_seconds = len(waveform) / config.sample_rate
    features["duration_seconds"] = float(duration_seconds)

    # MFCCs
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(mfcc, "mfcc"))

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(chroma, "chroma"))

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(spectral_contrast, "spectral_contrast"))

    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=waveform,
        frame_length=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(zero_crossing_rate.flatten(), "zcr"))

    # RMS energy / loudness
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(rms.flatten(), "rms"))

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(spectral_centroid.flatten(), "spectral_centroid"))

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    features.update(safe_stat_features(spectral_bandwidth.flatten(), "spectral_bandwidth"))

    # Pitch
    features.update(extract_pitch_features(waveform, config))

    return features


def extract_feature_dataframe(
    metadata: pd.DataFrame,
    ml_services_root: Path,
    config: Optional[AudioFeatureConfig] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract feature vectors for a metadata DataFrame.

    Args:
        metadata: DataFrame containing file_path and labels.
        ml_services_root: Root directory of ml-services.
        config: Feature extraction config.
        limit: Optional limit for quick testing.

    Returns:
        DataFrame containing features and label columns.
    """
    if config is None:
        config = AudioFeatureConfig()

    rows: List[Dict] = []
    working_metadata = metadata.head(limit).copy() if limit else metadata.copy()

    total = len(working_metadata)

    for index, row in working_metadata.iterrows():
        file_path = Path(row["file_path"])
        audio_path = file_path if file_path.is_absolute() else ml_services_root / file_path

        try:
            feature_row = extract_baseline_audio_features(audio_path, config)
            required_columns = [
                "filename",
                "actor_id",
                "emotion_label",
                "sentiment_label",
                "split",
            ]

            missing_columns = [
                column for column in required_columns if column not in working_metadata.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Metadata is missing required columns during feature extraction: {missing_columns}"
                )

            feature_row["filename"] = row["filename"]
            feature_row["actor_id"] = int(row["actor_id"])
            feature_row["emotion_label"] = row["emotion_label"]
            feature_row["sentiment_label"] = row["sentiment_label"]
            feature_row["split"] = row["split"]
            rows.append(feature_row)

        except Exception as exc:
            print(f"[WARN] Failed to extract features for {row['filename']}: {exc}")

        if (len(rows) % 250 == 0 and len(rows) > 0) or len(rows) == total:
            print(f"Extracted features for {len(rows)}/{total} files")

    if not rows:
        raise ValueError("No features were extracted.")

    return pd.DataFrame(rows)