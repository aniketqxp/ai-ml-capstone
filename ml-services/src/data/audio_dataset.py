"""
Audio dataset utilities for the CREMA-D sentiment analysis module.

This module loads the processed CREMA-D metadata CSV, resolves audio paths,
loads waveforms, resamples audio to the model sample rate, and exposes a clean
PyTorch Dataset that will be used later for Wav2Vec2 training and evaluation.

Run validation with:

    cd ml-services
    python -m src.data.test_audio_dataset
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_METADATA_PATH = ML_SERVICES_ROOT / "data" / "processed" / "cremad_metadata.csv"

DEFAULT_SAMPLE_RATE = 16_000


EmotionTask = Literal["emotion", "sentiment"]


@dataclass(frozen=True)
class LabelEncoding:
    """
    Label encoding information used for model training.

    id_to_label:
        Maps numeric class IDs to readable labels.

    label_to_id:
        Maps readable labels to numeric class IDs.
    """

    id_to_label: Dict[int, str]
    label_to_id: Dict[str, int]


EMOTION_LABELS: List[str] = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sadness",
]

SENTIMENT_LABELS: List[str] = [
    "Negative",
    "Neutral",
    "Positive",
]


def build_label_encoding(task: EmotionTask = "emotion") -> LabelEncoding:
    """
    Build a stable label encoder for the selected task.

    Args:
        task:
            "emotion" for 6-class emotion classification.
            "sentiment" for 3-class positive/neutral/negative classification.

    Returns:
        LabelEncoding with label_to_id and id_to_label mappings.
    """
    if task == "emotion":
        labels = EMOTION_LABELS
    elif task == "sentiment":
        labels = SENTIMENT_LABELS
    else:
        raise ValueError(f"Unsupported task: {task}")

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    return LabelEncoding(id_to_label=id_to_label, label_to_id=label_to_id)


def load_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """
    Load the processed CREMA-D metadata CSV.

    Args:
        metadata_path: Path to cremad_metadata.csv.

    Returns:
        Metadata DataFrame.

    Raises:
        FileNotFoundError:
            If the metadata CSV does not exist.
        ValueError:
            If required columns are missing.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Run this first from ml-services:\n"
            "python -m src.data.cremad_dataset"
        )

    dataframe = pd.read_csv(metadata_path)

    required_columns = {
        "file_path",
        "filename",
        "actor_id",
        "emotion_label",
        "sentiment_label",
        "split",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_columns)}"
        )

    return dataframe


def get_split_dataframe(
    metadata: pd.DataFrame,
    split: Literal["train", "validation", "test"],
) -> pd.DataFrame:
    """
    Filter metadata by split.

    Args:
        metadata: Full metadata DataFrame.
        split: train, validation, or test.

    Returns:
        Filtered DataFrame for the selected split.
    """
    valid_splits = {"train", "validation", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")

    split_dataframe = metadata[metadata["split"] == split].copy()

    if split_dataframe.empty:
        raise ValueError(f"No records found for split: {split}")

    split_dataframe = split_dataframe.reset_index(drop=True)
    return split_dataframe


def resolve_audio_path(file_path_value: str) -> Path:
    """
    Resolve audio path from metadata.

    The metadata should store relative paths like:
        data/raw/cremad/AudioWAV/1001_DFA_ANG_XX.wav

    This function also supports absolute paths for backward compatibility.
    """
    path = Path(file_path_value)

    if path.is_absolute():
        return path

    return ML_SERVICES_ROOT / path


def load_audio_file(
    audio_path: Path,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load one audio file as mono waveform and resample to target sample rate.

    Args:
        audio_path:
            Path to a WAV file.
        target_sample_rate:
            Target sampling rate expected by speech models like Wav2Vec2.
        max_duration_seconds:
            Optional duration limit. If provided, audio longer than this will
            be trimmed. This is useful for keeping training batches stable.

    Returns:
        waveform:
            Float32 NumPy array with shape [num_samples].
        sample_rate:
            The target sample rate.

    Raises:
        FileNotFoundError:
            If the audio file does not exist.
        ValueError:
            If the loaded audio is empty.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = max_duration_seconds if max_duration_seconds else None

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=target_sample_rate,
        mono=True,
        duration=duration,
    )

    if waveform.size == 0:
        raise ValueError(f"Loaded empty audio file: {audio_path}")

    waveform = waveform.astype(np.float32)

    return waveform, sample_rate


class CremadAudioDataset(Dataset):
    """
    PyTorch Dataset for CREMA-D audio classification.

    This dataset can be used for:
        1. Emotion classification: anger, disgust, fear, happy, neutral, sadness
        2. Sentiment classification: Negative, Neutral, Positive

    For Step 3, it returns raw waveform tensors.
    In the next step, we will connect it to a Wav2Vec2 processor.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        task: EmotionTask = "emotion",
        target_sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            metadata:
                Metadata DataFrame for a selected split.
            task:
                "emotion" or "sentiment".
            target_sample_rate:
                Target sample rate for audio loading.
            max_duration_seconds:
                Optional trimming duration.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.task = task
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.label_encoding = build_label_encoding(task)

        self.label_column = (
            "emotion_label" if self.task == "emotion" else "sentiment_label"
        )

        self._validate_labels()

    def _validate_labels(self) -> None:
        """
        Validate that all labels in the metadata exist in the label encoder.
        """
        known_labels = set(self.label_encoding.label_to_id.keys())
        actual_labels = set(self.metadata[self.label_column].unique())
        unknown_labels = actual_labels - known_labels

        if unknown_labels:
            raise ValueError(
                f"Unknown labels found for task '{self.task}': {sorted(unknown_labels)}"
            )

    def __len__(self) -> int:
        """Return number of audio samples."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict:
        """
        Load one audio sample.

        Returns:
            Dictionary containing waveform, label, and metadata.
        """
        row = self.metadata.iloc[index]

        audio_path = resolve_audio_path(row["file_path"])
        waveform, sample_rate = load_audio_file(
            audio_path=audio_path,
            target_sample_rate=self.target_sample_rate,
            max_duration_seconds=self.max_duration_seconds,
        )

        label_name = row[self.label_column]
        label_id = self.label_encoding.label_to_id[label_name]

        return {
            "input_values": torch.tensor(waveform, dtype=torch.float32),
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_name": label_name,
            "sample_rate": sample_rate,
            "file_path": str(audio_path),
            "filename": row["filename"],
            "actor_id": int(row["actor_id"]),
            "split": row["split"],
        }


def build_cremad_datasets(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    task: EmotionTask = "emotion",
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration_seconds: Optional[float] = None,
) -> Dict[str, CremadAudioDataset]:
    """
    Build train, validation, and test PyTorch datasets.

    Args:
        metadata_path:
            Path to processed metadata CSV.
        task:
            "emotion" or "sentiment".
        target_sample_rate:
            Target sample rate.
        max_duration_seconds:
            Optional max audio duration.

    Returns:
        Dictionary with train, validation, and test datasets.
    """
    metadata = load_metadata(metadata_path)

    train_df = get_split_dataframe(metadata, "train")
    validation_df = get_split_dataframe(metadata, "validation")
    test_df = get_split_dataframe(metadata, "test")

    return {
        "train": CremadAudioDataset(
            metadata=train_df,
            task=task,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=max_duration_seconds,
        ),
        "validation": CremadAudioDataset(
            metadata=validation_df,
            task=task,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=max_duration_seconds,
        ),
        "test": CremadAudioDataset(
            metadata=test_df,
            task=task,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=max_duration_seconds,
        ),
    }


def summarize_dataset(metadata_path: Path = DEFAULT_METADATA_PATH) -> Dict:
    """
    Return a compact metadata summary for debugging and reports.
    """
    metadata = load_metadata(metadata_path)

    summary = {
        "total_records": int(len(metadata)),
        "splits": metadata["split"].value_counts().to_dict(),
        "emotion_labels": metadata["emotion_label"].value_counts().to_dict(),
        "sentiment_labels": metadata["sentiment_label"].value_counts().to_dict(),
        "actors_per_split": {
            split: int(metadata[metadata["split"] == split]["actor_id"].nunique())
            for split in sorted(metadata["split"].unique())
        },
    }

    return summary