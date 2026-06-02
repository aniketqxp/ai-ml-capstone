"""
CREMA-D metadata preparation script for the capstone sentiment analysis module.

This script scans CREMA-D audio files, extracts labels from filenames, maps them
to the project's unified emotion/sentiment labels, and creates a clean metadata
CSV for training, evaluation, and inference.

Run from ml-services:

    python -m src.data.cremad_dataset

Expected input:
    ml-services/data/raw/cremad/AudioWAV/*.wav

Generated outputs:
    ml-services/data/processed/cremad_metadata.csv
    ml-services/data/processed/cremad_summary.json
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data.label_mapping import (
    is_negative_emotion,
    map_cremad_emotion_code,
    map_cremad_intensity_code,
    map_emotion_to_sentiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_AUDIO_DIR = ML_SERVICES_ROOT / "data" / "raw" / "cremad" / "AudioWAV"
DEFAULT_OUTPUT_CSV = ML_SERVICES_ROOT / "data" / "processed" / "cremad_metadata.csv"
DEFAULT_OUTPUT_SUMMARY = ML_SERVICES_ROOT / "data" / "processed" / "cremad_summary.json"

RANDOM_SEED = 42


@dataclass
class CremadAudioRecord:
    """
    One parsed CREMA-D audio file record.

    CREMA-D filenames follow this structure:
        ActorID_SentenceCode_EmotionCode_IntensityCode.wav

    Example:
        1001_DFA_ANG_XX.wav
    """

    file_path: str
    filename: str
    actor_id: int
    sentence_code: str
    emotion_code: str
    intensity_code: str
    emotion_label: str
    sentiment_label: str
    intensity_label: str
    is_negative: bool
    split: str = "unassigned"


def parse_cremad_filename(file_path: Path) -> CremadAudioRecord:
    """
    Parse a CREMA-D filename and return a structured metadata record.

    Args:
        file_path: Path to one CREMA-D .wav file.

    Returns:
        CremadAudioRecord containing labels and metadata.

    Raises:
        ValueError: If the filename does not match the expected CREMA-D format.
    """
    filename = file_path.name
    stem = file_path.stem
    parts = stem.split("_")

    if len(parts) != 4:
        raise ValueError(
            f"Invalid CREMA-D filename format: {filename}. "
            "Expected format: ActorID_SentenceCode_EmotionCode_IntensityCode.wav"
        )

    actor_id_raw, sentence_code, emotion_code, intensity_code = parts

    try:
        actor_id = int(actor_id_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid actor ID in filename: {filename}") from exc

    emotion = map_cremad_emotion_code(emotion_code)
    sentiment = map_emotion_to_sentiment(emotion)
    intensity = map_cremad_intensity_code(intensity_code)

    relative_file_path = file_path.relative_to(ML_SERVICES_ROOT)

    return CremadAudioRecord(

        file_path=str(relative_file_path),
        filename=filename,
        actor_id=actor_id,
        sentence_code=sentence_code,
        emotion_code=emotion_code.upper(),
        intensity_code=intensity_code.upper(),
        emotion_label=emotion.value,
        sentiment_label=sentiment.value,
        intensity_label=intensity,
        is_negative=is_negative_emotion(emotion),
    )


def scan_cremad_audio_files(audio_dir: Path) -> Tuple[List[CremadAudioRecord], List[str]]:
    """
    Scan CREMA-D audio files and parse metadata from filenames.

    Args:
        audio_dir: Directory containing CREMA-D .wav files.

    Returns:
        A tuple of valid records and warning messages.
    """
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"CREMA-D audio directory not found: {audio_dir}\n"
            "Place the AudioWAV folder inside ml-services/data/raw/cremad/"
        )

    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {audio_dir}")

    records: List[CremadAudioRecord] = []
    warnings: List[str] = []

    for file_path in wav_files:
        try:
            records.append(parse_cremad_filename(file_path))
        except ValueError as exc:
            warnings.append(str(exc))

    if not records:
        raise ValueError("No valid CREMA-D records were parsed.")

    return records, warnings


def create_speaker_independent_splits(
    records: List[CremadAudioRecord],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> List[CremadAudioRecord]:
    """
    Assign train/validation/test splits by actor ID.

    This is important for professional ML evaluation because the same speaker
    should not appear in both training and testing. If the same actor appears
    across splits, the model may look better than it really is because it has
    already learned that speaker's voice characteristics.

    Args:
        records: Parsed CREMA-D records.
        train_ratio: Percentage of actors assigned to training.
        validation_ratio: Percentage of actors assigned to validation.
        test_ratio: Percentage of actors assigned to testing.
        seed: Random seed for reproducibility.

    Returns:
        Records with the split field assigned.
    """
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    actor_ids = sorted({record.actor_id for record in records})

    random.seed(seed)
    random.shuffle(actor_ids)

    total_actors = len(actor_ids)
    train_end = int(total_actors * train_ratio)
    validation_end = train_end + int(total_actors * validation_ratio)

    train_actors = set(actor_ids[:train_end])
    validation_actors = set(actor_ids[train_end:validation_end])
    test_actors = set(actor_ids[validation_end:])

    for record in records:
        if record.actor_id in train_actors:
            record.split = "train"
        elif record.actor_id in validation_actors:
            record.split = "validation"
        elif record.actor_id in test_actors:
            record.split = "test"
        else:
            record.split = "unassigned"

    return records


def records_to_dataframe(records: List[CremadAudioRecord]) -> pd.DataFrame:
    """
    Convert parsed records into a pandas DataFrame.
    """
    dataframe = pd.DataFrame([asdict(record) for record in records])

    ordered_columns = [
        "file_path",
        "filename",
        "actor_id",
        "sentence_code",
        "emotion_code",
        "emotion_label",
        "sentiment_label",
        "intensity_code",
        "intensity_label",
        "is_negative",
        "split",
    ]

    return dataframe[ordered_columns]


def build_dataset_summary(dataframe: pd.DataFrame, warnings: List[str]) -> Dict:
    """
    Build a dataset summary for report writing and debugging.
    """
    split_distribution = dataframe["split"].value_counts().to_dict()
    emotion_distribution = dataframe["emotion_label"].value_counts().to_dict()
    sentiment_distribution = dataframe["sentiment_label"].value_counts().to_dict()

    actors_by_split = {
        split: int(dataframe[dataframe["split"] == split]["actor_id"].nunique())
        for split in sorted(dataframe["split"].unique())
    }

    emotion_by_split = defaultdict(dict)
    for split in sorted(dataframe["split"].unique()):
        split_df = dataframe[dataframe["split"] == split]
        emotion_by_split[split] = split_df["emotion_label"].value_counts().to_dict()

    summary = {
        "dataset_name": "CREMA-D",
        "total_audio_files": int(len(dataframe)),
        "total_actors": int(dataframe["actor_id"].nunique()),
        "split_distribution": split_distribution,
        "actors_by_split": actors_by_split,
        "emotion_distribution": emotion_distribution,
        "sentiment_distribution": sentiment_distribution,
        "emotion_distribution_by_split": dict(emotion_by_split),
        "negative_class_count": int(dataframe["is_negative"].sum()),
        "non_negative_class_count": int((~dataframe["is_negative"]).sum()),
        "warnings_count": len(warnings),
        "warnings": warnings[:20],
    }

    return summary


def save_outputs(
    dataframe: pd.DataFrame,
    summary: Dict,
    output_csv: Path,
    output_summary: Path,
) -> None:
    """
    Save metadata CSV and summary JSON.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_csv, index=False)

    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def print_summary(summary: Dict, output_csv: Path, output_summary: Path) -> None:
    """
    Print a readable summary after metadata generation.
    """
    print("\nCREMA-D metadata preparation completed successfully.")
    print("-" * 60)
    print(f"Total audio files: {summary['total_audio_files']}")
    print(f"Total actors: {summary['total_actors']}")
    print(f"Split distribution: {summary['split_distribution']}")
    print(f"Actors by split: {summary['actors_by_split']}")
    print(f"Emotion distribution: {summary['emotion_distribution']}")
    print(f"Sentiment distribution: {summary['sentiment_distribution']}")
    print(f"Warnings: {summary['warnings_count']}")
    print("-" * 60)
    print(f"Saved metadata CSV to: {output_csv}")
    print(f"Saved summary JSON to: {output_summary}\n")


def prepare_cremad_metadata(
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
) -> pd.DataFrame:
    """
    Main function used by scripts, notebooks, and future pipeline code.

    Args:
        audio_dir: Directory containing CREMA-D .wav files.
        output_csv: Destination path for metadata CSV.
        output_summary: Destination path for summary JSON.

    Returns:
        Metadata DataFrame.
    """
    records, warnings = scan_cremad_audio_files(audio_dir)
    records = create_speaker_independent_splits(records)
    dataframe = records_to_dataframe(records)
    summary = build_dataset_summary(dataframe, warnings)
    save_outputs(dataframe, summary, output_csv, output_summary)
    print_summary(summary, output_csv, output_summary)

    return dataframe


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare CREMA-D metadata for audio sentiment analysis."
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Path to CREMA-D AudioWAV directory.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path where the metadata CSV will be saved.",
    )

    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Path where the dataset summary JSON will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_cremad_metadata(
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
    )