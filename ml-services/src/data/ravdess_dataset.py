"""
RAVDESS metadata preparation for audio emotion classification.

This script scans the RAVDESS speech audio files, parses emotion labels from
filenames, maps them into the project emotion schema, and creates a clean
metadata CSV for training/evaluation.

Expected RAVDESS filename format:
    03-01-05-01-01-01-01.wav

Filename parts:
    modality-vocal_channel-emotion-intensity-statement-repetition-actor

Emotion codes:
    01 = neutral
    02 = calm
    03 = happy
    04 = sad
    05 = angry
    06 = fearful
    07 = disgust
    08 = surprised

Project mapping:
    neutral  -> neutral
    calm     -> neutral
    happy    -> happy
    sad      -> sadness
    angry    -> anger
    fearful  -> fear
    disgust  -> disgust
    surprised is skipped because the project has no surprise class.

Run from ml-services:

    python -m src.data.ravdess_dataset
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

RAVDESS_RAW_DIR = ML_SERVICES_ROOT / "data" / "raw" / "ravdess"
PROCESSED_DIR = ML_SERVICES_ROOT / "data" / "processed"

RAVDESS_METADATA_PATH = PROCESSED_DIR / "ravdess_metadata.csv"
RAVDESS_SUMMARY_PATH = PROCESSED_DIR / "ravdess_summary.json"


RAVDESS_EMOTION_CODE_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


RAVDESS_TO_PROJECT_EMOTION = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "disgust": "disgust",
}


NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}


@dataclass(frozen=True)
class RavdessMetadataRecord:
    """
    One RAVDESS audio metadata row.
    """

    file_path: str
    filename: str
    dataset: str
    actor_id: int
    modality_code: str
    vocal_channel_code: str
    emotion_code: str
    ravdess_emotion_label: str
    emotion_label: str
    sentiment_label: str
    intensity_code: str
    statement_code: str
    repetition_code: str
    is_negative: bool
    split: str


def infer_sentiment_label(emotion_label: str) -> str:
    """
    Map project emotion label to business sentiment.
    """
    if emotion_label in NEGATIVE_EMOTIONS:
        return "Negative"

    if emotion_label == "happy":
        return "Positive"

    return "Neutral"


def find_ravdess_audio_files(raw_dir: Path = RAVDESS_RAW_DIR) -> List[Path]:
    """
    Find all RAVDESS WAV files recursively.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"RAVDESS raw directory not found: {raw_dir}\n"
            "Expected the dataset under data/raw/ravdess/"
        )

    audio_files = sorted(raw_dir.rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found under: {raw_dir}\n"
            "Make sure RAVDESS was extracted correctly."
        )

    return audio_files


def parse_ravdess_filename(file_path: Path) -> Optional[Dict]:
    """
    Parse one RAVDESS filename.

    Returns None for unsupported labels, such as surprised.
    """
    filename = file_path.name
    stem = file_path.stem
    parts = stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename format: {filename}")

    (
        modality_code,
        vocal_channel_code,
        emotion_code,
        intensity_code,
        statement_code,
        repetition_code,
        actor_code,
    ) = parts

    if emotion_code not in RAVDESS_EMOTION_CODE_MAP:
        raise ValueError(f"Unknown RAVDESS emotion code {emotion_code} in {filename}")

    ravdess_emotion = RAVDESS_EMOTION_CODE_MAP[emotion_code]

    # Skip surprised because our current project schema does not include surprise.
    if ravdess_emotion == "surprised":
        return None

    if ravdess_emotion not in RAVDESS_TO_PROJECT_EMOTION:
        raise ValueError(f"No project mapping for RAVDESS emotion: {ravdess_emotion}")

    project_emotion = RAVDESS_TO_PROJECT_EMOTION[ravdess_emotion]
    actor_id = int(actor_code)

    return {
        "filename": filename,
        "actor_id": actor_id,
        "modality_code": modality_code,
        "vocal_channel_code": vocal_channel_code,
        "emotion_code": emotion_code,
        "ravdess_emotion_label": ravdess_emotion,
        "emotion_label": project_emotion,
        "sentiment_label": infer_sentiment_label(project_emotion),
        "intensity_code": intensity_code,
        "statement_code": statement_code,
        "repetition_code": repetition_code,
        "is_negative": project_emotion in NEGATIVE_EMOTIONS,
    }


def create_actor_split(
    actor_ids: List[int],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[int, str]:
    """
    Create speaker-aware train/validation/test split.

    This keeps each actor in only one split.
    """
    unique_actors = sorted(set(actor_ids))
    random_generator = random.Random(seed)
    random_generator.shuffle(unique_actors)

    total_actors = len(unique_actors)
    train_count = int(total_actors * train_ratio)
    validation_count = int(total_actors * validation_ratio)

    train_actors = set(unique_actors[:train_count])
    validation_actors = set(unique_actors[train_count : train_count + validation_count])
    test_actors = set(unique_actors[train_count + validation_count :])

    actor_to_split = {}

    for actor_id in train_actors:
        actor_to_split[actor_id] = "train"

    for actor_id in validation_actors:
        actor_to_split[actor_id] = "validation"

    for actor_id in test_actors:
        actor_to_split[actor_id] = "test"

    return actor_to_split


def build_ravdess_metadata() -> pd.DataFrame:
    """
    Build RAVDESS metadata dataframe.
    """
    audio_files = find_ravdess_audio_files()
    parsed_records = []

    skipped_files = []

    for file_path in audio_files:
        parsed = parse_ravdess_filename(file_path)

        if parsed is None:
            skipped_files.append(file_path.name)
            continue

        parsed["file_path"] = str(file_path)
        parsed["dataset"] = "RAVDESS"
        parsed_records.append(parsed)

    if not parsed_records:
        raise ValueError("No usable RAVDESS records were parsed.")

    actor_ids = [record["actor_id"] for record in parsed_records]
    actor_to_split = create_actor_split(actor_ids)

    records: List[RavdessMetadataRecord] = []

    for record in parsed_records:
        actor_id = record["actor_id"]

        metadata_record = RavdessMetadataRecord(
            file_path=record["file_path"],
            filename=record["filename"],
            dataset=record["dataset"],
            actor_id=actor_id,
            modality_code=record["modality_code"],
            vocal_channel_code=record["vocal_channel_code"],
            emotion_code=record["emotion_code"],
            ravdess_emotion_label=record["ravdess_emotion_label"],
            emotion_label=record["emotion_label"],
            sentiment_label=record["sentiment_label"],
            intensity_code=record["intensity_code"],
            statement_code=record["statement_code"],
            repetition_code=record["repetition_code"],
            is_negative=record["is_negative"],
            split=actor_to_split[actor_id],
        )

        records.append(metadata_record)

    dataframe = pd.DataFrame([asdict(record) for record in records])

    dataframe = dataframe.sort_values(
        by=["split", "actor_id", "emotion_label", "filename"]
    ).reset_index(drop=True)

    return dataframe


def create_summary(dataframe: pd.DataFrame) -> Dict:
    """
    Create RAVDESS metadata summary.
    """
    summary = {
        "dataset": "RAVDESS",
        "total_records": int(len(dataframe)),
        "splits": dataframe["split"].value_counts().to_dict(),
        "actors_per_split": dataframe.groupby("split")["actor_id"]
        .nunique()
        .to_dict(),
        "emotion_labels": dataframe["emotion_label"].value_counts().to_dict(),
        "ravdess_original_emotions": dataframe["ravdess_emotion_label"]
        .value_counts()
        .to_dict(),
        "sentiment_labels": dataframe["sentiment_label"].value_counts().to_dict(),
        "notes": [
            "Surprised samples are skipped because the project emotion schema does not include surprise.",
            "Calm samples are mapped to neutral.",
            "Split is speaker-aware, so each actor belongs to only one split.",
        ],
    }

    return summary


def save_metadata_and_summary(dataframe: pd.DataFrame, summary: Dict) -> None:
    """
    Save metadata CSV and summary JSON.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(RAVDESS_METADATA_PATH, index=False)

    with RAVDESS_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    dataframe = build_ravdess_metadata()
    summary = create_summary(dataframe)
    save_metadata_and_summary(dataframe, summary)

    print("\nRAVDESS metadata preparation completed successfully.")
    print("-" * 70)
    print(f"Total usable audio files: {summary['total_records']}")
    print(f"Split distribution: {summary['splits']}")
    print(f"Actors by split: {summary['actors_per_split']}")
    print(f"Emotion distribution: {summary['emotion_labels']}")
    print(f"Original RAVDESS emotions: {summary['ravdess_original_emotions']}")
    print(f"Sentiment distribution: {summary['sentiment_labels']}")
    print("-" * 70)
    print(f"Saved metadata CSV to: {RAVDESS_METADATA_PATH}")
    print(f"Saved summary JSON to: {RAVDESS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()