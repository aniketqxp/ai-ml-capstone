"""
Validation test for RAVDESS metadata preparation.

Run from ml-services:

    python -m src.data.test_ravdess_dataset
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

RAVDESS_METADATA_PATH = ML_SERVICES_ROOT / "data" / "processed" / "ravdess_metadata.csv"
RAVDESS_SUMMARY_PATH = ML_SERVICES_ROOT / "data" / "processed" / "ravdess_summary.json"


EXPECTED_EMOTIONS = {"anger", "disgust", "fear", "happy", "neutral", "sadness"}
EXPECTED_SPLITS = {"train", "validation", "test"}


def main() -> None:
    if not RAVDESS_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {RAVDESS_METADATA_PATH}")

    if not RAVDESS_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary: {RAVDESS_SUMMARY_PATH}")

    df = pd.read_csv(RAVDESS_METADATA_PATH)

    with RAVDESS_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    print("\nLoading RAVDESS metadata summary...")
    print(json.dumps(summary, indent=2))

    actual_emotions = set(df["emotion_label"].unique())
    actual_splits = set(df["split"].unique())

    missing_emotions = EXPECTED_EMOTIONS - actual_emotions
    unexpected_emotions = actual_emotions - EXPECTED_EMOTIONS

    if missing_emotions:
        raise ValueError(f"Missing expected emotions: {missing_emotions}")

    if unexpected_emotions:
        raise ValueError(f"Unexpected emotions found: {unexpected_emotions}")

    if actual_splits != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected splits: {actual_splits}")

    actor_split_counts = df.groupby("actor_id")["split"].nunique()
    actors_in_multiple_splits = actor_split_counts[actor_split_counts > 1]

    if not actors_in_multiple_splits.empty:
        raise ValueError(
            "Some actors appear in multiple splits:\n"
            f"{actors_in_multiple_splits}"
        )

    missing_files = [
        file_path for file_path in df["file_path"].tolist() if not Path(file_path).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Some audio files listed in metadata do not exist. "
            f"First missing file: {missing_files[0]}"
        )

    print("\nRAVDESS dataset validation completed successfully.")
    print("-" * 70)
    print(f"Total records: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Actors per split: {df.groupby('split')['actor_id'].nunique().to_dict()}")
    print(f"Emotion labels: {df['emotion_label'].value_counts().to_dict()}")
    print("-" * 70)


if __name__ == "__main__":
    main()