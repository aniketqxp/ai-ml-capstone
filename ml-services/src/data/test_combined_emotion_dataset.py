"""
Validation test for combined CREMA-D + RAVDESS metadata.

Run from ml-services:

    python -m src.data.test_combined_emotion_dataset
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

COMBINED_METADATA_PATH = (
    ML_SERVICES_ROOT / "data" / "processed" / "combined_emotion_metadata.csv"
)
COMBINED_SUMMARY_PATH = (
    ML_SERVICES_ROOT / "data" / "processed" / "combined_emotion_summary.json"
)

EXPECTED_DATASETS = {"CREMA-D", "RAVDESS"}
EXPECTED_EMOTIONS = {"anger", "disgust", "fear", "happy", "neutral", "sadness"}
EXPECTED_SPLITS = {"train", "validation", "test"}


def main() -> None:
    if not COMBINED_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {COMBINED_METADATA_PATH}")

    if not COMBINED_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary: {COMBINED_SUMMARY_PATH}")

    df = pd.read_csv(COMBINED_METADATA_PATH)

    with COMBINED_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    print("\nLoading combined metadata summary...")
    print(json.dumps(summary, indent=2))

    actual_datasets = set(df["dataset"].unique())
    actual_emotions = set(df["emotion_label"].unique())
    actual_splits = set(df["split"].unique())

    if actual_datasets != EXPECTED_DATASETS:
        raise ValueError(f"Unexpected datasets: {actual_datasets}")

    if actual_emotions != EXPECTED_EMOTIONS:
        raise ValueError(f"Unexpected emotions: {actual_emotions}")

    if actual_splits != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected splits: {actual_splits}")

    if df["file_path"].duplicated().any():
        raise ValueError("Duplicate file paths found in combined metadata.")

    missing_files = [
        file_path for file_path in df["file_path"].tolist() if not Path(file_path).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Some audio files listed in combined metadata do not exist. "
            f"First missing file: {missing_files[0]}"
        )

    actor_split_counts = df.groupby("global_actor_id")["split"].nunique()
    actors_in_multiple_splits = actor_split_counts[actor_split_counts > 1]

    if not actors_in_multiple_splits.empty:
        raise ValueError(
            "Some global actors appear in multiple splits:\n"
            f"{actors_in_multiple_splits}"
        )

    print("\nCombined metadata validation completed successfully.")
    print("-" * 80)
    print(f"Total records: {len(df)}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Emotion labels: {df['emotion_label'].value_counts().to_dict()}")
    print(f"Dataset x Split:\n{pd.crosstab(df['dataset'], df['split'])}")
    print("-" * 80)


if __name__ == "__main__":
    main()