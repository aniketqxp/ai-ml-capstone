"""
Combined CREMA-D + RAVDESS metadata preparation.

This script combines the already-processed CREMA-D and RAVDESS metadata files
into one unified metadata CSV for Model V2 training.

Model V1:
    CREMA-D only

Model V2:
    CREMA-D + RAVDESS

Why:
    CREMA-D gives a strong baseline, while RAVDESS adds more labeled emotional
    speech data. Combining them can improve generalization and reduce confusion
    between emotions such as sadness/fear and happy/anger.

Run from ml-services:

    python -m src.data.combined_emotion_dataset
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

PROCESSED_DIR = ML_SERVICES_ROOT / "data" / "processed"

CREMAD_METADATA_PATH = PROCESSED_DIR / "cremad_metadata.csv"
RAVDESS_METADATA_PATH = PROCESSED_DIR / "ravdess_metadata.csv"

COMBINED_METADATA_PATH = PROCESSED_DIR / "combined_emotion_metadata.csv"
COMBINED_SUMMARY_PATH = PROCESSED_DIR / "combined_emotion_summary.json"


REQUIRED_COLUMNS = [
    "file_path",
    "filename",
    "dataset",
    "actor_id",
    "emotion_label",
    "sentiment_label",
    "is_negative",
    "split",
]


EXPECTED_EMOTIONS = {"anger", "disgust", "fear", "happy", "neutral", "sadness"}
EXPECTED_SPLITS = {"train", "validation", "test"}


def load_metadata(path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load metadata CSV and validate basic existence.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_name} metadata: {path}\n"
            "Run the dataset metadata preparation script first."
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"{dataset_name} metadata is empty: {path}")

    return dataframe


def ensure_dataset_column(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Ensure a dataset column exists.

    Older CREMA-D metadata may not include a dataset column, so we add it here.
    """
    dataframe = dataframe.copy()

    if "dataset" not in dataframe.columns:
        dataframe["dataset"] = dataset_name

    dataframe["dataset"] = dataframe["dataset"].fillna(dataset_name)

    return dataframe


def standardize_metadata_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Keep only the common columns needed for combined training.

    Extra dataset-specific columns are intentionally not included in the combined
    metadata to keep the training pipeline simple.
    """
    dataframe = ensure_dataset_column(dataframe, dataset_name)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} metadata is missing required columns: {missing_columns}"
        )

    standardized = dataframe[REQUIRED_COLUMNS].copy()
    standardized["dataset"] = dataset_name

    # Make actor IDs globally unique across datasets.
    # Example:
    #   CREMAD_1001
    #   RAVDESS_1
    standardized["source_actor_id"] = standardized["actor_id"].astype(str)
    standardized["global_actor_id"] = (
        standardized["dataset"].astype(str)
        + "_"
        + standardized["actor_id"].astype(str)
    )

    return standardized


def validate_combined_metadata(dataframe: pd.DataFrame) -> None:
    """
    Validate combined metadata quality.
    """
    missing_emotions = EXPECTED_EMOTIONS - set(dataframe["emotion_label"].unique())
    unexpected_emotions = set(dataframe["emotion_label"].unique()) - EXPECTED_EMOTIONS

    if missing_emotions:
        raise ValueError(f"Missing expected emotions: {missing_emotions}")

    if unexpected_emotions:
        raise ValueError(f"Unexpected emotion labels: {unexpected_emotions}")

    actual_splits = set(dataframe["split"].unique())
    if actual_splits != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split labels: {actual_splits}")

    missing_paths = [
        path for path in dataframe["file_path"].tolist() if not Path(path).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Some audio paths in combined metadata do not exist. "
            f"First missing path: {missing_paths[0]}"
        )

    duplicate_rows = dataframe.duplicated(subset=["file_path"]).sum()
    if duplicate_rows > 0:
        raise ValueError(f"Found duplicate file paths: {duplicate_rows}")


def create_combined_summary(dataframe: pd.DataFrame) -> Dict:
    """
    Create summary statistics for combined CREMA-D + RAVDESS metadata.
    """
    summary = {
        "dataset": "CREMA-D + RAVDESS",
        "total_records": int(len(dataframe)),
        "datasets": dataframe["dataset"].value_counts().to_dict(),
        "splits": dataframe["split"].value_counts().to_dict(),
        "split_by_dataset": dataframe.groupby(["dataset", "split"])
        .size()
        .unstack(fill_value=0)
        .to_dict(),
        "emotion_labels": dataframe["emotion_label"].value_counts().to_dict(),
        "emotion_by_dataset": dataframe.groupby(["dataset", "emotion_label"])
        .size()
        .unstack(fill_value=0)
        .to_dict(),
        "sentiment_labels": dataframe["sentiment_label"].value_counts().to_dict(),
        "sentiment_by_dataset": dataframe.groupby(["dataset", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .to_dict(),
        "global_actors_per_split": dataframe.groupby("split")["global_actor_id"]
        .nunique()
        .to_dict(),
        "actors_by_dataset": dataframe.groupby("dataset")["global_actor_id"]
        .nunique()
        .to_dict(),
        "notes": [
            "Combined metadata uses the existing speaker-aware split from each dataset.",
            "global_actor_id prevents actor ID collisions between CREMA-D and RAVDESS.",
            "RAVDESS calm is mapped to neutral.",
            "RAVDESS surprised samples are excluded.",
            "This metadata is prepared for Model V2 training.",
        ],
    }

    return summary


def build_combined_metadata() -> pd.DataFrame:
    """
    Build combined CREMA-D + RAVDESS metadata.
    """
    cremad_df = load_metadata(CREMAD_METADATA_PATH, "CREMA-D")
    ravdess_df = load_metadata(RAVDESS_METADATA_PATH, "RAVDESS")

    cremad_standardized = standardize_metadata_columns(cremad_df, "CREMA-D")
    ravdess_standardized = standardize_metadata_columns(ravdess_df, "RAVDESS")

    combined_df = pd.concat(
        [cremad_standardized, ravdess_standardized],
        ignore_index=True,
    )

    combined_df = combined_df.sort_values(
        by=["split", "dataset", "emotion_label", "filename"]
    ).reset_index(drop=True)

    validate_combined_metadata(combined_df)

    return combined_df


def save_combined_outputs(dataframe: pd.DataFrame, summary: Dict) -> None:
    """
    Save combined metadata and summary.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(COMBINED_METADATA_PATH, index=False)

    with COMBINED_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    combined_df = build_combined_metadata()
    summary = create_combined_summary(combined_df)
    save_combined_outputs(combined_df, summary)

    print("\nCombined CREMA-D + RAVDESS metadata preparation completed successfully.")
    print("-" * 80)
    print(f"Total records: {summary['total_records']}")
    print(f"Dataset distribution: {summary['datasets']}")
    print(f"Split distribution: {summary['splits']}")
    print(f"Emotion distribution: {summary['emotion_labels']}")
    print(f"Sentiment distribution: {summary['sentiment_labels']}")
    print(f"Actors by dataset: {summary['actors_by_dataset']}")
    print(f"Global actors per split: {summary['global_actors_per_split']}")
    print("-" * 80)
    print(f"Saved metadata CSV to: {COMBINED_METADATA_PATH}")
    print(f"Saved summary JSON to: {COMBINED_SUMMARY_PATH}")


if __name__ == "__main__":
    main()