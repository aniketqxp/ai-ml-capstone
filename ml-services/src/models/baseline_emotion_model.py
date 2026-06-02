"""
Baseline speech emotion classifier for the capstone sentiment module.

This script trains a classical ML baseline using acoustic features extracted
from CREMA-D audio. The baseline is useful because it gives measurable results
before fine-tuning Wav2Vec2.

Run from ml-services:

    python -m src.models.baseline_emotion_model

Optional quick test:

    python -m src.models.baseline_emotion_model --limit-per-split 200
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.audio_dataset import DEFAULT_METADATA_PATH, load_metadata
from src.features.audio_feature_extractor import (
    AudioFeatureConfig,
    extract_feature_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_FEATURES_CSV = ML_SERVICES_ROOT / "data" / "processed" / "cremad_baseline_features.csv"
DEFAULT_MODEL_PATH = ML_SERVICES_ROOT / "outputs" / "models" / "baseline_emotion_model.joblib"
DEFAULT_REPORT_PATH = ML_SERVICES_ROOT / "outputs" / "reports" / "baseline_emotion_report.json"
DEFAULT_CONFUSION_MATRIX_PATH = (
    ML_SERVICES_ROOT / "outputs" / "reports" / "baseline_emotion_confusion_matrix.csv"
)


LABEL_COLUMN = "emotion_label"
NON_FEATURE_COLUMNS = {
    "filename",
    "actor_id",
    "emotion_label",
    "sentiment_label",
    "split",
}


def prepare_or_load_features(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    features_csv: Path = DEFAULT_FEATURES_CSV,
    force_rebuild: bool = False,
    limit_per_split: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load saved baseline features or build them from audio files.

    Feature extraction can take time, so we cache the result as CSV.
    """
    if features_csv.exists() and not force_rebuild and limit_per_split is None:
        print(f"Loading existing feature CSV: {features_csv}")
        return pd.read_csv(features_csv)

    print("Building baseline acoustic features from audio files...")
    metadata = load_metadata(metadata_path)

    if limit_per_split is not None:
        limited_parts = []

        for split_name in ["train", "validation", "test"]:
            split_df = metadata[metadata["split"] == split_name].head(limit_per_split)
            limited_parts.append(split_df)

        metadata = pd.concat(limited_parts, ignore_index=True)

        print(f"Using quick-test limit: {limit_per_split} samples per split")
        print(f"Quick-test metadata columns: {list(metadata.columns)}")
        print(f"Quick-test split distribution: {metadata['split'].value_counts().to_dict()}")

    feature_config = AudioFeatureConfig(
        sample_rate=16_000,
        n_mfcc=20,
        max_duration_seconds=6.0,
    )

    features_df = extract_feature_dataframe(
        metadata=metadata,
        ml_services_root=ML_SERVICES_ROOT,
        config=feature_config,
    )

    if limit_per_split is None:
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_csv, index=False)
        print(f"Saved feature CSV to: {features_csv}")
    else:
        print("Quick-test mode: feature CSV was not saved.")

    return features_df


def split_features_and_labels(
    features_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split features into train, validation, and test sets.
    """
    train_df = features_df[features_df["split"] == "train"].copy()
    validation_df = features_df[features_df["split"] == "validation"].copy()
    test_df = features_df[features_df["split"] == "test"].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Train, validation, and test splits must all contain samples.")

    feature_columns = [
        column for column in features_df.columns if column not in NON_FEATURE_COLUMNS
    ]

    x_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN]

    x_validation = validation_df[feature_columns]
    y_validation = validation_df[LABEL_COLUMN]

    x_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN]

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def build_baseline_pipeline() -> Pipeline:
    """
    Build the baseline model pipeline.

    Random Forest is used because it handles nonlinear relationships and works
    well as a strong classical baseline for tabular acoustic features.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x: pd.DataFrame,
    y_true: pd.Series,
    split_name: str,
) -> Dict:
    """
    Evaluate a trained model on one split.
    """
    y_pred = model.predict(x)

    return {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Save test confusion matrix as CSV.
    """
    labels = sorted(y_test.unique())
    y_pred = model.predict(x_test)

    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(output_path)


def train_baseline_model(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    features_csv: Path = DEFAULT_FEATURES_CSV,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    confusion_matrix_path: Path = DEFAULT_CONFUSION_MATRIX_PATH,
    force_rebuild_features: bool = False,
    limit_per_split: Optional[int] = None,
) -> Dict:
    """
    Train and evaluate the baseline emotion classifier.

    Returns:
        Evaluation report dictionary.
    """
    features_df = prepare_or_load_features(
        metadata_path=metadata_path,
        features_csv=features_csv,
        force_rebuild=force_rebuild_features,
        limit_per_split=limit_per_split,
    )

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = split_features_and_labels(features_df)

    print("\nTraining baseline emotion model...")
    model = build_baseline_pipeline()
    model.fit(x_train, y_train)

    print("Evaluating baseline model...")
    validation_report = evaluate_model(model, x_validation, y_validation, "validation")
    test_report = evaluate_model(model, x_test, y_test, "test")

    full_report = {
        "model_name": "RandomForest acoustic baseline",
        "task": "6-class speech emotion classification",
        "label_column": LABEL_COLUMN,
        "feature_count": int(x_train.shape[1]),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_validation)),
        "test_samples": int(len(x_test)),
        "validation": validation_report,
        "test": test_report,
    }

    if limit_per_split is None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(full_report, file, indent=2)

        save_confusion_matrix(model, x_test, y_test, confusion_matrix_path)

        print(f"\nSaved model to: {model_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved confusion matrix to: {confusion_matrix_path}")
    else:
        print("\nQuick-test mode: model and reports were not saved.")

    print("\nBaseline Results")
    print("-" * 60)
    print(f"Validation accuracy: {validation_report['accuracy']:.4f}")
    print(f"Validation macro F1: {validation_report['macro_f1']:.4f}")
    print(f"Test accuracy: {test_report['accuracy']:.4f}")
    print(f"Test macro F1: {test_report['macro_f1']:.4f}")
    print("-" * 60)

    return full_report


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train baseline CREMA-D emotion classifier."
    )

    parser.add_argument(
        "--force-rebuild-features",
        action="store_true",
        help="Re-extract features even if cached CSV already exists.",
    )

    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional quick-test limit per split. Does not save outputs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_baseline_model(
        force_rebuild_features=args.force_rebuild_features,
        limit_per_split=args.limit_per_split,
    )