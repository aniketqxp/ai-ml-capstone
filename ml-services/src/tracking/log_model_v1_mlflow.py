"""
Log Model V1 baseline results to MLflow.

Run from ml-services:

    python -m src.tracking.log_model_v1_mlflow

This logs the current locked Model V1:
    - emotion-pretrained Wav2Vec2
    - CREMA-D training/evaluation
    - validation/test metrics
    - confusion matrix
    - model notes
"""

import json
from pathlib import Path
from typing import Dict, Any

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

MODEL_V1_SUMMARY_PATH = REPORTS_DIR / "model_v1_summary.json"
MODEL_V1_CONFUSION_MATRIX_PATH = REPORTS_DIR / "model_v1_confusion_matrix.csv"
MODEL_V1_NOTES_PATH = REPORTS_DIR / "model_v1_notes.md"

EXPERIMENT_NAME = "audio_sentiment_emotion_classification"
RUN_NAME = "model_v1_wav2vec2_cremad"


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON file safely.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_required_files() -> None:
    """
    Make sure Model V1 files exist before logging to MLflow.
    """
    required_files = [
        MODEL_V1_SUMMARY_PATH,
        MODEL_V1_CONFUSION_MATRIX_PATH,
        MODEL_V1_NOTES_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing Model V1 files. Please lock Model V1 first.\n"
            f"{missing_text}"
        )


def log_model_v1_to_mlflow() -> None:
    """
    Log Model V1 baseline metrics, parameters, and artifacts to MLflow.
    """
    validate_required_files()

    report = load_json(MODEL_V1_SUMMARY_PATH)

    mlflow.set_tracking_uri(f"sqlite:///{ML_SERVICES_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # Core run identity
        mlflow.set_tag("model_version", "v1")
        mlflow.set_tag("module", "audio_sentiment_analysis")
        mlflow.set_tag("dataset", "CREMA-D")
        mlflow.set_tag("purpose", "locked_baseline")

        # Parameters
        mlflow.log_param("model_name", report.get("model_name"))
        mlflow.log_param("base_checkpoint", report.get("base_checkpoint"))
        mlflow.log_param("task", report.get("task"))
        mlflow.log_param("train_samples", report.get("train_samples"))
        mlflow.log_param("validation_samples", report.get("validation_samples"))
        mlflow.log_param("test_samples", report.get("test_samples"))
        mlflow.log_param("num_epochs", report.get("num_epochs"))
        mlflow.log_param("batch_size", report.get("batch_size"))
        mlflow.log_param("learning_rate", report.get("learning_rate"))
        mlflow.log_param("weight_decay", report.get("weight_decay"))
        mlflow.log_param("device", report.get("device"))
        mlflow.log_param("split_strategy", "speaker-aware")
        mlflow.log_param("training_framework", "Hugging Face Transformers")

        # Validation metrics
        validation = report.get("validation", {})
        mlflow.log_metric(
            "validation_accuracy",
            float(validation.get("eval_accuracy", 0.0)),
        )
        mlflow.log_metric(
            "validation_macro_f1",
            float(validation.get("eval_macro_f1", 0.0)),
        )
        mlflow.log_metric(
            "validation_weighted_f1",
            float(validation.get("eval_weighted_f1", 0.0)),
        )
        mlflow.log_metric(
            "validation_loss",
            float(validation.get("eval_loss", 0.0)),
        )

        # Test metrics
        test = report.get("test", {})
        mlflow.log_metric("test_accuracy", float(test.get("accuracy", 0.0)))
        mlflow.log_metric("test_macro_f1", float(test.get("macro_f1", 0.0)))
        mlflow.log_metric("test_weighted_f1", float(test.get("weighted_f1", 0.0)))

        # Artifacts
        mlflow.log_artifact(str(MODEL_V1_SUMMARY_PATH), artifact_path="reports")
        mlflow.log_artifact(str(MODEL_V1_CONFUSION_MATRIX_PATH), artifact_path="reports")
        mlflow.log_artifact(str(MODEL_V1_NOTES_PATH), artifact_path="reports")

        run_id = mlflow.active_run().info.run_id

    print("\nModel V1 logged to MLflow successfully.")
    print("-" * 60)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run ID: {run_id}")
    print(f"Tracking URI: sqlite:///{ML_SERVICES_ROOT / 'mlflow.db'}")
    print("-" * 60)


if __name__ == "__main__":
    log_model_v1_to_mlflow()