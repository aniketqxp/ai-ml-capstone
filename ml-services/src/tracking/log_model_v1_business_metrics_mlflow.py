"""
Log Model V1 business sentiment evaluation to MLflow.

Run from ml-services:

    python -m src.tracking.log_model_v1_business_metrics_mlflow
"""

import json
from pathlib import Path
from typing import Any, Dict

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

BUSINESS_REPORT_PATH = REPORTS_DIR / "model_v1_business_sentiment_report.json"
BUSINESS_CONFUSION_MATRIX_PATH = (
    REPORTS_DIR / "model_v1_business_sentiment_confusion_matrix.csv"
)

EXPERIMENT_NAME = "audio_sentiment_emotion_classification"
RUN_NAME = "model_v1_business_sentiment_evaluation"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def log_business_metrics_to_mlflow() -> None:
    report = load_json(BUSINESS_REPORT_PATH)

    mlflow.set_tracking_uri(f"sqlite:///{ML_SERVICES_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tag("model_version", "v1")
        mlflow.set_tag("module", "audio_sentiment_analysis")
        mlflow.set_tag("dataset", "CREMA-D")
        mlflow.set_tag("purpose", "business_sentiment_evaluation")

        mlflow.log_param("evaluation_type", report.get("evaluation_type"))
        mlflow.log_param("business_labels", "Negative/Escalated, Neutral, Positive/Calm")
        mlflow.log_param("source", "model_v1_confusion_matrix")

        mlflow.log_metric(
            "raw_emotion_accuracy",
            float(report.get("raw_emotion_accuracy", 0.0)),
        )
        mlflow.log_metric(
            "business_sentiment_accuracy",
            float(report.get("business_sentiment_accuracy", 0.0)),
        )
        mlflow.log_metric(
            "business_sentiment_macro_f1",
            float(report.get("business_sentiment_macro_f1", 0.0)),
        )
        mlflow.log_metric(
            "business_sentiment_weighted_f1",
            float(report.get("business_sentiment_weighted_f1", 0.0)),
        )

        mlflow.log_artifact(str(BUSINESS_REPORT_PATH), artifact_path="reports")
        mlflow.log_artifact(
            str(BUSINESS_CONFUSION_MATRIX_PATH),
            artifact_path="reports",
        )

        run_id = mlflow.active_run().info.run_id

    print("\nBusiness sentiment metrics logged to MLflow successfully.")
    print("-" * 70)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run ID: {run_id}")
    print("-" * 70)


if __name__ == "__main__":
    log_business_metrics_to_mlflow()