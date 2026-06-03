"""
Log Model V1 error analysis to MLflow.

Run from ml-services:

    python -m src.tracking.log_model_v1_error_analysis_mlflow
"""

import json
from pathlib import Path
from typing import Any, Dict

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

ERROR_ANALYSIS_JSON_PATH = REPORTS_DIR / "model_v1_error_analysis.json"
ERROR_ANALYSIS_MD_PATH = REPORTS_DIR / "model_v1_error_analysis.md"

EXPERIMENT_NAME = "audio_sentiment_emotion_classification"
RUN_NAME = "model_v1_error_analysis"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def log_error_analysis_to_mlflow() -> None:
    report = load_json(ERROR_ANALYSIS_JSON_PATH)

    mlflow.set_tracking_uri(f"sqlite:///{ML_SERVICES_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    weakest_classes = report["interpretation"]["weakest_classes"]
    strongest_classes = report["interpretation"]["strongest_classes"]
    top_confusions = report["top_emotion_confusions"]

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tag("model_version", "v1")
        mlflow.set_tag("module", "audio_sentiment_analysis")
        mlflow.set_tag("dataset", "CREMA-D")
        mlflow.set_tag("purpose", "error_analysis")

        mlflow.log_metric(
            "raw_emotion_accuracy",
            float(report.get("raw_emotion_accuracy", 0.0)),
        )
        mlflow.log_metric(
            "business_sentiment_accuracy",
            float(report.get("business_sentiment_accuracy", 0.0)),
        )

        if weakest_classes:
            mlflow.log_param("weakest_class", weakest_classes[0]["class_name"])
            mlflow.log_metric(
                "weakest_class_recall",
                float(weakest_classes[0]["recall"]),
            )

        if strongest_classes:
            mlflow.log_param("strongest_class", strongest_classes[0]["class_name"])
            mlflow.log_metric(
                "strongest_class_recall",
                float(strongest_classes[0]["recall"]),
            )

        if top_confusions:
            top_confusion = top_confusions[0]
            mlflow.log_param(
                "top_emotion_confusion",
                f"{top_confusion['actual']} -> {top_confusion['predicted']}",
            )
            mlflow.log_metric(
                "top_emotion_confusion_count",
                float(top_confusion["count"]),
            )

        mlflow.log_artifact(str(ERROR_ANALYSIS_JSON_PATH), artifact_path="reports")
        mlflow.log_artifact(str(ERROR_ANALYSIS_MD_PATH), artifact_path="reports")

        run_id = mlflow.active_run().info.run_id

    print("\nModel V1 error analysis logged to MLflow successfully.")
    print("-" * 70)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run ID: {run_id}")
    print("-" * 70)


if __name__ == "__main__":
    log_error_analysis_to_mlflow()