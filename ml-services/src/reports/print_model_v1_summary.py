"""
Print Model V1 summary.

Run from ml-services:

    python -m src.reports.print_model_v1_summary
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

MODEL_V1_REPORT_PATH = ML_SERVICES_ROOT / "outputs" / "reports" / "model_v1_summary.json"
MODEL_V1_CONFUSION_MATRIX_PATH = (
    ML_SERVICES_ROOT / "outputs" / "reports" / "model_v1_confusion_matrix.csv"
)


def main() -> None:
    if not MODEL_V1_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing report: {MODEL_V1_REPORT_PATH}")

    with MODEL_V1_REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    print("\nModel V1 Summary")
    print("-" * 60)
    print(f"Model: {report.get('model_name')}")
    print(f"Base checkpoint: {report.get('base_checkpoint')}")
    print(f"Task: {report.get('task')}")
    print(f"Train samples: {report.get('train_samples')}")
    print(f"Validation samples: {report.get('validation_samples')}")
    print(f"Test samples: {report.get('test_samples')}")
    print(f"Epochs: {report.get('num_epochs')}")
    print(f"Batch size: {report.get('batch_size')}")
    print(f"Learning rate: {report.get('learning_rate')}")
    print()
    print(f"Validation accuracy: {report['validation']['eval_accuracy']:.4f}")
    print(f"Validation macro F1: {report['validation']['eval_macro_f1']:.4f}")
    print(f"Test accuracy: {report['test']['accuracy']:.4f}")
    print(f"Test macro F1: {report['test']['macro_f1']:.4f}")
    print("-" * 60)
    print(f"Confusion matrix: {MODEL_V1_CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
