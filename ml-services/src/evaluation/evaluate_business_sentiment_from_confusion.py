"""
Reusable business sentiment evaluation from an emotion confusion matrix.

This script converts 6-class emotion results into 3-class business sentiment:

    Negative/Escalated = anger, disgust, fear, sadness
    Neutral = neutral
    Positive/Calm = happy

Run from ml-services:

    python -m src.evaluation.evaluate_business_sentiment_from_confusion \
      --model-version model_v2_cremad_ravdess \
      --confusion-matrix-path outputs/reports/model_v2_cremad_ravdess_confusion_matrix.csv \
      --model-report-path outputs/reports/model_v2_cremad_ravdess_report.json \
      --output-prefix model_v2_cremad_ravdess
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"
REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

BUSINESS_LABELS = ["Negative/Escalated", "Neutral", "Positive/Calm"]

EMOTION_TO_BUSINESS_SENTIMENT = {
    "anger": "Negative/Escalated",
    "disgust": "Negative/Escalated",
    "fear": "Negative/Escalated",
    "sadness": "Negative/Escalated",
    "neutral": "Neutral",
    "happy": "Positive/Calm",
}


def normalize_label(label: str) -> str:
    """
    Normalize confusion matrix labels.

    Example:
        actual_anger -> anger
        predicted_sadness -> sadness
    """
    return (
        str(label)
        .replace("actual_", "")
        .replace("predicted_", "")
        .strip()
        .lower()
    )


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def expand_confusion_matrix_to_labels(
    confusion_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Convert a confusion matrix back into y_true and y_pred label lists.
    """
    y_true: List[str] = []
    y_pred: List[str] = []

    for actual_label in confusion_df.index:
        actual_emotion = normalize_label(actual_label)

        for predicted_label in confusion_df.columns:
            predicted_emotion = normalize_label(predicted_label)
            count = int(confusion_df.loc[actual_label, predicted_label])

            y_true.extend([actual_emotion] * count)
            y_pred.extend([predicted_emotion] * count)

    return y_true, y_pred


def map_to_business_sentiment(emotions: List[str]) -> List[str]:
    """
    Convert raw emotion labels into business sentiment labels.
    """
    mapped = []

    for emotion in emotions:
        if emotion not in EMOTION_TO_BUSINESS_SENTIMENT:
            raise ValueError(f"Unknown emotion label: {emotion}")

        mapped.append(EMOTION_TO_BUSINESS_SENTIMENT[emotion])

    return mapped


def evaluate_business_sentiment(
    model_version: str,
    confusion_matrix_path: Path,
    model_report_path: Path,
    output_prefix: str,
) -> Dict:
    """
    Evaluate business sentiment metrics from an emotion confusion matrix.
    """
    if not confusion_matrix_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {confusion_matrix_path}")

    confusion_df = pd.read_csv(confusion_matrix_path, index_col=0)
    model_report = load_json(model_report_path)

    y_true_emotions, y_pred_emotions = expand_confusion_matrix_to_labels(confusion_df)

    y_true_business = map_to_business_sentiment(y_true_emotions)
    y_pred_business = map_to_business_sentiment(y_pred_emotions)

    raw_emotion_accuracy = accuracy_score(y_true_emotions, y_pred_emotions)
    business_accuracy = accuracy_score(y_true_business, y_pred_business)

    business_macro_f1 = f1_score(
        y_true_business,
        y_pred_business,
        labels=BUSINESS_LABELS,
        average="macro",
        zero_division=0,
    )

    business_weighted_f1 = f1_score(
        y_true_business,
        y_pred_business,
        labels=BUSINESS_LABELS,
        average="weighted",
        zero_division=0,
    )

    business_matrix = confusion_matrix(
        y_true_business,
        y_pred_business,
        labels=BUSINESS_LABELS,
    )

    business_matrix_df = pd.DataFrame(
        business_matrix,
        index=[f"actual_{label}" for label in BUSINESS_LABELS],
        columns=[f"predicted_{label}" for label in BUSINESS_LABELS],
    )

    output_report_path = REPORTS_DIR / f"{output_prefix}_business_sentiment_report.json"
    output_matrix_path = REPORTS_DIR / f"{output_prefix}_business_sentiment_confusion_matrix.csv"

    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    business_matrix_df.to_csv(output_matrix_path)

    report = {
        "model_version": model_version,
        "run_name": model_report.get("run_name"),
        "model_name": model_report.get("model_name"),
        "base_checkpoint": model_report.get("base_checkpoint"),
        "dataset_source": model_report.get(
            "dataset_source",
            model_report.get("dataset", "Unknown"),
        ),
        "evaluation_type": "business_sentiment_from_emotion_confusion_matrix",
        "business_label_mapping": EMOTION_TO_BUSINESS_SENTIMENT,
        "total_test_samples": len(y_true_business),
        "raw_emotion_accuracy": float(raw_emotion_accuracy),
        "business_sentiment_accuracy": float(business_accuracy),
        "business_sentiment_macro_f1": float(business_macro_f1),
        "business_sentiment_weighted_f1": float(business_weighted_f1),
        "business_sentiment_classification_report": classification_report(
            y_true_business,
            y_pred_business,
            labels=BUSINESS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "business_sentiment_confusion_matrix_path": str(output_matrix_path),
        "notes": (
            "Business sentiment groups anger, disgust, fear, and sadness as "
            "Negative/Escalated, neutral as Neutral, and happy as Positive/Calm. "
            "This metric is more aligned with call-center escalation detection "
            "than exact 6-class emotion accuracy alone."
        ),
    }

    with output_report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nBusiness Sentiment Evaluation")
    print("-" * 70)
    print(f"Model version: {model_version}")
    print(f"Dataset source: {report['dataset_source']}")
    print(f"Total test samples: {report['total_test_samples']}")
    print(f"Raw emotion accuracy: {report['raw_emotion_accuracy']:.4f}")
    print(f"Business sentiment accuracy: {report['business_sentiment_accuracy']:.4f}")
    print(f"Business sentiment macro F1: {report['business_sentiment_macro_f1']:.4f}")
    print(f"Business sentiment weighted F1: {report['business_sentiment_weighted_f1']:.4f}")
    print("-" * 70)
    print(f"Saved report to: {output_report_path}")
    print(f"Saved confusion matrix to: {output_matrix_path}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate business sentiment metrics from an emotion confusion matrix."
    )

    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Model version name, e.g., model_v2_cremad_ravdess.",
    )

    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        required=True,
        help="Path to 6-class emotion confusion matrix CSV.",
    )

    parser.add_argument(
        "--model-report-path",
        type=Path,
        required=True,
        help="Path to model report JSON.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Prefix for saved business sentiment report files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_business_sentiment(
        model_version=args.model_version,
        confusion_matrix_path=args.confusion_matrix_path,
        model_report_path=args.model_report_path,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()