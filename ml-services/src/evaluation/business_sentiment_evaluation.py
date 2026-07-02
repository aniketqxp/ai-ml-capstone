"""
Business-level sentiment evaluation for Model V1.

This script converts the 6-class emotion confusion matrix into a 3-class
business sentiment evaluation:

    Negative/Escalated = anger, disgust, fear, sadness
    Neutral = neutral
    Positive/Calm = happy

Why:
    In call-center quality analysis, exact emotion classification is useful,
    but the business decision often depends on whether the caller sounds
    negative/escalated, neutral, or positive/calm.

Run from ml-services:

    python -m src.evaluation.business_sentiment_evaluation
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

MODEL_V1_EMOTION_CONFUSION_MATRIX_PATH = REPORTS_DIR / "model_v1_confusion_matrix.csv"
MODEL_V1_SUMMARY_PATH = REPORTS_DIR / "model_v1_summary.json"

BUSINESS_REPORT_PATH = REPORTS_DIR / "model_v1_business_sentiment_report.json"
BUSINESS_CONFUSION_MATRIX_PATH = (
    REPORTS_DIR / "model_v1_business_sentiment_confusion_matrix.csv"
)

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
    Normalize confusion matrix row/column labels.

    Example:
        actual_anger -> anger
        predicted_sadness -> sadness
    """
    return (
        label.replace("actual_", "")
        .replace("predicted_", "")
        .strip()
        .lower()
    )


def load_model_v1_summary() -> Dict:
    """
    Load Model V1 summary JSON.
    """
    if not MODEL_V1_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing Model V1 summary: {MODEL_V1_SUMMARY_PATH}")

    with MODEL_V1_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_emotion_confusion_matrix() -> pd.DataFrame:
    """
    Load Model V1 6-class emotion confusion matrix.
    """
    if not MODEL_V1_EMOTION_CONFUSION_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Missing Model V1 confusion matrix: {MODEL_V1_EMOTION_CONFUSION_MATRIX_PATH}"
        )

    return pd.read_csv(MODEL_V1_EMOTION_CONFUSION_MATRIX_PATH, index_col=0)


def expand_confusion_matrix_to_labels(
    confusion_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Expand confusion matrix counts back into true/predicted label lists.

    This lets us reuse sklearn accuracy/F1/classification_report functions.

    Returns:
        y_true_emotions
        y_pred_emotions
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


def map_emotions_to_business_sentiment(emotions: List[str]) -> List[str]:
    """
    Convert emotion labels into business sentiment labels.
    """
    mapped_labels = []

    for emotion in emotions:
        if emotion not in EMOTION_TO_BUSINESS_SENTIMENT:
            raise ValueError(f"Unknown emotion label: {emotion}")

        mapped_labels.append(EMOTION_TO_BUSINESS_SENTIMENT[emotion])

    return mapped_labels


def evaluate_business_sentiment() -> Dict:
    """
    Calculate business-level sentiment metrics for Model V1.
    """
    model_summary = load_model_v1_summary()
    emotion_confusion_df = load_emotion_confusion_matrix()

    y_true_emotions, y_pred_emotions = expand_confusion_matrix_to_labels(
        emotion_confusion_df
    )

    y_true_business = map_emotions_to_business_sentiment(y_true_emotions)
    y_pred_business = map_emotions_to_business_sentiment(y_pred_emotions)

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

    BUSINESS_CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    business_matrix_df.to_csv(BUSINESS_CONFUSION_MATRIX_PATH)

    business_report = classification_report(
        y_true_business,
        y_pred_business,
        labels=BUSINESS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report = {
        "model_version": "model_v1",
        "model_name": model_summary.get("model_name"),
        "base_checkpoint": model_summary.get("base_checkpoint"),
        "dataset": "CREMA-D",
        "evaluation_type": "business_sentiment_from_emotion_confusion_matrix",
        "business_label_mapping": EMOTION_TO_BUSINESS_SENTIMENT,
        "total_test_samples": len(y_true_business),
        "raw_emotion_accuracy": float(raw_emotion_accuracy),
        "business_sentiment_accuracy": float(business_accuracy),
        "business_sentiment_macro_f1": float(business_macro_f1),
        "business_sentiment_weighted_f1": float(business_weighted_f1),
        "business_sentiment_classification_report": business_report,
        "business_sentiment_confusion_matrix_path": str(
            BUSINESS_CONFUSION_MATRIX_PATH
        ),
        "notes": (
            "Business sentiment groups anger, disgust, fear, and sadness as "
            "Negative/Escalated, neutral as Neutral, and happy as Positive/Calm. "
            "This metric is more aligned with the call-center risk use case than "
            "raw emotion accuracy alone."
        ),
    }

    BUSINESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUSINESS_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


def main() -> None:
    report = evaluate_business_sentiment()

    print("\nModel V1 Business Sentiment Evaluation")
    print("-" * 70)
    print(f"Total test samples: {report['total_test_samples']}")
    print(f"Raw emotion accuracy: {report['raw_emotion_accuracy']:.4f}")
    print(
        "Business sentiment accuracy: "
        f"{report['business_sentiment_accuracy']:.4f}"
    )
    print(
        "Business sentiment macro F1: "
        f"{report['business_sentiment_macro_f1']:.4f}"
    )
    print(
        "Business sentiment weighted F1: "
        f"{report['business_sentiment_weighted_f1']:.4f}"
    )
    print("-" * 70)
    print(f"Saved report to: {BUSINESS_REPORT_PATH}")
    print(f"Saved confusion matrix to: {BUSINESS_CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()