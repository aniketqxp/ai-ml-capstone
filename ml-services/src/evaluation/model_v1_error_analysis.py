"""
Model V1 error analysis.

This script analyzes:
    1. Raw 6-class emotion confusion matrix
    2. Business-level sentiment confusion matrix

It generates:
    - JSON error analysis report
    - Markdown error analysis report

Run from ml-services:

    python -m src.evaluation.model_v1_error_analysis
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

EMOTION_CONFUSION_MATRIX_PATH = REPORTS_DIR / "model_v1_confusion_matrix.csv"
BUSINESS_CONFUSION_MATRIX_PATH = (
    REPORTS_DIR / "model_v1_business_sentiment_confusion_matrix.csv"
)
MODEL_V1_SUMMARY_PATH = REPORTS_DIR / "model_v1_summary.json"

ERROR_ANALYSIS_JSON_PATH = REPORTS_DIR / "model_v1_error_analysis.json"
ERROR_ANALYSIS_MD_PATH = REPORTS_DIR / "model_v1_error_analysis.md"


def normalize_matrix_label(label: str) -> str:
    """
    Normalize row/column names from confusion matrix.

    Examples:
        actual_anger -> anger
        predicted_fear -> fear
        actual_Negative/Escalated -> Negative/Escalated
    """
    return (
        str(label)
        .replace("actual_", "")
        .replace("predicted_", "")
        .strip()
    )


def load_csv_matrix(path: Path) -> pd.DataFrame:
    """
    Load a confusion matrix CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required confusion matrix: {path}")

    return pd.read_csv(path, index_col=0)


def load_json(path: Path) -> Dict:
    """
    Load JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def analyze_per_class_performance(confusion_df: pd.DataFrame) -> List[Dict]:
    """
    Calculate per-class performance from a confusion matrix.

    For each class:
        - support
        - correct predictions
        - recall
        - most common wrong prediction
    """
    results = []

    for row_label in confusion_df.index:
        class_name = normalize_matrix_label(row_label)
        row = confusion_df.loc[row_label]

        support = int(row.sum())
        correct_column = f"predicted_{class_name}"

        correct = int(row.get(correct_column, 0))
        recall = correct / support if support > 0 else 0.0

        wrong_predictions = row.drop(labels=[correct_column], errors="ignore")
        wrong_predictions = wrong_predictions[wrong_predictions > 0]

        if not wrong_predictions.empty:
            most_confused_column = wrong_predictions.idxmax()
            most_confused_class = normalize_matrix_label(most_confused_column)
            most_confused_count = int(wrong_predictions.max())
        else:
            most_confused_class = None
            most_confused_count = 0

        results.append(
            {
                "class_name": class_name,
                "support": support,
                "correct": correct,
                "recall": float(recall),
                "most_confused_with": most_confused_class,
                "most_confused_count": most_confused_count,
            }
        )

    results.sort(key=lambda item: item["recall"], reverse=True)
    return results


def extract_top_confusions(
    confusion_df: pd.DataFrame,
    top_n: int = 10,
) -> List[Dict]:
    """
    Extract the largest off-diagonal confusion pairs.

    Example:
        actual_sadness predicted_fear = 35
    """
    confusions = []

    for row_label in confusion_df.index:
        actual_class = normalize_matrix_label(row_label)

        for column_label in confusion_df.columns:
            predicted_class = normalize_matrix_label(column_label)

            if actual_class == predicted_class:
                continue

            count = int(confusion_df.loc[row_label, column_label])

            if count > 0:
                confusions.append(
                    {
                        "actual": actual_class,
                        "predicted": predicted_class,
                        "count": count,
                    }
                )

    confusions.sort(key=lambda item: item["count"], reverse=True)
    return confusions[:top_n]


def build_interpretation(
    emotion_performance: List[Dict],
    emotion_confusions: List[Dict],
    business_confusions: List[Dict],
) -> Dict:
    """
    Build human-readable interpretation and recommended fixes.
    """
    weakest_classes = sorted(
        emotion_performance,
        key=lambda item: item["recall"],
    )[:3]

    strongest_classes = sorted(
        emotion_performance,
        key=lambda item: item["recall"],
        reverse=True,
    )[:3]

    recommendations = [
        {
            "issue": "Sadness is weaker than other classes and is often confused with fear or neutral.",
            "reason": "Sad, worried, tired, or low-energy speech can be acoustically similar.",
            "recommended_fix": "Add RAVDESS examples, apply light augmentation, and check sadness/fear label mapping carefully.",
        },
        {
            "issue": "Happy speech is sometimes confused with anger or negative emotion.",
            "reason": "Happy and angry speech can both be high-energy with higher pitch or louder voice.",
            "recommended_fix": "Use business-level sentiment accuracy and consider voice-feature support to distinguish positive high energy from aggressive high energy.",
        },
        {
            "issue": "Some negative emotions are confused with each other.",
            "reason": "Anger, disgust, fear, and sadness are all negative states and can overlap in vocal tone.",
            "recommended_fix": "Report both raw emotion accuracy and business sentiment accuracy, since the business use case mainly needs escalation/negative-state detection.",
        },
        {
            "issue": "The model is trained on acted emotional speech.",
            "reason": "CREMA-D is useful for labeled emotion learning but not identical to real call-center calls.",
            "recommended_fix": "Use AppTek for realistic inference/demo and manually label a small AppTek subset if accuracy on call-center-like audio is needed.",
        },
    ]

    return {
        "strongest_classes": strongest_classes,
        "weakest_classes": weakest_classes,
        "main_emotion_confusions": emotion_confusions[:5],
        "main_business_confusions": business_confusions[:5],
        "recommendations_for_model_v2": recommendations,
    }


def create_markdown_report(report: Dict) -> str:
    """
    Create a clean Markdown error analysis report.
    """
    lines = []

    lines.append("# Model V1 Error Analysis")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    lines.append(f"- Model version: {report['model_version']}")
    lines.append(f"- Model name: {report['model_name']}")
    lines.append(f"- Base checkpoint: {report['base_checkpoint']}")
    lines.append(f"- Dataset: {report['dataset']}")
    lines.append(f"- Raw emotion accuracy: {report['raw_emotion_accuracy']:.4f}")
    lines.append(f"- Business sentiment accuracy: {report['business_sentiment_accuracy']:.4f}")
    lines.append("")

    lines.append("## Strongest Emotion Classes")
    lines.append("")
    for item in report["interpretation"]["strongest_classes"]:
        lines.append(
            f"- {item['class_name']}: recall={item['recall']:.4f} "
            f"({item['correct']}/{item['support']} correct)"
        )
    lines.append("")

    lines.append("## Weakest Emotion Classes")
    lines.append("")
    for item in report["interpretation"]["weakest_classes"]:
        confused_with = item["most_confused_with"] or "none"
        lines.append(
            f"- {item['class_name']}: recall={item['recall']:.4f} "
            f"({item['correct']}/{item['support']} correct), "
            f"most confused with {confused_with} "
            f"({item['most_confused_count']} cases)"
        )
    lines.append("")

    lines.append("## Top Emotion Confusions")
    lines.append("")
    lines.append("| Actual Emotion | Predicted Emotion | Count |")
    lines.append("|---|---:|---:|")
    for item in report["top_emotion_confusions"]:
        lines.append(f"| {item['actual']} | {item['predicted']} | {item['count']} |")
    lines.append("")

    lines.append("## Top Business Sentiment Confusions")
    lines.append("")
    lines.append("| Actual Sentiment | Predicted Sentiment | Count |")
    lines.append("|---|---:|---:|")
    for item in report["top_business_confusions"]:
        lines.append(f"| {item['actual']} | {item['predicted']} | {item['count']} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Model V1 performs strongly on the main call-center sentiment objective. "
        "The raw 6-class emotion accuracy is lower than the business-level sentiment "
        "accuracy because several raw emotion mistakes still fall within the same "
        "business category. For example, fear predicted as sadness is wrong at the "
        "emotion level but still correct as Negative/Escalated."
    )
    lines.append("")

    lines.append("## Recommended Improvements for Model V2")
    lines.append("")
    for item in report["interpretation"]["recommendations_for_model_v2"]:
        lines.append(f"### {item['issue']}")
        lines.append("")
        lines.append(f"- Reason: {item['reason']}")
        lines.append(f"- Recommended fix: {item['recommended_fix']}")
        lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Add RAVDESS support and standardize labels.")
    lines.append("2. Train Model V2 on CREMA-D + RAVDESS.")
    lines.append("3. Track V2 experiments with MLflow.")
    lines.append("4. Compare V1 and V2 using raw emotion and business sentiment metrics.")
    lines.append("5. Use AppTek for realistic call-center inference/demo.")
    lines.append("")

    return "\n".join(lines)


def run_error_analysis() -> Dict:
    """
    Run full Model V1 error analysis and save JSON + Markdown reports.
    """
    model_summary = load_json(MODEL_V1_SUMMARY_PATH)
    emotion_confusion_df = load_csv_matrix(EMOTION_CONFUSION_MATRIX_PATH)
    business_confusion_df = load_csv_matrix(BUSINESS_CONFUSION_MATRIX_PATH)

    emotion_performance = analyze_per_class_performance(emotion_confusion_df)
    business_performance = analyze_per_class_performance(business_confusion_df)

    top_emotion_confusions = extract_top_confusions(emotion_confusion_df, top_n=10)
    top_business_confusions = extract_top_confusions(business_confusion_df, top_n=10)

    raw_emotion_accuracy = float(model_summary["test"]["accuracy"])

    # Business accuracy from business confusion matrix.
    business_correct = int(
        sum(
            business_confusion_df.loc[row, f"predicted_{normalize_matrix_label(row)}"]
            for row in business_confusion_df.index
            if f"predicted_{normalize_matrix_label(row)}" in business_confusion_df.columns
        )
    )
    business_total = int(business_confusion_df.values.sum())
    business_accuracy = business_correct / business_total if business_total else 0.0

    report = {
        "model_version": "model_v1",
        "model_name": model_summary.get("model_name"),
        "base_checkpoint": model_summary.get("base_checkpoint"),
        "dataset": "CREMA-D",
        "raw_emotion_accuracy": raw_emotion_accuracy,
        "business_sentiment_accuracy": float(business_accuracy),
        "emotion_per_class_performance": emotion_performance,
        "business_per_class_performance": business_performance,
        "top_emotion_confusions": top_emotion_confusions,
        "top_business_confusions": top_business_confusions,
    }

    report["interpretation"] = build_interpretation(
        emotion_performance=emotion_performance,
        emotion_confusions=top_emotion_confusions,
        business_confusions=top_business_confusions,
    )

    ERROR_ANALYSIS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ERROR_ANALYSIS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    markdown_report = create_markdown_report(report)

    with ERROR_ANALYSIS_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(markdown_report)

    return report


def main() -> None:
    report = run_error_analysis()

    print("\nModel V1 Error Analysis")
    print("-" * 70)
    print(f"Raw emotion accuracy: {report['raw_emotion_accuracy']:.4f}")
    print(f"Business sentiment accuracy: {report['business_sentiment_accuracy']:.4f}")
    print("\nStrongest classes:")
    for item in report["interpretation"]["strongest_classes"]:
        print(f"  - {item['class_name']}: recall={item['recall']:.4f}")

    print("\nWeakest classes:")
    for item in report["interpretation"]["weakest_classes"]:
        print(
            f"  - {item['class_name']}: recall={item['recall']:.4f}, "
            f"most confused with {item['most_confused_with']}"
        )

    print("\nTop confusions:")
    for item in report["top_emotion_confusions"][:5]:
        print(f"  - {item['actual']} -> {item['predicted']}: {item['count']}")

    print("-" * 70)
    print(f"Saved JSON report to: {ERROR_ANALYSIS_JSON_PATH}")
    print(f"Saved Markdown report to: {ERROR_ANALYSIS_MD_PATH}")


if __name__ == "__main__":
    main()