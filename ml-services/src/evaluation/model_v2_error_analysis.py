"""
Model V2 error analysis.

This script analyzes:
    1. Raw 6-class emotion confusion matrix
    2. Business-level sentiment confusion matrix

Model V2:
    CREMA-D + RAVDESS

Run from ml-services:

    python -m src.evaluation.model_v2_error_analysis
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"
REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

MODEL_REPORT_PATH = REPORTS_DIR / "model_v2_cremad_ravdess_report.json"
EMOTION_CONFUSION_MATRIX_PATH = (
    REPORTS_DIR / "model_v2_cremad_ravdess_confusion_matrix.csv"
)
BUSINESS_CONFUSION_MATRIX_PATH = (
    REPORTS_DIR / "model_v2_cremad_ravdess_business_sentiment_confusion_matrix.csv"
)

ERROR_ANALYSIS_JSON_PATH = REPORTS_DIR / "model_v2_error_analysis.json"
ERROR_ANALYSIS_MD_PATH = REPORTS_DIR / "model_v2_error_analysis.md"


def normalize_matrix_label(label: str) -> str:
    """
    Normalize confusion matrix row/column names.

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


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {path}")

    return pd.read_csv(path, index_col=0)


def analyze_per_class_performance(confusion_df: pd.DataFrame) -> List[Dict]:
    """
    Calculate recall and main confusion for each class.
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


def extract_top_confusions(confusion_df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
    """
    Extract largest off-diagonal mistakes.
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
    top_emotion_confusions: List[Dict],
    top_business_confusions: List[Dict],
) -> Dict:
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
            "issue": "Sadness is still confused with fear and neutral.",
            "reason": "Sadness, fear, and low-energy neutral speech can overlap acoustically.",
            "recommended_fix": "Try augmentation and unfreezing deeper Wav2Vec2 layers. Also evaluate CREMA-D and RAVDESS separately to see which dataset causes this confusion.",
        },
        {
            "issue": "Positive/Calm is still sometimes predicted as Negative/Escalated.",
            "reason": "Happy speech can be high-energy and acoustically close to anger.",
            "recommended_fix": "Tune thresholds for business sentiment and consider adding acoustic feature support for positive high-energy vs aggressive high-energy speech.",
        },
        {
            "issue": "Raw accuracy improved only slightly after adding RAVDESS.",
            "reason": "RAVDESS improves data diversity, but the current training strategy may not fully adapt the model.",
            "recommended_fix": "Run controlled hyperparameter experiments and test partial unfreezing instead of only frozen feature encoder training.",
        },
        {
            "issue": "Combined test accuracy may hide dataset-specific weakness.",
            "reason": "CREMA-D and RAVDESS have different speakers, recording conditions, and acting styles.",
            "recommended_fix": "Evaluate Model V2 separately on CREMA-D test and RAVDESS test before deciding the next training experiment.",
        },
    ]

    return {
        "strongest_classes": strongest_classes,
        "weakest_classes": weakest_classes,
        "main_emotion_confusions": top_emotion_confusions[:5],
        "main_business_confusions": top_business_confusions[:5],
        "recommendations_for_model_v3": recommendations,
    }


def create_markdown_report(report: Dict) -> str:
    lines = []

    lines.append("# Model V2 Error Analysis")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    lines.append(f"- Model version: {report['model_version']}")
    lines.append(f"- Run name: {report['run_name']}")
    lines.append(f"- Dataset source: {report['dataset_source']}")
    lines.append(f"- Raw emotion accuracy: {report['raw_emotion_accuracy']:.4f}")
    lines.append(f"- Raw emotion macro F1: {report['raw_emotion_macro_f1']:.4f}")
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
        "Model V2 improves raw emotion accuracy slightly compared with Model V1, "
        "but the remaining errors show that some emotional states are still acoustically close. "
        "The largest issue remains confusion among sadness, fear, and some neutral samples. "
        "At the business level, the most important remaining issue is Positive/Calm being "
        "misclassified as Negative/Escalated."
    )
    lines.append("")

    lines.append("## Recommended Improvements for Model V3")
    lines.append("")
    for item in report["interpretation"]["recommendations_for_model_v3"]:
        lines.append(f"### {item['issue']}")
        lines.append("")
        lines.append(f"- Reason: {item['reason']}")
        lines.append(f"- Recommended fix: {item['recommended_fix']}")
        lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Evaluate Model V2 separately on CREMA-D test and RAVDESS test.")
    lines.append("2. Run controlled hyperparameter tuning experiments.")
    lines.append("3. Try partial unfreezing of Wav2Vec2 layers.")
    lines.append("4. Add training-only audio augmentation.")
    lines.append("5. Use AppTek for realistic call-center inference/demo.")
    lines.append("")

    return "\n".join(lines)


def run_error_analysis() -> Dict:
    model_report = load_json(MODEL_REPORT_PATH)
    emotion_confusion_df = load_csv_matrix(EMOTION_CONFUSION_MATRIX_PATH)
    business_confusion_df = load_csv_matrix(BUSINESS_CONFUSION_MATRIX_PATH)

    emotion_performance = analyze_per_class_performance(emotion_confusion_df)
    business_performance = analyze_per_class_performance(business_confusion_df)

    top_emotion_confusions = extract_top_confusions(emotion_confusion_df, top_n=10)
    top_business_confusions = extract_top_confusions(business_confusion_df, top_n=10)

    business_correct = 0
    for row in business_confusion_df.index:
        label = normalize_matrix_label(row)
        column = f"predicted_{label}"
        if column in business_confusion_df.columns:
            business_correct += int(business_confusion_df.loc[row, column])

    business_total = int(business_confusion_df.values.sum())
    business_accuracy = business_correct / business_total if business_total else 0.0

    report = {
        "model_version": "model_v2",
        "run_name": model_report.get("run_name"),
        "dataset_source": model_report.get("dataset_source"),
        "raw_emotion_accuracy": float(model_report["test"]["accuracy"]),
        "raw_emotion_macro_f1": float(model_report["test"]["macro_f1"]),
        "business_sentiment_accuracy": float(business_accuracy),
        "emotion_per_class_performance": emotion_performance,
        "business_per_class_performance": business_performance,
        "top_emotion_confusions": top_emotion_confusions,
        "top_business_confusions": top_business_confusions,
    }

    report["interpretation"] = build_interpretation(
        emotion_performance=emotion_performance,
        top_emotion_confusions=top_emotion_confusions,
        top_business_confusions=top_business_confusions,
    )

    with ERROR_ANALYSIS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    markdown = create_markdown_report(report)

    with ERROR_ANALYSIS_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(markdown)

    return report


def main() -> None:
    report = run_error_analysis()

    print("\nModel V2 Error Analysis")
    print("-" * 80)
    print(f"Dataset source: {report['dataset_source']}")
    print(f"Raw emotion accuracy: {report['raw_emotion_accuracy']:.4f}")
    print(f"Raw emotion macro F1: {report['raw_emotion_macro_f1']:.4f}")
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

    print("\nTop emotion confusions:")
    for item in report["top_emotion_confusions"][:5]:
        print(f"  - {item['actual']} -> {item['predicted']}: {item['count']}")

    print("\nTop business confusions:")
    for item in report["top_business_confusions"][:5]:
        print(f"  - {item['actual']} -> {item['predicted']}: {item['count']}")

    print("-" * 80)
    print(f"Saved JSON report to: {ERROR_ANALYSIS_JSON_PATH}")
    print(f"Saved Markdown report to: {ERROR_ANALYSIS_MD_PATH}")


if __name__ == "__main__":
    main()