"""
Compare Model V1 and Model V2 results.

Model V1:
    CREMA-D only

Model V2:
    CREMA-D + RAVDESS

Run from ml-services:

    python -m src.evaluation.compare_model_v1_v2
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"
REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

V1_REPORT_PATH = REPORTS_DIR / "model_v1_summary.json"
V1_BUSINESS_REPORT_PATH = REPORTS_DIR / "model_v1_business_sentiment_report.json"

V2_REPORT_PATH = REPORTS_DIR / "model_v2_cremad_ravdess_report.json"
V2_BUSINESS_REPORT_PATH = (
    REPORTS_DIR / "model_v2_cremad_ravdess_business_sentiment_report.json"
)

COMPARISON_JSON_PATH = REPORTS_DIR / "model_v1_vs_v2_comparison.json"
COMPARISON_MD_PATH = REPORTS_DIR / "model_v1_vs_v2_comparison.md"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_comparison() -> Dict:
    v1 = load_json(V1_REPORT_PATH)
    v1_business = load_json(V1_BUSINESS_REPORT_PATH)

    v2 = load_json(V2_REPORT_PATH)
    v2_business = load_json(V2_BUSINESS_REPORT_PATH)

    model_v1 = {
        "name": "Model V1",
        "dataset_source": "CREMA-D",
        "run_name": "model_v1_wav2vec2_cremad",
        "raw_emotion_accuracy": float(v1["test"]["accuracy"]),
        "raw_emotion_macro_f1": float(v1["test"]["macro_f1"]),
        "business_sentiment_accuracy": float(
            v1_business["business_sentiment_accuracy"]
        ),
        "business_sentiment_macro_f1": float(
            v1_business["business_sentiment_macro_f1"]
        ),
    }

    model_v2 = {
        "name": "Model V2",
        "dataset_source": v2.get("dataset_source"),
        "run_name": v2.get("run_name"),
        "raw_emotion_accuracy": float(v2["test"]["accuracy"]),
        "raw_emotion_macro_f1": float(v2["test"]["macro_f1"]),
        "business_sentiment_accuracy": float(
            v2_business["business_sentiment_accuracy"]
        ),
        "business_sentiment_macro_f1": float(
            v2_business["business_sentiment_macro_f1"]
        ),
    }

    improvements = {
        "raw_emotion_accuracy_delta": (
            model_v2["raw_emotion_accuracy"] - model_v1["raw_emotion_accuracy"]
        ),
        "raw_emotion_macro_f1_delta": (
            model_v2["raw_emotion_macro_f1"] - model_v1["raw_emotion_macro_f1"]
        ),
        "business_sentiment_accuracy_delta": (
            model_v2["business_sentiment_accuracy"]
            - model_v1["business_sentiment_accuracy"]
        ),
        "business_sentiment_macro_f1_delta": (
            model_v2["business_sentiment_macro_f1"]
            - model_v1["business_sentiment_macro_f1"]
        ),
    }

    return {
        "model_v1": model_v1,
        "model_v2": model_v2,
        "improvements": improvements,
        "interpretation": {
            "summary": (
                "Model V2 was trained on CREMA-D + RAVDESS to improve generalization "
                "beyond the CREMA-D-only baseline. The comparison evaluates both raw "
                "6-class emotion accuracy and business-level sentiment accuracy."
            ),
            "business_relevance": (
                "Business sentiment accuracy is important because the capstone use case "
                "is call-center risk and escalation detection, where identifying "
                "Negative/Escalated, Neutral, and Positive/Calm states is more directly "
                "useful than exact emotion labels alone."
            ),
        },
    }


def create_markdown(comparison: Dict) -> str:
    v1 = comparison["model_v1"]
    v2 = comparison["model_v2"]
    improvements = comparison["improvements"]

    lines = []

    lines.append("# Model V1 vs Model V2 Comparison")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "Model V1 was trained on CREMA-D only. Model V2 was trained on the combined "
        "CREMA-D + RAVDESS dataset to improve generalization and reduce dependence "
        "on one emotional speech dataset."
    )
    lines.append("")

    lines.append("## Dataset Setup")
    lines.append("")
    lines.append("| Model | Dataset Source | Purpose |")
    lines.append("|---|---|---|")
    lines.append("| Model V1 | CREMA-D | Baseline supervised emotion model |")
    lines.append("| Model V2 | CREMA-D + RAVDESS | Improved/generalized emotion model |")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Model V1 | Model V2 | Change |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Raw emotion accuracy | {percent(v1['raw_emotion_accuracy'])} | "
        f"{percent(v2['raw_emotion_accuracy'])} | "
        f"{improvements['raw_emotion_accuracy_delta'] * 100:+.2f} pts |"
    )
    lines.append(
        f"| Raw emotion macro F1 | {percent(v1['raw_emotion_macro_f1'])} | "
        f"{percent(v2['raw_emotion_macro_f1'])} | "
        f"{improvements['raw_emotion_macro_f1_delta'] * 100:+.2f} pts |"
    )
    lines.append(
        f"| Business sentiment accuracy | {percent(v1['business_sentiment_accuracy'])} | "
        f"{percent(v2['business_sentiment_accuracy'])} | "
        f"{improvements['business_sentiment_accuracy_delta'] * 100:+.2f} pts |"
    )
    lines.append(
        f"| Business sentiment macro F1 | {percent(v1['business_sentiment_macro_f1'])} | "
        f"{percent(v2['business_sentiment_macro_f1'])} | "
        f"{improvements['business_sentiment_macro_f1_delta'] * 100:+.2f} pts |"
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Model V2 slightly improves raw emotion accuracy and macro F1 compared with Model V1. "
        "The improvement may be modest, but Model V2 is trained on a broader emotional speech "
        "dataset, which makes it more suitable for generalization than a CREMA-D-only model."
    )
    lines.append("")
    lines.append(
        "The business-level sentiment metric remains important because the project goal is "
        "call-center risk detection. In this setting, confusing fear with sadness is less severe "
        "than confusing Negative/Escalated speech with Neutral or Positive/Calm speech."
    )
    lines.append("")

    lines.append("## Next Improvement Direction")
    lines.append("")
    lines.append(
        "The next accuracy-improvement experiments should focus on hyperparameter tuning, "
        "freezing/unfreezing Wav2Vec2 layers, training-only audio augmentation, and "
        "dataset-specific evaluation on CREMA-D and RAVDESS separately."
    )
    lines.append("")
    lines.append(
        "After these supervised experiments, AppTek should be used for realistic call-center "
        "inference/demo. If manual labels are added for AppTek segments, it can also be used "
        "for domain-specific business sentiment evaluation."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    comparison = build_comparison()

    with COMPARISON_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    markdown = create_markdown(comparison)

    with COMPARISON_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(markdown)

    print("\nModel V1 vs Model V2 Comparison")
    print("-" * 80)

    rows = [
        {
            "Metric": "Raw emotion accuracy",
            "Model V1": percent(comparison["model_v1"]["raw_emotion_accuracy"]),
            "Model V2": percent(comparison["model_v2"]["raw_emotion_accuracy"]),
            "Change": f"{comparison['improvements']['raw_emotion_accuracy_delta'] * 100:+.2f} pts",
        },
        {
            "Metric": "Raw emotion macro F1",
            "Model V1": percent(comparison["model_v1"]["raw_emotion_macro_f1"]),
            "Model V2": percent(comparison["model_v2"]["raw_emotion_macro_f1"]),
            "Change": f"{comparison['improvements']['raw_emotion_macro_f1_delta'] * 100:+.2f} pts",
        },
        {
            "Metric": "Business sentiment accuracy",
            "Model V1": percent(comparison["model_v1"]["business_sentiment_accuracy"]),
            "Model V2": percent(comparison["model_v2"]["business_sentiment_accuracy"]),
            "Change": f"{comparison['improvements']['business_sentiment_accuracy_delta'] * 100:+.2f} pts",
        },
        {
            "Metric": "Business sentiment macro F1",
            "Model V1": percent(comparison["model_v1"]["business_sentiment_macro_f1"]),
            "Model V2": percent(comparison["model_v2"]["business_sentiment_macro_f1"]),
            "Change": f"{comparison['improvements']['business_sentiment_macro_f1_delta'] * 100:+.2f} pts",
        },
    ]

    print(pd.DataFrame(rows).to_string(index=False))
    print("-" * 80)
    print(f"Saved JSON to: {COMPARISON_JSON_PATH}")
    print(f"Saved Markdown to: {COMPARISON_MD_PATH}")


if __name__ == "__main__":
    main()