"""
Generate a 3-domain AppTek inference report.

This report summarizes selected V5 emotion/sentiment inference on AppTek
banking, healthcare, and telecommunications samples.

Run from ml-services:

    python -m src.reports.apptek_domain_inference_report
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

SUMMARY_CSV_PATH = ML_SERVICES_ROOT / "outputs" / "apptek" / "apptek_sentiment_summary.csv"

OUTPUT_JSON_PATH = (
    ML_SERVICES_ROOT / "outputs" / "reports" / "apptek_domain_inference_report.json"
)

OUTPUT_MD_PATH = (
    ML_SERVICES_ROOT / "outputs" / "reports" / "apptek_domain_inference_report.md"
)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"AppTek summary CSV not found: {SUMMARY_CSV_PATH}\n"
            "Run selected-domain AppTek inference first."
        )

    df = pd.read_csv(SUMMARY_CSV_PATH)

    required_columns = {
        "call_id",
        "domain",
        "raw_domain",
        "dominant_emotion",
        "overall_audio_sentiment",
        "audio_escalation_score",
        "risk_level",
        "confidence_level",
        "model_version",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    total_calls = len(df)
    model_version = df["model_version"].iloc[0] if total_calls > 0 else "Unknown"

    sentiment_distribution = df["overall_audio_sentiment"].value_counts().to_dict()
    risk_distribution = df["risk_level"].value_counts().to_dict()
    domain_counts = df["domain"].value_counts().to_dict()

    domain_summary = []

    for domain, domain_df in df.groupby("domain"):
        domain_summary.append(
            {
                "domain": domain,
                "raw_domains": sorted(domain_df["raw_domain"].dropna().unique().tolist()),
                "num_calls": int(len(domain_df)),
                "sentiment_distribution": domain_df[
                    "overall_audio_sentiment"
                ].value_counts().to_dict(),
                "risk_distribution": domain_df["risk_level"].value_counts().to_dict(),
                "dominant_emotion_distribution": domain_df[
                    "dominant_emotion"
                ].value_counts().to_dict(),
                "average_escalation_score": float(
                    domain_df["audio_escalation_score"].mean()
                ),
                "max_escalation_score": float(
                    domain_df["audio_escalation_score"].max()
                ),
                "highest_risk_call": domain_df.sort_values(
                    "audio_escalation_score", ascending=False
                ).iloc[0]["call_id"],
            }
        )

    highest_risk_row = df.sort_values("audio_escalation_score", ascending=False).iloc[0]
    lowest_risk_row = df.sort_values("audio_escalation_score", ascending=True).iloc[0]

    report = {
        "report_name": "AppTek 3-Domain Inference Report",
        "purpose": "Realistic call-center inference/demo using the selected V5 audio sentiment model.",
        "selected_model": model_version,
        "total_calls": int(total_calls),
        "domain_counts": domain_counts,
        "sentiment_distribution": sentiment_distribution,
        "risk_distribution": risk_distribution,
        "average_escalation_score": float(df["audio_escalation_score"].mean()),
        "highest_risk_call": {
            "call_id": highest_risk_row["call_id"],
            "domain": highest_risk_row["domain"],
            "dominant_emotion": highest_risk_row["dominant_emotion"],
            "overall_audio_sentiment": highest_risk_row["overall_audio_sentiment"],
            "risk_level": highest_risk_row["risk_level"],
            "audio_escalation_score": float(highest_risk_row["audio_escalation_score"]),
        },
        "lowest_risk_call": {
            "call_id": lowest_risk_row["call_id"],
            "domain": lowest_risk_row["domain"],
            "dominant_emotion": lowest_risk_row["dominant_emotion"],
            "overall_audio_sentiment": lowest_risk_row["overall_audio_sentiment"],
            "risk_level": lowest_risk_row["risk_level"],
            "audio_escalation_score": float(lowest_risk_row["audio_escalation_score"]),
        },
        "domain_summary": domain_summary,
        "important_note": (
            "AppTek is used here for realistic call-center inference and demo. "
            "These results are not supervised accuracy results because AppTek does not provide "
            "ground-truth emotion labels for this project task."
        ),
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    lines = []

    lines.append("# AppTek 3-Domain Inference Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report summarizes realistic call-center inference using the selected "
        "V5 audio sentiment model on AppTek samples from three domains: banking, "
        "healthcare, and telecommunications."
    )
    lines.append("")
    lines.append("## Selected Model")
    lines.append("")
    lines.append(f"- Model version: `{model_version}`")
    lines.append("- Model role: audio emotion and business sentiment inference")
    lines.append("")
    lines.append("## Dataset Setup")
    lines.append("")
    lines.append(f"- Total AppTek calls processed: {total_calls}")
    lines.append("- Samples per selected domain: 3")
    lines.append("- Processed duration per call: first 60 seconds")
    lines.append("- Domains used:")
    for domain, count in domain_counts.items():
        lines.append(f"  - {domain}: {count} calls")
    lines.append("")
    lines.append("## Overall Results")
    lines.append("")
    lines.append("### Sentiment Distribution")
    lines.append("")
    for sentiment, count in sentiment_distribution.items():
        lines.append(f"- {sentiment}: {count}")
    lines.append("")
    lines.append("### Risk Distribution")
    lines.append("")
    for risk, count in risk_distribution.items():
        lines.append(f"- {risk}: {count}")
    lines.append("")
    lines.append(
        f"- Average escalation score: {report['average_escalation_score']:.4f}"
    )
    lines.append("")
    lines.append("## Domain-Level Summary")
    lines.append("")
    lines.append(
        "| Domain | Calls | Sentiment Distribution | Risk Distribution | Avg Escalation | Highest-Risk Call |"
    )
    lines.append("|---|---:|---|---|---:|---|")

    for item in domain_summary:
        sentiment_text = ", ".join(
            f"{key}: {value}" for key, value in item["sentiment_distribution"].items()
        )
        risk_text = ", ".join(
            f"{key}: {value}" for key, value in item["risk_distribution"].items()
        )
        lines.append(
            f"| {item['domain']} | {item['num_calls']} | {sentiment_text} | "
            f"{risk_text} | {item['average_escalation_score']:.4f} | "
            f"{item['highest_risk_call']} |"
        )

    lines.append("")
    lines.append("## Highest-Risk Call")
    lines.append("")
    lines.append(f"- Call ID: `{report['highest_risk_call']['call_id']}`")
    lines.append(f"- Domain: {report['highest_risk_call']['domain']}")
    lines.append(f"- Dominant emotion: {report['highest_risk_call']['dominant_emotion']}")
    lines.append(
        f"- Overall sentiment: {report['highest_risk_call']['overall_audio_sentiment']}"
    )
    lines.append(f"- Risk level: {report['highest_risk_call']['risk_level']}")
    lines.append(
        f"- Escalation score: {report['highest_risk_call']['audio_escalation_score']:.4f}"
    )
    lines.append("")
    lines.append("## Important Limitation")
    lines.append("")
    lines.append(
        "AppTek is used for realistic call-center inference/demo, not supervised "
        "emotion accuracy. The supervised accuracy was measured on CREMA-D + RAVDESS. "
        "AppTek does not provide ground-truth emotion labels for this project task."
    )
    lines.append("")
    lines.append("## Output Generated")
    lines.append("")
    lines.append(
        "For each AppTek call, the system generated dominant emotion, business sentiment, "
        "confidence level, uncertainty flag, escalation score, risk level, and sentiment timeline."
    )

    with OUTPUT_MD_PATH.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print("\nAppTek domain inference report generated successfully.")
    print("-" * 80)
    print(f"Saved JSON report to: {OUTPUT_JSON_PATH}")
    print(f"Saved Markdown report to: {OUTPUT_MD_PATH}")
    print("-" * 80)

    print("\nDomain summary:")
    for item in domain_summary:
        print(
            f"{item['domain']}: calls={item['num_calls']}, "
            f"avg_escalation={item['average_escalation_score']:.4f}, "
            f"highest_risk_call={item['highest_risk_call']}"
        )


if __name__ == "__main__":
    main()