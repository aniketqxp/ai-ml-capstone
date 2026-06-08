"""
Run selected V5 emotion/sentiment inference on AppTek call-center samples.

This script uses the final selected V5 model through EmotionPredictor and
generates realistic call-center inference outputs.

Run smoke test from ml-services:

    python -m src.inference.run_apptek_inference --max-calls 2 --max-duration-seconds 30

Run full AppTek sample:

    python -m src.inference.run_apptek_inference --max-calls 20 --max-duration-seconds 60
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.inference.emotion_predictor import (
    EmotionPredictor,
    SELECTED_EMOTION_MODEL_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

APPTEK_METADATA_PATH = (
    ML_SERVICES_ROOT / "data" / "processed" / "apptek" / "apptek_metadata.csv"
)

APPTEK_RESULTS_DIR = (
    ML_SERVICES_ROOT / "outputs" / "apptek" / "sentiment_results"
)

APPTEK_SUMMARY_PATH = (
    ML_SERVICES_ROOT / "outputs" / "apptek" / "apptek_sentiment_summary.csv"
)


def result_to_dict(result: Any) -> Dict:
    """
    Convert AudioSentimentResult to dictionary for JSON saving.
    Supports Pydantic v2, Pydantic v1, or dataclass-like objects.
    """
    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "dict"):
        return result.dict()

    if hasattr(result, "__dict__"):
        return result.__dict__

    raise TypeError(f"Cannot convert result to dict: {type(result)}")


def run_apptek_inference(
    metadata_path: Path = APPTEK_METADATA_PATH,
    max_calls: Optional[int] = None,
    max_duration_seconds: float = 60.0,
    build_timeline: bool = True,
) -> None:
    """
    Run selected emotion model on AppTek samples.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"AppTek metadata not found: {metadata_path}\n"
            "Run python -m src.data.apptek_dataset first."
        )

    metadata = pd.read_csv(metadata_path)

    if max_calls is not None:
        metadata = metadata.head(max_calls)

    APPTEK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    APPTEK_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictor = EmotionPredictor(max_duration_seconds=max_duration_seconds)

    summary_rows = []

    print("\nRunning AppTek inference")
    print("-" * 80)
    print(f"Selected model: {SELECTED_EMOTION_MODEL_VERSION}")
    print(f"Metadata: {metadata_path}")
    print(f"Calls to process: {len(metadata)}")
    print(f"Max duration per call: {max_duration_seconds} seconds")
    print(f"Build timeline: {build_timeline}")
    print("-" * 80)

    for _, row in metadata.iterrows():
        call_id = row["call_id"]
        audio_path = ML_SERVICES_ROOT / row["audio_path"]
        domain = row.get("selected_domain", row.get("domain", ""))
        raw_domain = row.get("raw_domain", domain)

        print(f"Processing {call_id} | domain={domain} | raw_domain={raw_domain} | audio={audio_path.name}")

        result = predictor.analyze_audio(
            audio_path=audio_path,
            call_id=call_id,
            build_timeline=build_timeline,
        )

        result_dict = result_to_dict(result)

        # Add AppTek metadata context to output.
        result_dict["apptek_metadata"] = {
            "domain": domain,
            "raw_domain": raw_domain,
            "gender": row.get("gender", ""),
            "accent": row.get("accent", ""),
            "duration_seconds": float(row.get("duration_seconds", 0.0)),
            "text_preview": str(row.get("text", ""))[:500],
        }

        output_path = APPTEK_RESULTS_DIR / f"{call_id}_sentiment.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result_dict, file, indent=2)

        summary_rows.append(
            {
                "call_id": call_id,
                "domain": domain,
                "raw_domain": raw_domain,
                "gender": row.get("gender", ""),
                "accent": row.get("accent", ""),
                "duration_seconds": row.get("duration_seconds", 0.0),
                "processed_duration_seconds": max_duration_seconds,
                "overall_audio_sentiment": result_dict.get("overall_audio_sentiment"),
                "dominant_emotion": result_dict.get("dominant_emotion"),
                "negative_emotion_probability": result_dict.get(
                    "negative_emotion_probability"
                ),
                "anger_probability": result_dict.get("anger_probability"),
                "stress_probability": result_dict.get("stress_probability"),
                "sadness_probability": result_dict.get("sadness_probability"),
                "anxiety_probability": result_dict.get("anxiety_probability"),
                "calm_probability": result_dict.get("calm_probability"),
                "audio_escalation_score": result_dict.get("audio_escalation_score"),
                "risk_level": result_dict.get("risk_level"),
                "prediction_confidence": result_dict.get("prediction_confidence"),
                "confidence_level": result_dict.get("confidence_level"),
                "uncertain_prediction": result_dict.get("uncertain_prediction"),
                "model_version": result_dict.get("model_version"),
                "result_json": str(output_path.relative_to(ML_SERVICES_ROOT)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(APPTEK_SUMMARY_PATH, index=False)

    print("\nAppTek inference completed successfully.")
    print("-" * 80)
    print(f"Saved JSON results to: {APPTEK_RESULTS_DIR}")
    print(f"Saved summary CSV to: {APPTEK_SUMMARY_PATH}")
    print("-" * 80)

    print("\nSummary preview:")
    print(
        summary_df[
            [
                "call_id",
                "domain",
                "dominant_emotion",
                "overall_audio_sentiment",
                "audio_escalation_score",
                "risk_level",
                "confidence_level",
            ]
        ].head()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected V5 emotion inference on AppTek samples."
    )

    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=APPTEK_METADATA_PATH,
        help="Path to AppTek metadata CSV.",
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum number of AppTek calls to process.",
    )

    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=60.0,
        help="Maximum duration per call to process.",
    )

    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="Disable sentiment timeline generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_apptek_inference(
        metadata_path=args.metadata_path,
        max_calls=args.max_calls,
        max_duration_seconds=args.max_duration_seconds,
        build_timeline=not args.no_timeline,
    )