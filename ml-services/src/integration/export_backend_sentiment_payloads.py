import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_INPUT_DIR = ML_SERVICES_ROOT / "outputs" / "apptek" / "simple_sentiment_results"
DEFAULT_OUTPUT_DIR = ML_SERVICES_ROOT / "outputs" / "backend" / "sentiment_calls"
DEFAULT_MANIFEST_PATH = ML_SERVICES_ROOT / "outputs" / "backend" / "sentiment_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def infer_domain_from_path(path):
    parent = path.parent.name

    if parent in {"banking", "health", "healthcare", "telecom"}:
        return parent

    call_id = path.stem.replace("_simple_sentiment", "").lower()

    if "banking" in call_id:
        return "banking"
    if "health" in call_id:
        return "health"
    if "telecom" in call_id:
        return "telecom"

    return "unknown"


def get_risk_level(max_escalation_score):
    if max_escalation_score is None:
        return "Unknown"
    if max_escalation_score >= 0.50:
        return "High"
    if max_escalation_score >= 0.25:
        return "Medium"
    return "Low"


def build_backend_payload(simple_payload, source_file, domain):
    call_id = simple_payload["call_id"]
    model_version = simple_payload.get("model_version")
    segments = simple_payload.get("segments", [])

    status_counts = Counter(seg.get("processing_status") for seg in segments)

    successful_segments = [
        seg for seg in segments
        if seg.get("processing_status") == "success"
    ]

    sentiment_counts = Counter(
        seg.get("sentiment")
        for seg in successful_segments
        if seg.get("sentiment") is not None
    )

    emotion_counts = Counter(
        seg.get("dominant_emotion")
        for seg in successful_segments
        if seg.get("dominant_emotion") is not None
    )

    escalation_scores = [
        float(seg.get("escalation_score"))
        for seg in successful_segments
        if seg.get("escalation_score") is not None
    ]

    avg_escalation_score = round(sum(escalation_scores) / len(escalation_scores), 6) if escalation_scores else None
    max_escalation_score = round(max(escalation_scores), 6) if escalation_scores else None

    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else None
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
    risk_level = get_risk_level(max_escalation_score)

    seq_id_counts = Counter(seg.get("seq_id") for seg in segments)
    duplicate_seq_ids = sorted([
        seq_id for seq_id, count in seq_id_counts.items()
        if seq_id is not None and count > 1
    ])

    backend_segments = []

    for segment_index, seg in enumerate(segments, start=1):
        backend_segments.append({
            "segment_index": segment_index,
            "segment_key": f"{call_id}_{segment_index:04d}",
            "seq_id": seg.get("seq_id"),
            "sentiment": seg.get("sentiment"),
            "dominant_emotion": seg.get("dominant_emotion"),
            "escalation_score": seg.get("escalation_score"),
            "processing_status": seg.get("processing_status"),

            # Confidence fields used for calibration.
            "confidence_level": seg.get("confidence_level"),
            "prediction_confidence": seg.get("prediction_confidence"),
            "emotion_confidence": seg.get("emotion_confidence"),
            "sentiment_confidence": seg.get("sentiment_confidence"),
            "negative_emotion_probability": seg.get("negative_emotion_probability"),
            
        })

    return {
        "call_id": call_id,
        "domain": domain,
        "model_version": model_version,
        "source_file": str(source_file.relative_to(ML_SERVICES_ROOT)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "join_keys": {
            "recommended_unique_key": "call_id + segment_index",
            "transcript_alignment_key": "call_id + seq_id",
            "note": "seq_id is preserved from transcription. segment_index is added because one transcript file contains a duplicate seq_id."
        },
        "call_summary": {
            "total_segments": len(segments),
            "successful_segments": len(successful_segments),
            "skipped_segments": status_counts.get("skipped_too_short", 0),
            "dominant_sentiment": dominant_sentiment,
            "dominant_emotion": dominant_emotion,
            "average_escalation_score": avg_escalation_score,
            "max_escalation_score": max_escalation_score,
            "risk_level": risk_level,
            "processing_status_distribution": dict(status_counts),
            "sentiment_distribution": dict(sentiment_counts),
            "emotion_distribution": dict(emotion_counts),
            "duplicate_seq_ids": duplicate_seq_ids,
        },
        "segments": backend_segments,
    }


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.rglob("*_simple_sentiment.json"))

    input_files = [
        path for path in input_files
        if path.parent.name in {"banking", "health", "healthcare", "telecom"}
    ]

    manifest_calls = []
    total_segments = 0
    successful_segments = 0
    failed_calls = 0

    for input_file in input_files:
        simple_payload = load_json(input_file)
        domain = infer_domain_from_path(input_file)

        try:
            backend_payload = build_backend_payload(simple_payload, input_file, domain)

            call_id = backend_payload["call_id"]
            output_path = output_dir / domain / f"{call_id}_backend_sentiment.json"
            save_json(output_path, backend_payload)

            total_segments += backend_payload["call_summary"]["total_segments"]
            successful_segments += backend_payload["call_summary"]["successful_segments"]

            manifest_calls.append({
                "call_id": call_id,
                "domain": domain,
                "status": "success",
                "total_segments": backend_payload["call_summary"]["total_segments"],
                "successful_segments": backend_payload["call_summary"]["successful_segments"],
                "risk_level": backend_payload["call_summary"]["risk_level"],
                "dominant_sentiment": backend_payload["call_summary"]["dominant_sentiment"],
                "dominant_emotion": backend_payload["call_summary"]["dominant_emotion"],
                "duplicate_seq_ids": backend_payload["call_summary"]["duplicate_seq_ids"],
                "backend_payload_path": str(output_path.relative_to(ML_SERVICES_ROOT)),
            })

        except Exception as error:
            failed_calls += 1
            manifest_calls.append({
                "call_id": simple_payload.get("call_id"),
                "domain": domain,
                "status": f"failed: {error}",
                "total_segments": None,
                "successful_segments": None,
                "risk_level": None,
                "dominant_sentiment": None,
                "dominant_emotion": None,
                "duplicate_seq_ids": [],
                "backend_payload_path": None,
            })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "AppTek timestamped sentiment batch",
        "model": "Model V5 audio emotion and sentiment inference",
        "total_calls": len(manifest_calls),
        "successful_calls": sum(1 for row in manifest_calls if row["status"] == "success"),
        "failed_calls": failed_calls,
        "total_segments": total_segments,
        "successful_segments": successful_segments,
        "note": "Backend-ready sentiment payloads. Use call_id + segment_index as the safest unique key. seq_id is still included for transcript alignment.",
        "calls": manifest_calls,
    }

    save_json(manifest_path, manifest)

    print("Backend sentiment export completed")
    print("-" * 80)
    print(f"Input files: {len(input_files)}")
    print(f"Successful calls: {manifest['successful_calls']}")
    print(f"Failed calls: {manifest['failed_calls']}")
    print(f"Total segments: {total_segments}")
    print(f"Successful segments: {successful_segments}")
    print(f"Output folder: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
