import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--include-skipped", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as file:
        detailed = json.load(file)

    simple_segments = []

    for segment in detailed.get("segments", []):
        status = segment.get("processing_status")

        if status != "success" and not args.include_skipped:
            continue

        if status == "success":
            sentiment = segment.get("overall_audio_sentiment")
            dominant_emotion = segment.get("dominant_emotion")
            escalation_score = round(float(segment.get("audio_escalation_score", 0.0)), 4)
        else:
            sentiment = None
            dominant_emotion = None
            escalation_score = None

        simple_segments.append(
            {
                "seq_id": segment.get("seq_id"),
                "sentiment": sentiment,
                "dominant_emotion": dominant_emotion,
                "escalation_score": escalation_score,
                "processing_status": status,
                "confidence_level": segment.get("confidence_level"),
                "prediction_confidence": segment.get("prediction_confidence"),
                "emotion_confidence": segment.get("emotion_confidence"),
                "sentiment_confidence": segment.get("sentiment_confidence"),
                "negative_emotion_probability": segment.get("negative_emotion_probability"),
            }
        )

    model_version = detailed.get("model_version")

    if not model_version:
        for segment in detailed.get("segments", []):
            if segment.get("model_version"):
                model_version = segment.get("model_version")
                break

    output_payload = {
        "call_id": args.call_id,
        "model_version": model_version,
        "segments": simple_segments,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, indent=2)

    print("Saved simple sentiment schema:")
    print(output_path)
    print("Returned segments:", len(simple_segments))


if __name__ == "__main__":
    main()
