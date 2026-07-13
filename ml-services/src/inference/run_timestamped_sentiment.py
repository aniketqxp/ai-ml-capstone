"""
Run V5 audio sentiment analysis using timestamped speaker segments.

Supports:
1. Single audio file:
   "audio_path": "path/to/audio.wav"

2. Speaker-based stereo channel files:
   "audio_paths": {
       "agent": "path/to/agent_channel.wav",
       "customer": "path/to/customer_channel.wav"
   }
"""

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from src.inference.emotion_predictor import EmotionPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

OUTPUT_DIR = ML_SERVICES_ROOT / "outputs" / "apptek" / "timestamped_sentiment_results"
SUMMARY_PATH = ML_SERVICES_ROOT / "outputs" / "apptek" / "apptek_timestamped_sentiment_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-path", required=True)
    parser.add_argument("--min-segment-seconds", type=float, default=1.0)
    parser.add_argument("--max-segment-seconds", type=float, default=30.0)
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ML_SERVICES_ROOT / path
    return path


def result_to_dict(result):
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return result


def normalize_value(value):
    if hasattr(value, "value"):
        return value.value
    return value


def clean_result_dict(result_dict):
    cleaned = {}
    for key, value in result_dict.items():
        if key == "sentiment_timeline":
            continue

        if isinstance(value, dict):
            cleaned[key] = {
                sub_key: normalize_value(sub_value)
                for sub_key, sub_value in value.items()
            }
        else:
            cleaned[key] = normalize_value(value)

    return cleaned


def load_audio_segment(audio_path, start_time, end_time):
    audio_info = sf.info(str(audio_path))
    sample_rate = audio_info.samplerate

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    audio_data, sr = sf.read(
        str(audio_path),
        start=start_sample,
        stop=end_sample,
        dtype="float32",
    )

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    return audio_data, sr


def write_temp_wav(audio_data, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    sf.write(temp_path, audio_data, sample_rate)
    return temp_path


def split_long_segment(segment, max_segment_seconds):
    start = float(segment["start_time_seconds"])
    end = float(segment["end_time_seconds"])

    if end - start <= max_segment_seconds:
        return [segment]

    chunks = []
    chunk_id = 0
    current = start

    while current < end:
        chunk_end = min(current + max_segment_seconds, end)
        new_segment = dict(segment)
        new_segment["segment_id"] = f"{segment['segment_id']}_{chunk_id}"
        new_segment["start_time_seconds"] = round(current, 3)
        new_segment["end_time_seconds"] = round(chunk_end, 3)
        new_segment["parent_segment_id"] = segment["segment_id"]
        chunks.append(new_segment)
        current = chunk_end
        chunk_id += 1

    return chunks


def get_audio_path_for_segment(payload, speaker):
    speaker = str(speaker).lower()

    if "audio_paths" in payload:
        audio_paths = payload["audio_paths"]

        if speaker in audio_paths:
            return resolve_path(audio_paths[speaker])

        if "unknown" in audio_paths:
            return resolve_path(audio_paths["unknown"])

        raise ValueError(f"No audio path found for speaker={speaker}")

    if "audio_path" in payload:
        return resolve_path(payload["audio_path"])

    raise ValueError("No audio_path or audio_paths found in segment payload.")


def calculate_call_summary(call_id, payload, segment_results):
    valid_results = [
        item for item in segment_results
        if item.get("processing_status") == "success"
    ]

    if not valid_results:
        return {
            "call_id": call_id,
            "source_apptek_id": payload.get("source_apptek_id"),
            "analysis_mode": "timestamped_segments",
            "overall_audio_sentiment": "Unknown",
            "dominant_emotion": "unknown",
            "audio_escalation_score": 0.0,
            "max_escalation_score": 0.0,
            "risk_level": "Unknown",
            "confidence_level": "Low",
            "total_segments": len(segment_results),
            "successful_segments": 0,
        }

    sentiments = [item["overall_audio_sentiment"] for item in valid_results]
    emotions = [item["dominant_emotion"] for item in valid_results]
    risks = [item["risk_level"] for item in valid_results]
    escalation_scores = [float(item["audio_escalation_score"]) for item in valid_results]

    sentiment_counts = Counter(sentiments)
    emotion_counts = Counter(emotions)
    risk_counts = Counter(risks)

    avg_escalation = float(np.mean(escalation_scores))
    max_escalation = float(np.max(escalation_scores))

    peak_segment = max(valid_results, key=lambda item: float(item["audio_escalation_score"]))

    if max_escalation >= 0.70:
        final_risk = "High"
    elif avg_escalation >= 0.30 or max_escalation >= 0.45:
        final_risk = "Medium"
    else:
        final_risk = "Low"

    model_version = valid_results[0].get("model_version")

    return {
        "call_id": call_id,
        "source_apptek_id": payload.get("source_apptek_id"),
        "analysis_mode": "timestamped_segments",
        "model_version": model_version,
        "overall_audio_sentiment": sentiment_counts.most_common(1)[0][0],
        "dominant_emotion": emotion_counts.most_common(1)[0][0],
        "audio_escalation_score": round(avg_escalation, 6),
        "max_escalation_score": round(max_escalation, 6),
        "risk_level": final_risk,
        "confidence_level": Counter(
            [item["confidence_level"] for item in valid_results]
        ).most_common(1)[0][0],
        "total_segments": len(segment_results),
        "successful_segments": len(valid_results),
        "sentiment_distribution": dict(sentiment_counts),
        "emotion_distribution": dict(emotion_counts),
        "risk_distribution": dict(risk_counts),
        "peak_segment": {
            "segment_id": peak_segment["segment_id"],
            "speaker": peak_segment.get("speaker", "unknown"),
            "start_time_seconds": peak_segment["start_time_seconds"],
            "end_time_seconds": peak_segment["end_time_seconds"],
            "dominant_emotion": peak_segment["dominant_emotion"],
            "overall_audio_sentiment": peak_segment["overall_audio_sentiment"],
            "audio_escalation_score": peak_segment["audio_escalation_score"],
            "risk_level": peak_segment["risk_level"],
        },
    }


def append_summary(call_summary):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "call_id": call_summary.get("call_id"),
        "source_apptek_id": call_summary.get("source_apptek_id"),
        "analysis_mode": call_summary.get("analysis_mode"),
        "model_version": call_summary.get("model_version"),
        "overall_audio_sentiment": call_summary.get("overall_audio_sentiment"),
        "dominant_emotion": call_summary.get("dominant_emotion"),
        "audio_escalation_score": call_summary.get("audio_escalation_score"),
        "max_escalation_score": call_summary.get("max_escalation_score"),
        "risk_level": call_summary.get("risk_level"),
        "confidence_level": call_summary.get("confidence_level"),
        "total_segments": call_summary.get("total_segments"),
        "successful_segments": call_summary.get("successful_segments"),
        "peak_segment_id": call_summary.get("peak_segment", {}).get("segment_id"),
        "peak_segment_speaker": call_summary.get("peak_segment", {}).get("speaker"),
        "peak_start_time_seconds": call_summary.get("peak_segment", {}).get("start_time_seconds"),
        "peak_end_time_seconds": call_summary.get("peak_segment", {}).get("end_time_seconds"),
        "peak_emotion": call_summary.get("peak_segment", {}).get("dominant_emotion"),
        "peak_sentiment": call_summary.get("peak_segment", {}).get("overall_audio_sentiment"),
        "peak_escalation_score": call_summary.get("peak_segment", {}).get("audio_escalation_score"),
    }

    write_header = not SUMMARY_PATH.exists()

    with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    segments_path = resolve_path(args.segments_path)

    with segments_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    call_id = payload["call_id"]

    raw_segments = payload["segments"]

    expanded_segments = []
    for segment in raw_segments:
        expanded_segments.extend(
            split_long_segment(segment, args.max_segment_seconds)
        )

    predictor = EmotionPredictor()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segment_results = []

    print("\nRunning timestamped audio sentiment")
    print("-" * 80)
    print(f"Call ID: {call_id}")
    print(f"Source AppTek ID: {payload.get('source_apptek_id')}")
    print(f"Raw segments: {len(raw_segments)}")
    print(f"Expanded segments: {len(expanded_segments)}")
    print("-" * 80)

    for segment in expanded_segments:
        segment_id = segment["segment_id"]
        speaker = str(segment.get("speaker", "unknown")).lower()
        start_time = float(segment["start_time_seconds"])
        end_time = float(segment["end_time_seconds"])
        duration = end_time - start_time

        if duration < args.min_segment_seconds:
            segment_results.append(
                {
                    "segment_id": segment_id,
                    "seq_id": segment.get("seq_id"),
                    "speaker": speaker,
                    "start_time_seconds": start_time,
                    "end_time_seconds": end_time,
                    "duration_seconds": round(duration, 3),
                    "processing_status": "skipped_too_short",
                }
            )
            continue

        audio_path = get_audio_path_for_segment(payload, speaker)

        print(
            f"Processing {segment_id} | speaker={speaker} | "
            f"{start_time:.2f}s-{end_time:.2f}s | audio={audio_path.name}"
        )

        temp_path = None

        try:
            audio_data, sample_rate = load_audio_segment(audio_path, start_time, end_time)
            temp_path = write_temp_wav(audio_data, sample_rate)

            result = predictor.analyze_audio(str(temp_path))
            result_dict = clean_result_dict(result_to_dict(result))

            segment_results.append(
                {
                    "segment_id": segment_id,
                    "seq_id": segment.get("seq_id"),
                    "speaker": speaker,
                    "audio_path": str(audio_path.relative_to(ML_SERVICES_ROOT)),
                    "start_time_seconds": start_time,
                    "end_time_seconds": end_time,
                    "duration_seconds": round(duration, 3),
                    "processing_status": "success",
                    "dominant_emotion": result_dict.get("dominant_emotion"),
                    "overall_audio_sentiment": result_dict.get("overall_audio_sentiment"),
                    "audio_escalation_score": result_dict.get("audio_escalation_score"),
                    "risk_level": result_dict.get("risk_level"),
                    "confidence_level": result_dict.get("confidence_level"),

                    # Raw model confidence from EmotionPredictor.
                    "prediction_confidence": result_dict.get("prediction_confidence"),
                    "emotion_confidence": result_dict.get("prediction_confidence"),
                    "negative_emotion_probability": result_dict.get("negative_emotion_probability"),

                    # Approximate sentiment confidence.
                    # For negative sentiment, use negative_emotion_probability.
                    # For positive/neutral sentiment, use prediction_confidence.
                    "sentiment_confidence": (
                        result_dict.get("negative_emotion_probability")
                        if str(result_dict.get("overall_audio_sentiment")).lower() == "negative"
                        else result_dict.get("prediction_confidence")
                    ),

                    "model_version": result_dict.get("model_version"),
                    "parent_segment_id": segment.get("parent_segment_id"),
                    "text": segment.get("text"),
                }
            )

        except Exception as exc:
            segment_results.append(
                {
                    "segment_id": segment_id,
                    "seq_id": segment.get("seq_id"),
                    "speaker": speaker,
                    "start_time_seconds": start_time,
                    "end_time_seconds": end_time,
                    "duration_seconds": round(duration, 3),
                    "processing_status": "failed",
                    "error": str(exc),
                }
            )

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    call_summary = calculate_call_summary(call_id, payload, segment_results)

    output_payload = {
        **call_summary,
        "audio_paths": payload.get("audio_paths"),
        "channel_mapping": payload.get("channel_mapping"),
        "segments": segment_results,
    }

    output_json_path = OUTPUT_DIR / f"{call_id}_timestamped_sentiment.json"

    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, indent=2)

    append_summary(call_summary)

    print("\nTimestamped sentiment completed.")
    print("-" * 80)
    print(f"Saved JSON: {output_json_path}")
    print(f"Updated summary CSV: {SUMMARY_PATH}")
    print("-" * 80)
    print(json.dumps(call_summary, indent=2))


if __name__ == "__main__":
    main()
