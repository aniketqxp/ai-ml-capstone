"""
Merge backend sentiment payloads with extracted audio features.

This creates a dashboard-ready sentiment payload where each transcript segment
contains sentiment results plus raw audio features such as pitch, volume,
energy, pauses, and speech rate.
"""

import argparse
import json
from pathlib import Path


def build_audio_feature_series(sentiment_payload):
    """
    Build a clean call-level feature series for dashboard line graphs.

    The dashboard can use this directly to plot:
    - pitch over call
    - volume over call
    - energy over call
    - speech rate over call
    - pause ratio over call
    - escalation over call
    """
    series = []

    for segment in sentiment_payload.get("segments", []):
        audio = segment.get("audio_features") or {}

        series.append({
            "segment_index": segment.get("segment_index"),
            "seq_id": segment.get("seq_id"),
            "speaker": segment.get("speaker"),
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "text": segment.get("text"),

            "sentiment": segment.get("sentiment"),
            "dominant_emotion": segment.get("dominant_emotion"),
            "escalation_score": segment.get("escalation_score"),

            "pitch_mean_hz": audio.get("pitch_mean_hz"),
            "pitch_level": audio.get("pitch_level"),

            "volume_db_mean": audio.get("volume_db_mean"),
            "volume_level": audio.get("volume_level"),

            "rms_energy_mean": audio.get("rms_energy_mean"),
            "rms_energy_max": audio.get("rms_energy_max"),

            "pause_ratio": audio.get("pause_ratio"),
            "pause_level": audio.get("pause_level"),
            "pause_count": audio.get("pause_count"),

            "speech_rate_words_per_minute": audio.get("speech_rate_words_per_minute"),
            "speech_rate_level": audio.get("speech_rate_level"),
        })

    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    sentiment_path = Path(args.sentiment_path)
    features_path = Path(args.features_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentiment = json.load(open(sentiment_path, "r", encoding="utf-8"))
    features = json.load(open(features_path, "r", encoding="utf-8"))

    feature_map_by_seq_id = {
        item.get("seq_id"): item
        for item in features.get("segments", [])
        if item.get("seq_id") is not None
    }

    feature_map_by_segment_index = {
        item.get("segment_index"): item
        for item in features.get("segments", [])
        if item.get("segment_index") is not None
    }

    matched_count = 0

    for segment in sentiment.get("segments", []):
        seq_id = segment.get("seq_id")
        segment_index = segment.get("segment_index")

        feature_item = feature_map_by_seq_id.get(seq_id)

        if feature_item is None:
            feature_item = feature_map_by_segment_index.get(segment_index)

        if feature_item is None:
            segment["audio_features"] = None
            continue

        segment["speaker"] = feature_item.get("speaker")
        segment["start_time"] = feature_item.get("start_time")
        segment["end_time"] = feature_item.get("end_time")
        segment["text"] = feature_item.get("text")
        segment["audio_features"] = feature_item.get("audio_features")

        if segment["audio_features"] is not None:
            matched_count += 1

    sentiment["audio_feature_version"] = features.get("feature_extraction_version")
    sentiment["has_audio_features"] = matched_count > 0
    sentiment["audio_feature_match_summary"] = {
        "matched_segments": matched_count,
        "total_sentiment_segments": len(sentiment.get("segments", [])),
    }

    sentiment["dashboard_audio_feature_series"] = build_audio_feature_series(sentiment)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment, f, indent=2)

    print(f"Saved merged sentiment + audio features to: {output_path}")
    print(f"Matched segments: {matched_count}/{len(sentiment.get('segments', []))}")
    print("Dashboard audio feature series added.")


if __name__ == "__main__":
    main()