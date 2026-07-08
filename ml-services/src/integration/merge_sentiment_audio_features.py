"""
Merge backend sentiment payloads with extracted audio features.
"""

import argparse
import json
from pathlib import Path


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
        item.get("seq_id"): item.get("audio_features")
        for item in features.get("segments", [])
        if item.get("seq_id") is not None
    }

    feature_map_by_segment_index = {
        item.get("segment_index"): item.get("audio_features")
        for item in features.get("segments", [])
        if item.get("segment_index") is not None
    }

    matched_count = 0

    for segment in sentiment.get("segments", []):
        seq_id = segment.get("seq_id")
        segment_index = segment.get("segment_index")

        audio_features = feature_map_by_seq_id.get(seq_id)

        if audio_features is None:
            audio_features = feature_map_by_segment_index.get(segment_index)

        segment["audio_features"] = audio_features

        if audio_features is not None:
            matched_count += 1

    sentiment["audio_feature_version"] = features.get("feature_extraction_version")
    sentiment["has_audio_features"] = matched_count > 0
    sentiment["audio_feature_match_summary"] = {
        "matched_segments": matched_count,
        "total_sentiment_segments": len(sentiment.get("segments", [])),
    }

    

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment, f, indent=2)

    print(f"Saved merged sentiment + audio features to: {output_path}")


if __name__ == "__main__":
    main()