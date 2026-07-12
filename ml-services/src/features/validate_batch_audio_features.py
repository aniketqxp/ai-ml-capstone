"""
Validate batch sentiment files with audio features.
"""

import json
from pathlib import Path
from collections import Counter


def main():
    root = Path("outputs/backend/sentiment_calls_with_features")
    files = sorted(root.rglob("*_backend_sentiment_with_features.json"))

    print(f"Files found: {len(files)}")

    status_counts = Counter()
    domain_counts = Counter()

    failed = []

    for path in files:
        data = json.load(open(path, "r", encoding="utf-8"))

        call_id = data.get("call_id")
        domain = data.get("domain")
        domain_counts[domain] += 1

        summary = data.get("audio_feature_match_summary", {})
        matched = summary.get("matched_segments", 0)
        total = summary.get("total_sentiment_segments", 0)

        has_features = data.get("has_audio_features")
        series = data.get("dashboard_audio_feature_series", [])
        segments = data.get("segments", [])
        segments_with_explanations = [
            segment for segment in segments
            if segment.get("escalation_explanation") is not None
            and segment.get("explainability_flags") is not None
        ]

        audio_summary = data.get("audio_feature_summary", {})
        speaker_summary = data.get("speaker_audio_feature_summary", {})
        customer_summary = speaker_summary.get("customer", {})
        agent_summary = speaker_summary.get("agent", {})
        customer_trend = data.get("customer_escalation_trend", {})

        if (
            has_features
            and matched == total
            and len(series) == total
            and audio_summary.get("total_segments_with_audio_features") == total
            and len(segments_with_explanations) == total
            and customer_summary.get("total_segments") is not None
            and agent_summary.get("total_segments") is not None
            and customer_trend.get("trend") is not None
        ):
            status_counts["valid"] += 1
        else:
            status_counts["invalid"] += 1
            failed.append({
                "call_id": call_id,
                "path": str(path),
                "matched": matched,
                "total": total,
                "series_len": len(series),
                "has_audio_features": has_features,
            })

    print("Domain counts:", dict(domain_counts))
    print("Status counts:", dict(status_counts))

    if failed:
        print("\nFailed files:")
        for item in failed:
            print(item)
    else:
        print("\nAll files are valid.")


if __name__ == "__main__":
    main()