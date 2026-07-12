"""
Merge backend sentiment payloads with extracted audio features.

This creates a dashboard-ready sentiment payload where each transcript segment
contains sentiment results plus raw audio features such as pitch, volume,
energy, pauses, and speech rate.
"""

import argparse
import json
from pathlib import Path

def normalize_label(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def build_segment_explainability(segment):
    """
    Build human-readable reasons for why a segment may be risky or important.

    This does not replace the model prediction.
    It explains the prediction using sentiment, emotion, escalation score,
    and extracted audio features.
    """
    sentiment = normalize_label(segment.get("sentiment"))
    emotion = normalize_label(segment.get("dominant_emotion"))
    escalation_score = segment.get("escalation_score")

    audio = segment.get("audio_features") or {}

    pitch_level = normalize_label(audio.get("pitch_level"))
    volume_level = normalize_label(audio.get("volume_level"))
    pause_level = normalize_label(audio.get("pause_level"))
    speech_rate_level = normalize_label(audio.get("speech_rate_level"))

    flags = {
        "negative_sentiment": sentiment == "negative",
        "anger_emotion": emotion in {"anger", "angry"},
        "sadness_or_fear_emotion": emotion in {"sadness", "sad", "fear", "fearful"},
        "high_escalation_score": escalation_score is not None and escalation_score >= 0.6,
        "medium_escalation_score": escalation_score is not None and 0.4 <= escalation_score < 0.6,
        "high_pitch": pitch_level == "high",
        "high_volume": volume_level == "high",
        "fast_speech": speech_rate_level == "fast",
        "high_pause": pause_level == "high",
    }

    reasons = []

    if flags["negative_sentiment"]:
        reasons.append("Negative sentiment detected")

    if flags["anger_emotion"]:
        reasons.append("Anger-related emotion detected")

    if flags["sadness_or_fear_emotion"]:
        reasons.append("Emotion may indicate customer distress")

    if flags["high_escalation_score"]:
        reasons.append("High escalation score")

    elif flags["medium_escalation_score"]:
        reasons.append("Medium escalation score")

    if flags["high_pitch"]:
        reasons.append("High pitch detected")

    if flags["high_volume"]:
        reasons.append("High volume detected")

    if flags["fast_speech"]:
        reasons.append("Fast speech rate detected")

    if flags["high_pause"]:
        reasons.append("Long or frequent pauses detected")

    if not reasons:
        reasons.append("No strong escalation indicators detected")

    return {
        "explainability_flags": flags,
        "escalation_explanation": reasons,
    }

def safe_average(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def count_level(segments, feature_name, expected_level):
    count = 0

    for segment in segments:
        audio = segment.get("audio_features") or {}
        if audio.get(feature_name) == expected_level:
            count += 1

    return count


def find_max_segment(segments, feature_name):
    best_segment = None
    best_value = None

    for segment in segments:
        audio = segment.get("audio_features") or {}
        value = audio.get(feature_name)

        if value is None:
            continue

        if best_value is None or value > best_value:
            best_value = value
            best_segment = segment

    if best_segment is None:
        return {
            "segment_index": None,
            "seq_id": None,
            "speaker": None,
            "value": None,
        }

    return {
        "segment_index": best_segment.get("segment_index"),
        "seq_id": best_segment.get("seq_id"),
        "speaker": best_segment.get("speaker"),
        "value": round(best_value, 4),
    }


def build_audio_feature_summary(sentiment_payload):
    segments = sentiment_payload.get("segments", [])

    segments_with_audio = [
        segment for segment in segments
        if segment.get("audio_features") is not None
    ]

    pitch_values = []
    volume_values = []
    energy_values = []
    pause_values = []
    speech_rate_values = []

    for segment in segments_with_audio:
        audio = segment.get("audio_features") or {}

        pitch_values.append(audio.get("pitch_mean_hz"))
        volume_values.append(audio.get("volume_db_mean"))
        energy_values.append(audio.get("rms_energy_mean"))
        pause_values.append(audio.get("pause_ratio"))
        speech_rate_values.append(audio.get("speech_rate_words_per_minute"))

    return {
        "total_segments": len(segments),
        "total_segments_with_audio_features": len(segments_with_audio),

        "average_pitch_hz": safe_average(pitch_values),
        "average_volume_db": safe_average(volume_values),
        "average_energy": safe_average(energy_values),
        "average_pause_ratio": safe_average(pause_values),
        "average_speech_rate_wpm": safe_average(speech_rate_values),

        "high_pitch_segments": count_level(segments_with_audio, "pitch_level", "high"),
        "high_volume_segments": count_level(segments_with_audio, "volume_level", "high"),
        "fast_speech_segments": count_level(segments_with_audio, "speech_rate_level", "fast"),
        "high_pause_segments": count_level(segments_with_audio, "pause_level", "high"),

        "highest_pitch_segment": find_max_segment(segments_with_audio, "pitch_mean_hz"),
        "highest_volume_segment": find_max_segment(segments_with_audio, "volume_db_mean"),
        "highest_energy_segment": find_max_segment(segments_with_audio, "rms_energy_mean"),
        "highest_pause_ratio_segment": find_max_segment(segments_with_audio, "pause_ratio"),
        "highest_speech_rate_segment": find_max_segment(segments_with_audio, "speech_rate_words_per_minute"),
    }

def build_single_speaker_summary(segments, speaker_name):
    speaker_segments = [
        segment for segment in segments
        if normalize_label(segment.get("speaker")) == normalize_label(speaker_name)
    ]

    segments_with_audio = [
        segment for segment in speaker_segments
        if segment.get("audio_features") is not None
    ]

    pitch_values = []
    volume_values = []
    energy_values = []
    pause_values = []
    speech_rate_values = []
    escalation_values = []
    sentiments = []
    emotions = []

    for segment in speaker_segments:
        escalation = segment.get("escalation_score")
        if escalation is not None:
            escalation_values.append(escalation)

        sentiments.append(segment.get("sentiment"))
        emotions.append(segment.get("dominant_emotion"))

    for segment in segments_with_audio:
        audio = segment.get("audio_features") or {}

        pitch_values.append(audio.get("pitch_mean_hz"))
        volume_values.append(audio.get("volume_db_mean"))
        energy_values.append(audio.get("rms_energy_mean"))
        pause_values.append(audio.get("pause_ratio"))
        speech_rate_values.append(audio.get("speech_rate_words_per_minute"))

    return {
        "speaker": speaker_name,
        "total_segments": len(speaker_segments),
        "total_segments_with_audio_features": len(segments_with_audio),

        "average_escalation_score": safe_average(escalation_values),
        "dominant_sentiment": dominant_label(sentiments),
        "dominant_emotion": dominant_label(emotions),

        "average_pitch_hz": safe_average(pitch_values),
        "average_volume_db": safe_average(volume_values),
        "average_energy": safe_average(energy_values),
        "average_pause_ratio": safe_average(pause_values),
        "average_speech_rate_wpm": safe_average(speech_rate_values),

        "high_pitch_segments": count_level(segments_with_audio, "pitch_level", "high"),
        "high_volume_segments": count_level(segments_with_audio, "volume_level", "high"),
        "fast_speech_segments": count_level(segments_with_audio, "speech_rate_level", "fast"),
        "high_pause_segments": count_level(segments_with_audio, "pause_level", "high"),

        "highest_pitch_segment": find_max_segment(segments_with_audio, "pitch_mean_hz"),
        "highest_volume_segment": find_max_segment(segments_with_audio, "volume_db_mean"),
        "highest_energy_segment": find_max_segment(segments_with_audio, "rms_energy_mean"),
        "highest_pause_ratio_segment": find_max_segment(segments_with_audio, "pause_ratio"),
        "highest_speech_rate_segment": find_max_segment(segments_with_audio, "speech_rate_words_per_minute"),
    }


def build_speaker_audio_feature_summary(sentiment_payload):
    segments = sentiment_payload.get("segments", [])

    return {
        "customer": build_single_speaker_summary(segments, "CUSTOMER"),
        "agent": build_single_speaker_summary(segments, "AGENT"),
    }

def build_customer_escalation_trend(sentiment_payload):
    """
    Analyze whether customer escalation increases, decreases, or stays stable
    across the call.

    This uses only CUSTOMER segments and compares the first half of the call
    with the second half.
    """
    segments = sentiment_payload.get("segments", [])

    customer_segments = [
        segment for segment in segments
        if normalize_label(segment.get("speaker")) == "customer"
        and segment.get("escalation_score") is not None
    ]

    if len(customer_segments) < 4:
        return {
            "customer_segments_analyzed": len(customer_segments),
            "first_half_average_escalation": None,
            "second_half_average_escalation": None,
            "trend_delta": None,
            "trend": "not_enough_customer_segments",
            "trend_explanation": "Not enough customer segments with escalation scores to calculate a reliable trend.",
        }

    midpoint = len(customer_segments) // 2

    first_half = customer_segments[:midpoint]
    second_half = customer_segments[midpoint:]

    first_half_scores = [
        segment.get("escalation_score")
        for segment in first_half
        if segment.get("escalation_score") is not None
    ]

    second_half_scores = [
        segment.get("escalation_score")
        for segment in second_half
        if segment.get("escalation_score") is not None
    ]

    first_avg = safe_average(first_half_scores)
    second_avg = safe_average(second_half_scores)

    if first_avg is None or second_avg is None:
        return {
            "customer_segments_analyzed": len(customer_segments),
            "first_half_average_escalation": first_avg,
            "second_half_average_escalation": second_avg,
            "trend_delta": None,
            "trend": "not_enough_customer_segments",
            "trend_explanation": "Not enough valid escalation scores to calculate customer escalation trend.",
        }

    trend_delta = round(second_avg - first_avg, 4)

    if trend_delta >= 0.10:
        trend = "increasing"
        explanation = "Customer escalation increased in the second half of the call."
    elif trend_delta <= -0.10:
        trend = "decreasing"
        explanation = "Customer escalation decreased in the second half of the call."
    else:
        trend = "stable"
        explanation = "Customer escalation stayed relatively stable across the call."

    return {
        "customer_segments_analyzed": len(customer_segments),
        "first_half_average_escalation": first_avg,
        "second_half_average_escalation": second_avg,
        "trend_delta": trend_delta,
        "trend": trend,
        "trend_explanation": explanation,
    }


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
            "explainability_flags": segment.get("explainability_flags"),
            "escalation_explanation": segment.get("escalation_explanation"),
        })

    return series


def dominant_label(values):
    values = [v for v in values if v is not None]

    if not values:
        return None

    counts = {}

    for value in values:
        label = str(value).strip()
        counts[label] = counts.get(label, 0) + 1

    return max(counts, key=counts.get)


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

        feature_item = feature_map_by_segment_index.get(segment_index)

        if feature_item is None:
            feature_item = feature_map_by_seq_id.get(seq_id)

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

        explanation = build_segment_explainability(segment)
        segment["explainability_flags"] = explanation["explainability_flags"]
        segment["escalation_explanation"] = explanation["escalation_explanation"]
        

    sentiment["audio_feature_version"] = features.get("feature_extraction_version")
    sentiment["has_audio_features"] = matched_count > 0
    sentiment["audio_feature_match_summary"] = {
        "matched_segments": matched_count,
        "total_sentiment_segments": len(sentiment.get("segments", [])),
    }

    sentiment["dashboard_audio_feature_series"] = build_audio_feature_series(sentiment)
    sentiment["audio_feature_summary"] = build_audio_feature_summary(sentiment)
    sentiment["speaker_audio_feature_summary"] = build_speaker_audio_feature_summary(sentiment)
    sentiment["customer_escalation_trend"] = build_customer_escalation_trend(sentiment)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment, f, indent=2)

    print(f"Saved merged sentiment + audio features to: {output_path}")
    print(f"Matched segments: {matched_count}/{len(sentiment.get('segments', []))}")
    print("Dashboard audio feature series added.")


if __name__ == "__main__":
    main()