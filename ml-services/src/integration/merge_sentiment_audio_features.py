"""
Merge backend sentiment payloads with extracted audio features.

This creates a dashboard-ready sentiment payload where each transcript segment
contains sentiment results plus raw audio features such as pitch, volume,
energy, pauses, and speech rate.
"""

import argparse
import json
from pathlib import Path





def build_calibrated_call_sentiment(payload: dict) -> dict:
    """
    Build a business-level customer sentiment summary using only model outputs.

    This avoids transcript keyword rules and relies on:
    - raw sentiment labels
    - raw emotion labels
    - escalation scores
    - repeated negative signals
    - customer-only segments
    """
    segments = payload.get("segments", []) or []

    customer_segments = [
        segment for segment in segments
        if normalize_label(segment.get("speaker")) == "customer"
    ]

    if not customer_segments:
        return {
            "customer_sentiment_calibrated": None,
            "calibration_note": "No customer segments available for calibration.",
            "negative_customer_segments": 0,
            "positive_customer_segments": 0,
            "strong_negative_customer_segments": 0,
        }

    negative_count = 0
    positive_count = 0
    neutral_count = 0
    strong_negative_count = 0
    high_escalation_count = 0
    anger_fear_count = 0

    for segment in customer_segments:
        sentiment = normalize_label(segment.get("sentiment"))
        emotion = normalize_label(segment.get("dominant_emotion"))
        escalation_score = safe_float(segment.get("escalation_score"), 0.0)

        if sentiment == "negative":
            negative_count += 1
        elif sentiment == "positive":
            positive_count += 1
        else:
            neutral_count += 1

        if escalation_score >= 0.45 and is_confident_segment(segment):
            high_escalation_count += 1

        if emotion in {"anger", "angry", "fear", "fearful"} and is_confident_segment(segment):
            anger_fear_count += 1

        if is_strong_negative_customer_signal(segment):
            strong_negative_count += 1

    total = len(customer_segments)
    negative_ratio = negative_count / total
    positive_ratio = positive_count / total

    if strong_negative_count >= 3 or high_escalation_count >= 3 or anger_fear_count >= 2:
        calibrated = "Concerned"
    elif negative_ratio >= 0.55 and strong_negative_count >= 1:
        calibrated = "Mixed/Concerned"
    elif positive_ratio >= 0.35 and strong_negative_count == 0:
        calibrated = "Mostly Positive"
    else:
        calibrated = "Neutral/Mixed"

    return {
        "customer_sentiment_calibrated": calibrated,
        "negative_customer_segments": negative_count,
        "positive_customer_segments": positive_count,
        "neutral_customer_segments": neutral_count,
        "strong_negative_customer_segments": strong_negative_count,
        "high_escalation_customer_segments": high_escalation_count,
        "anger_fear_customer_segments": anger_fear_count,
        "negative_customer_ratio": round(negative_ratio, 4),
        "positive_customer_ratio": round(positive_ratio, 4),
        "calibration_note": (
            "Calibrated sentiment uses model sentiment, emotion, escalation score, "
            "speaker role, and repeated negative signals. It does not use transcript keywords."
        ),
    }

def normalize_label(value):
    if value is None:
        return ""
    return str(value).strip().lower()

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_confident_segment(segment: dict) -> bool:
    """
    Decide whether the model output is reliable enough to use as risk evidence.
    This prevents weak negative predictions from creating false escalation warnings.
    """
    emotion_confidence = safe_float(segment.get("emotion_confidence"), 0.0)
    sentiment_confidence = safe_float(segment.get("sentiment_confidence"), 0.0)
    negative_probability = safe_float(segment.get("negative_emotion_probability"), 0.0)

    audio = segment.get("audio_features") or {}
    quality = audio.get("audio_quality_flags") or {}

    very_short = bool(quality.get("very_short_segment"))
    low_energy = bool(quality.get("low_energy_segment"))

    if very_short:
        return False

    if low_energy and emotion_confidence < 0.75:
        return False

    return (
        emotion_confidence >= 0.65
        or sentiment_confidence >= 0.70
        or negative_probability >= 0.70
    )


def is_strong_negative_customer_signal(segment: dict) -> bool:
    """
    Strong customer concern signal using only model/audio outputs.
    No transcript keywords.
    """
    speaker = normalize_label(segment.get("speaker"))
    sentiment = normalize_label(segment.get("sentiment"))
    emotion = normalize_label(segment.get("dominant_emotion"))
    escalation_score = safe_float(segment.get("escalation_score"), 0.0)
    negative_probability = safe_float(segment.get("negative_emotion_probability"), 0.0)
    emotion_confidence = safe_float(segment.get("emotion_confidence"), 0.0)

    if speaker != "customer":
        return False

    if not is_confident_segment(segment):
        return False

    if sentiment != "negative":
        return False

    if escalation_score >= 0.55:
        return True

    if (
        emotion in {"anger", "angry", "fear", "fearful"}
        and emotion_confidence >= 0.65
        and escalation_score >= 0.45
    ):
        return True

    if negative_probability >= 0.80 and escalation_score >= 0.45:
        return True

    return False


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
    audio_quality = audio.get("audio_quality_flags") or {}
    speech_rate_reliability = normalize_label(audio_quality.get("speech_rate_reliability"))

    pitch_level = normalize_label(audio.get("pitch_level"))
    volume_level = normalize_label(audio.get("volume_level"))
    pause_level = normalize_label(audio.get("pause_level"))
    speech_rate_level = normalize_label(audio.get("speech_rate_level"))

    flags = {
        "negative_sentiment": (
            sentiment == "negative"
            and is_confident_segment(segment)
            and safe_float(segment.get("negative_emotion_probability"), 0.0) >= 0.70
        ),
        "anger_emotion": (
            emotion in {"anger", "angry"}
            and is_confident_segment(segment)
            and safe_float(segment.get("emotion_confidence"), 0.0) >= 0.65
        ),
        "sadness_or_fear_emotion": (
            emotion in {"sadness", "sad", "fear", "fearful"}
            and is_confident_segment(segment)
            and safe_float(segment.get("emotion_confidence"), 0.0) >= 0.70
        ),
        "low_confidence_prediction": not is_confident_segment(segment),
        "high_escalation_score": escalation_score is not None and escalation_score >= 0.6,
        "medium_escalation_score": escalation_score is not None and 0.4 <= escalation_score < 0.6,
        "high_pitch": pitch_level == "high",
        "high_volume": volume_level == "high",
        "fast_speech": speech_rate_level in {"fast", "very_fast"} and speech_rate_reliability != "low",
        "unreliable_speech_rate": speech_rate_reliability == "low",
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

    if flags["unreliable_speech_rate"]:
        reasons.append("Speech rate may be unreliable due to short segment length")

    if flags["high_pause"]:
        reasons.append("Long or frequent pauses detected")
    if flags["low_confidence_prediction"]:
        reasons.append("Low confidence model prediction")

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


def build_manager_review_recommendation(payload: dict) -> dict:
    """
    Create a manager review recommendation using only sentiment/audio signals.

    Review is required only when multiple strong model signals agree.
    """
    call_summary = payload.get("call_summary", {}) or {}
    speaker_summary = payload.get("speaker_audio_feature_summary", {}) or {}
    customer_summary = speaker_summary.get("customer", {}) or {}
    customer_trend = payload.get("customer_escalation_trend", {}) or {}
    calibrated = payload.get("calibrated_sentiment_summary", {}) or {}
    segments = payload.get("segments", []) or []

    reasons = []
    notes = []

    risk_level = normalize_label(call_summary.get("risk_level"))
    trend = normalize_label(customer_trend.get("trend"))

    max_escalation = 0.0
    customer_negative_escalation_segments = 0
    customer_anger_fear_segments = 0
    customer_strong_negative_segments = 0

    for segment in segments:
        score = segment.get("escalation_score")

        if isinstance(score, (int, float)):
            max_escalation = max(max_escalation, score)

        if normalize_label(segment.get("speaker")) != "customer":
            continue

        emotion = normalize_label(segment.get("dominant_emotion"))
        escalation_score = safe_float(segment.get("escalation_score"), 0.0)

        if is_strong_negative_customer_signal(segment):
            customer_strong_negative_segments += 1

            if escalation_score >= 0.45:
                customer_negative_escalation_segments += 1

            if emotion in {"anger", "angry", "fear", "fearful"}:
                customer_anger_fear_segments += 1

    if risk_level == "high":
        reasons.append("Call risk level is High")

    if max_escalation >= 0.60:
        reasons.append(f"Maximum escalation score reached {round(max_escalation, 3)}")

    if trend == "increasing":
        reasons.append("Customer escalation increased during the call")

    if customer_negative_escalation_segments >= 3:
        reasons.append(
            f"{customer_negative_escalation_segments} customer segments had negative sentiment with elevated escalation"
        )

    if customer_anger_fear_segments >= 2:
        reasons.append(
            f"{customer_anger_fear_segments} customer segments showed negative anger/fear signals"
        )

    customer_sentiment = normalize_label(customer_summary.get("dominant_sentiment"))
    customer_emotion = normalize_label(customer_summary.get("dominant_emotion"))

    if customer_sentiment == "negative":
        notes.append("Raw model customer sentiment was Negative")

    if customer_emotion:
        notes.append(f"Raw model customer dominant emotion was {customer_emotion}")

    high_pause_segments = customer_summary.get("high_pause_segments", 0) or 0
    if high_pause_segments >= 10:
        notes.append(f"Customer had {high_pause_segments} high-pause segments")

    unreliable_speech_segments = 0

    for segment in segments:
        flags = (
            segment.get("audio_features", {}) or {}
        ).get("audio_quality_flags", {}) or {}

        if flags.get("unrealistic_speech_rate"):
            unreliable_speech_segments += 1

    if unreliable_speech_segments > 0:
        notes.append(
            f"{unreliable_speech_segments} segments had speech-rate reliability warnings"
        )

    review_required = len(reasons) > 0

    if risk_level == "high" or max_escalation >= 0.60 or trend == "increasing":
        review_level = "high"
    elif review_required:
        review_level = "medium"
    else:
        review_level = "low"

    if not reasons:
        reasons.append("No high-confidence manager review indicators detected")

    return {
        "review_required": review_required,
        "review_level": review_level,
        "max_escalation_score": round(max_escalation, 4),
        "customer_negative_escalation_segments": customer_negative_escalation_segments,
        "customer_anger_fear_segments": customer_anger_fear_segments,
        "customer_strong_negative_segments": customer_strong_negative_segments,
        "calibrated_customer_sentiment": calibrated.get("customer_sentiment_calibrated"),
        "reasons": reasons,
        "notes": notes,
    }


def calculate_segment_review_score(segment: dict) -> float:
    """
    Calculate a sentiment/audio-only flag score.

    This uses model outputs only:
    - escalation score
    - sentiment
    - emotion
    - speaker role
    - audio explainability flags

    It does not use transcript keywords.
    """
    score = 0.0

    escalation_score = segment.get("escalation_score")
    if isinstance(escalation_score, (int, float)):
        score += escalation_score

    sentiment = normalize_label(segment.get("sentiment"))
    emotion = normalize_label(segment.get("dominant_emotion"))
    speaker = normalize_label(segment.get("speaker"))
    flags = segment.get("explainability_flags") or {}

    if speaker == "customer":
        score += 0.10

    if sentiment == "negative" and is_confident_segment(segment):
        score += 0.20
    elif sentiment == "mixed" and is_confident_segment(segment):
        score += 0.10

    if emotion in {"anger", "angry", "fear", "fearful"} and is_confident_segment(segment):
        score += 0.25
    elif emotion in {"sadness", "sad", "disgust"} and is_confident_segment(segment):
        score += 0.08

    if not is_confident_segment(segment):
        score -= 0.20

    # Audio features are supporting signals only.
    if flags.get("high_pitch"):
        score += 0.03

    if flags.get("high_volume"):
        score += 0.05

    if flags.get("high_pause"):
        score += 0.05

    if flags.get("fast_speech"):
        score += 0.02

    if flags.get("high_escalation_score"):
        score += 0.20

    if flags.get("medium_escalation_score"):
        score += 0.08

    return round(max(score, 0.0), 4)

def build_top_risky_segments(sentiment_payload: dict, limit: int = 5) -> list:
    """
    Select sentiment/audio flagged moments.

    These are not keyword-based and not automatic escalations.
    They are only the strongest model-signal segments.
    """
    flagged_segments = []

    for segment in sentiment_payload.get("segments", []):
        review_score = calculate_segment_review_score(segment)

        sentiment = normalize_label(segment.get("sentiment"))
        emotion = normalize_label(segment.get("dominant_emotion"))
        speaker = normalize_label(segment.get("speaker"))
        escalation_score = segment.get("escalation_score") or 0
        explanations = segment.get("escalation_explanation") or []

        # Do not show positive segments as sentiment flags unless escalation is high.
        if sentiment == "positive" and escalation_score < 0.55:
            continue

        # Agent segments should only appear if they are genuinely high escalation.
        if speaker == "agent" and escalation_score < 0.55:
            continue

        should_keep = is_strong_negative_customer_signal(segment)

        if not should_keep:
            continue

        display_reasons = []

        if sentiment == "negative":
            display_reasons.append("Negative sentiment detected")

        if emotion in {"anger", "angry", "fear", "fearful"}:
            display_reasons.append("Emotion may indicate escalation")

        if escalation_score >= 0.55:
            display_reasons.append("High escalation score")

        audio_reasons = [
            reason for reason in explanations
            if reason in {
                "High pitch detected",
                "High volume detected",
                "Long or frequent pauses detected",
                "Fast speech rate detected",
            }
        ]

        if audio_reasons:
            display_reasons.append(f"Supporting audio signal: {audio_reasons[0]}")

        if not display_reasons:
            display_reasons.append("Model flagged this segment for review")

        flagged_segments.append({
            "segment_index": segment.get("segment_index"),
            "seq_id": segment.get("seq_id"),
            "speaker": segment.get("speaker"),
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "text": segment.get("text"),
            "sentiment": segment.get("sentiment"),
            "dominant_emotion": segment.get("dominant_emotion"),
            "escalation_score": escalation_score,
            "review_score": review_score,
            "reasons": display_reasons,
        })

    flagged_segments.sort(
        key=lambda item: (
            item.get("review_score") or 0,
            item.get("escalation_score") or 0,
        ),
        reverse=True,
    )

    return flagged_segments[:limit]



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
            "audio_quality_flags": audio.get("audio_quality_flags"),
            "speech_rate_reliability": audio.get("speech_rate_reliability"),
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
    sentiment["calibrated_sentiment_summary"] = build_calibrated_call_sentiment(sentiment)
    sentiment["manager_review_recommendation"] = build_manager_review_recommendation(sentiment)
    sentiment["top_risky_segments"] = build_top_risky_segments(sentiment)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment, f, indent=2)

    print(f"Saved merged sentiment + audio features to: {output_path}")
    print(f"Matched segments: {matched_count}/{len(sentiment.get('segments', []))}")
    print("Dashboard audio feature series added.")


if __name__ == "__main__":
    main()