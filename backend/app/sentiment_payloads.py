"""Build the frontend sentiment contract from persisted signal rows."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.models import CallAudioSummary, SentimentSegment


def _number(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _average(rows, *keys):
    values = [value for row in rows if (value := _number(row, *keys)) is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize_audio_series(series):
    rows = list(series or [])
    return {
        "total_segments_with_audio_features": len(rows),
        "average_pitch_hz": _average(rows, "pitch_mean_hz"),
        "average_volume_db": _average(rows, "volume_db_mean", "volume_db"),
        "average_energy": _average(rows, "rms_energy_mean", "rms_mean"),
        "average_pause_ratio": _average(rows, "pause_ratio"),
        "average_speech_rate_wpm": _average(
            rows,
            "speech_rate_words_per_minute",
        ),
    }


def _dominant(rows, key):
    values = [str(row.get(key)) for row in rows if row.get(key)]
    return Counter(values).most_common(1)[0][0] if values else None


def summarize_speakers(series):
    grouped = defaultdict(list)
    for row in series or []:
        grouped[str(row.get("speaker") or "UNKNOWN").upper()].append(row)

    return {
        speaker.lower(): {
            "speaker": speaker,
            "total_segments": len(rows),
            "dominant_sentiment": _dominant(rows, "sentiment"),
            "dominant_emotion": _dominant(rows, "dominant_emotion"),
            "average_escalation_score": _average(rows, "escalation_score"),
            **summarize_audio_series(rows),
        }
        for speaker, rows in grouped.items()
    }


def customer_escalation_trend(series):
    scores = [
        _number(row, "escalation_score")
        for row in series or []
        if str(row.get("speaker") or "").upper() == "CUSTOMER"
    ]
    scores = [value for value in scores if value is not None]
    if len(scores) < 3:
        return {}
    width = max(1, len(scores) // 3)
    early = sum(scores[:width]) / width
    late = sum(scores[-width:]) / width
    delta = late - early
    trend = "rising" if delta >= 0.08 else "falling" if delta <= -0.08 else "stable"
    return {
        "trend": trend,
        "early_mean": round(early, 4),
        "late_mean": round(late, 4),
        "trend_delta": round(delta, 4),
    }


def _serialize_segment(record):
    sentiment = record.sentiment
    return {
        "segment_index": record.segment_index,
        "segment_key": record.segment_key,
        "seq_id": record.seq_id,
        "speaker": record.speaker,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "text": record.text,
        "sentiment": sentiment,
        "sentiment_class": str(sentiment).lower() if sentiment else None,
        "dominant_emotion": record.dominant_emotion,
        "escalation_score": record.escalation_score,
        "processing_status": record.processing_status,
        "audio_features": record.audio_features,
        "explainability_flags": record.explainability_flags,
        "escalation_explanation": record.escalation_explanation,
    }


def build_sentiment_payload(db, call_id, stored=None):
    """Merge old storage JSON with richer rows already present in Supabase."""
    stored = dict(stored or {})
    summary_row = (
        db.query(CallAudioSummary)
        .filter(CallAudioSummary.call_id == call_id)
        .order_by(CallAudioSummary.created_at.desc())
        .first()
    )
    records = (
        db.query(SentimentSegment)
        .filter(SentimentSegment.call_id == call_id)
        .order_by(SentimentSegment.segment_index.asc())
        .all()
    )
    if not summary_row and not records:
        return stored or {"segments": []}

    stored_segments = stored.get("segments") or []
    by_index = {
        row.get("segment_index"): row
        for row in stored_segments
        if row.get("segment_index") is not None
    }
    segments = []
    for record in records:
        payload = dict(by_index.get(record.segment_index) or {})
        payload.update({
            key: value
            for key, value in _serialize_segment(record).items()
            if value is not None
        })
        segments.append(payload)
    if not segments:
        segments = stored_segments

    series = (
        (summary_row.dashboard_audio_feature_series if summary_row else None)
        or stored.get("dashboard_audio_feature_series")
        or []
    )
    call_summary = {
        **(stored.get("call_summary") or {}),
        **((summary_row.call_summary if summary_row else None) or {}),
    }
    audio_summary = stored.get("audio_feature_summary") or {}
    # The persisted feature series is authoritative. Some older artifacts contain
    # placeholder zeroes even though the database has Clara's extracted values.
    audio_summary = {**audio_summary, **summarize_audio_series(series)}

    return {
        **stored,
        "call_id": call_id,
        "domain": stored.get("domain") or getattr(summary_row, "domain", None),
        "model_version": (
            stored.get("model_version")
            or getattr(summary_row, "model_version", None)
        ),
        "has_audio_features": bool(
            series or getattr(summary_row, "has_audio_features", False)
        ),
        "audio_feature_version": (
            stored.get("audio_feature_version")
            or getattr(summary_row, "audio_feature_version", None)
        ),
        "segments": segments,
        "call_summary": call_summary,
        "audio_feature_summary": audio_summary,
        "speaker_audio_feature_summary": (
            stored.get("speaker_audio_feature_summary")
            or summarize_speakers(series)
        ),
        "customer_escalation_trend": (
            stored.get("customer_escalation_trend")
            or customer_escalation_trend(series)
        ),
        "dashboard_audio_feature_series": series,
    }


def load_signal_overviews(db):
    rows = db.query(CallAudioSummary).order_by(CallAudioSummary.created_at.asc()).all()
    result = {}
    for row in rows:
        summary = dict(row.call_summary or {})
        series = list(row.dashboard_audio_feature_series or [])
        audio = summarize_audio_series(series)
        audio.update(summary.get("audio_features") or {})
        result[row.call_id] = {
            "sentiment_available": bool(
                summary.get("dominant_sentiment")
                or any(point.get("sentiment") for point in series)
            ),
            "dominant_sentiment": summary.get("dominant_sentiment"),
            "dominant_emotion": summary.get("dominant_emotion"),
            "average_escalation_score": summary.get("average_escalation_score"),
            "max_escalation_score": summary.get("max_escalation_score"),
            "has_audio_features": bool(row.has_audio_features and series),
            **audio,
        }
    return result
