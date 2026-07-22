from __future__ import annotations

import json
from typing import Any


MAX_TRANSCRIPT_CHARS = 7500
MAX_EVIDENCE_SEGMENTS = 5
MAX_FIELD_CHARS = 1800


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _truncate_text(
    value: Any,
    max_chars: int = MAX_FIELD_CHARS,
) -> str | None:
    if value is None:
        return None

    text = str(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...[truncated]"


def _compact_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 3,
) -> Any:
    if depth >= max_depth:
        if isinstance(value, (dict, list)):
            return "[nested data omitted]"
        return value

    if isinstance(value, dict):
        compacted = {}

        excluded_keys = {
            "dashboard_audio_feature_series",
            "raw_probabilities",
            "emotion_probabilities",
            "probabilities",
            "class_probabilities",
            "logits",
            "embedding",
            "embeddings",
            "waveform",
            "audio_array",
            "full_segments",
            "all_segments",
            "segments",
            "raw_features",
            "feature_vector",
            "spectrogram",
            "mfcc",
        }

        for key, item in value.items():
            if key in excluded_keys:
                continue

            compacted[key] = _compact_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
            )

        return compacted

    if isinstance(value, list):
        maximum_items = 6 if depth <= 1 else 4

        return [
            _compact_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for item in value[:maximum_items]
        ]

    if isinstance(value, str):
        return _truncate_text(value, 600)

    return value


def _compact_transcript(
    transcript: Any,
) -> list[dict[str, Any]]:
    """
    Keep a representative chronological transcript for the supervisor.

    The detailed lower-level evaluators already processed the complete
    transcript. The supervisor receives the beginning, middle, and end
    plus all deterministic findings.
    """
    turns = [
        turn
        for turn in _safe_list(transcript)
        if isinstance(turn, dict)
        and str(turn.get("text") or "").strip()
    ]

    if not turns:
        return []

    if len(turns) <= 30:
        selected_turns = turns
    else:
        beginning = turns[:12]

        middle_index = len(turns) // 2
        middle = turns[
            max(12, middle_index - 3):
            min(len(turns) - 12, middle_index + 3)
        ]

        ending = turns[-12:]

        selected_turns = (
            beginning
            + [{
                "speaker": "SYSTEM",
                "timestamp": None,
                "text": (
                    "Some routine middle turns were omitted. "
                    "The deterministic evaluations below were produced "
                    "from the complete transcript."
                ),
            }]
            + middle
            + [{
                "speaker": "SYSTEM",
                "timestamp": None,
                "text": (
                    "Additional routine turns were omitted before "
                    "the final call section."
                ),
            }]
            + ending
        )

    compacted = []
    used_characters = 0

    for turn in selected_turns:
        text = str(turn.get("text") or "").strip()

        if not text:
            continue

        record = {
            "timestamp": (
                turn.get("timestamp")
                or turn.get("start")
                or turn.get("start_time")
            ),
            "speaker": turn.get("speaker"),
            "text": _truncate_text(text, 500),
        }

        record_size = len(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        if (
            compacted
            and used_characters + record_size
            > MAX_TRANSCRIPT_CHARS
        ):
            compacted.append({
                "timestamp": None,
                "speaker": "SYSTEM",
                "text": (
                    "Transcript remainder omitted because the context "
                    "budget was reached. Use the complete deterministic "
                    "evaluation and emotional summaries."
                ),
            })
            break

        compacted.append(record)
        used_characters += record_size

    return compacted


def _compact_segments(
    segments: list[dict[str, Any]],
    max_segments: int = MAX_EVIDENCE_SEGMENTS,
) -> list[dict[str, Any]]:
    """
    Keep the strongest sentence-level emotional moments only.
    """
    prepared = []

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        prepared.append({
            "seq_id": segment.get("seq_id"),
            "segment_index": segment.get("segment_index"),
            "speaker": segment.get("speaker"),
            "start_time": segment.get(
                "start_time",
                segment.get("start"),
            ),
            "end_time": segment.get(
                "end_time",
                segment.get("end"),
            ),
            "text": _truncate_text(
                segment.get("text"),
                250,
            ),
            "sentiment": segment.get(
                "sentiment",
                segment.get(
                    "overall_audio_sentiment"
                ),
            ),
            "dominant_emotion": segment.get(
                "dominant_emotion"
            ),
            "emotion_confidence": segment.get(
                "calibrated_emotion_confidence",
                segment.get(
                    "emotion_confidence",
                    segment.get(
                        "prediction_confidence"
                    ),
                ),
            ),
            "uncertain_prediction": segment.get(
                "uncertain_prediction"
            ),
            "escalation_score": segment.get(
                "escalation_score",
                segment.get(
                    "audio_escalation_score"
                ),
            ),
            "escalation_explanation": _safe_list(
                segment.get(
                    "escalation_explanation"
                )
            )[:2],
        })

    prepared.sort(
        key=lambda item: float(
            item.get("escalation_score") or 0
        ),
        reverse=True,
    )

    return prepared[:max_segments]


def _context_size(
    context: dict[str, Any],
) -> int:
    return len(
        json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def build_multimodal_llm_context(
    *,
    call_id: str,
    transcript: Any,
    evaluation: dict[str, Any],
    sentiment: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build a compact multimodal context for the AI Supervisor.

    The complete pipeline is represented, but large raw arrays and
    sentence-level payloads are excluded to stay within provider limits.
    """
    evaluation = _safe_dict(evaluation)
    sentiment = _safe_dict(sentiment)

    sentiment_segments = _safe_list(
        sentiment.get("segments")
    )

    top_risky_segments = _safe_list(
        sentiment.get("top_risky_segments")
    )

    source_evidence = (
        top_risky_segments
        if top_risky_segments
        else sentiment_segments
    )

    important_evidence = _compact_segments(
        source_evidence,
        max_segments=MAX_EVIDENCE_SEGMENTS,
    )

    sentiment_summary = {
        "call_summary": sentiment.get(
            "call_summary"
        ),
        "calibrated_sentiment_summary": sentiment.get(
            "calibrated_sentiment_summary"
        ),
        "audio_feature_summary": sentiment.get(
            "audio_feature_summary"
        ),
        "customer_escalation_trend": sentiment.get(
            "customer_escalation_trend"
        ),
        "manager_review_recommendation": sentiment.get(
            "manager_review_recommendation"
        ),
        "multi_signal_escalation_intelligence": sentiment.get(
            "multi_signal_escalation_intelligence"
        ),
        "temporal_emotion_trajectory": sentiment.get(
            "temporal_emotion_trajectory"
        ),
        "agent_empathy_tone_alignment": sentiment.get(
            "agent_empathy_tone_alignment"
        ),
        "model_name": sentiment.get("model_name"),
        "model_version": sentiment.get(
            "model_version"
        ),
        "segment_count": len(sentiment_segments),
    }

    deterministic_evaluation = {
        "compliance": evaluation.get("compliance"),
        "quality": evaluation.get("quality"),
        "workflow": evaluation.get("workflow"),
        "escalation": evaluation.get("escalation"),
        "investigation": evaluation.get(
            "investigation"
        ),
        "overall_summary": evaluation.get(
            "overall_summary"
        ),
        "rubric_version": evaluation.get(
            "rubric_version"
        ),
        "anchor_stats": evaluation.get(
            "_anchor_stats"
        ),
    }

    context = {
        "call_id": call_id,

        "transcript": _compact_transcript(
            transcript
        ),

        "deterministic_evaluation": _compact_value(
            deterministic_evaluation
        ),

        "sentiment_and_audio": _compact_value(
            sentiment_summary
        ),

        "important_emotional_evidence": important_evidence,

        "data_availability": {
            "has_transcript": bool(transcript),

            "has_compliance": bool(
                evaluation.get("compliance")
            ),

            "has_quality": bool(
                evaluation.get("quality")
            ),

            "has_workflow": bool(
                evaluation.get("workflow")
            ),

            "has_investigation": bool(
                evaluation.get("investigation")
            ),

            "has_sentiment": bool(sentiment),

            "has_sentiment_segments": bool(
                sentiment_segments
            ),

            "has_audio_features": bool(
                sentiment.get("has_audio_features")
                or sentiment.get(
                    "audio_feature_summary"
                )
            ),

            "has_calibrated_sentiment": bool(
                sentiment.get(
                    "calibrated_sentiment_summary"
                )
            ),

            "has_multi_signal_escalation": bool(
                sentiment.get(
                    "multi_signal_escalation_intelligence"
                )
            ),

            "has_temporal_analysis": bool(
                sentiment.get(
                    "temporal_emotion_trajectory"
                )
            ),

            "has_agent_tone_analysis": bool(
                sentiment.get(
                    "agent_empathy_tone_alignment"
                )
            ),
        },
    }
    MAX_CONTEXT_CHARACTERS = 22000

    if _context_size(context) > MAX_CONTEXT_CHARACTERS:
        context["important_emotional_evidence"] = (
            context["important_emotional_evidence"][:3]
        )

    if _context_size(context) > MAX_CONTEXT_CHARACTERS:
        transcript_context = context.get(
            "transcript",
            [],
        )

        if len(transcript_context) > 20:
            context["transcript"] = (
                transcript_context[:10]
                + [{
                    "timestamp": None,
                    "speaker": "SYSTEM",
                    "text": (
                        "Middle transcript turns omitted. "
                        "All deterministic findings still use "
                        "the complete transcript."
                    ),
                }]
                + transcript_context[-10:]
            )

    if _context_size(context) > MAX_CONTEXT_CHARACTERS:
        sentiment_context = context.get(
            "sentiment_and_audio",
            {},
        )

        sentiment_context.pop(
            "audio_feature_summary",
            None,
        )

    if _context_size(context) > MAX_CONTEXT_CHARACTERS:
        context["transcript"] = (
            context.get("transcript", [])[:8]
            + [{
                "timestamp": None,
                "speaker": "SYSTEM",
                "text": (
                    "Most transcript turns omitted to satisfy the "
                    "provider context limit. Complete deterministic "
                    "results and multimodal summaries remain available."
                ),
            }]
            + context.get("transcript", [])[-8:]
        )

    return context