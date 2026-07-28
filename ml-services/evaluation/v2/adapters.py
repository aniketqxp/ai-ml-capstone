"""Normalize current and enriched pipeline payloads into SignalBundle."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .schemas import (
    Modality,
    ModalityCoverage,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SignalRecord,
    SignalScope,
    SourceProvenance,
    Speaker,
    TranscriptSegment,
    Visibility,
)


_AUDIO_FEATURES = {
    "duration_seconds": ("acoustic.duration", "seconds"),
    "pitch_mean_hz": ("acoustic.pitch.mean", "hz"),
    "pitch_min_hz": ("acoustic.pitch.min", "hz"),
    "pitch_max_hz": ("acoustic.pitch.max", "hz"),
    "pitch_std_hz": ("acoustic.pitch.std", "hz"),
    "rms_energy_mean": ("acoustic.energy.rms_mean", None),
    "rms_energy_max": ("acoustic.energy.rms_max", None),
    "volume_db_mean": ("acoustic.volume.mean", "db"),
    "total_pause_duration_seconds": (
        "acoustic.pause.total_duration",
        "seconds",
    ),
    "pause_count": ("acoustic.pause.count", "count"),
    "pause_ratio": ("acoustic.pause.ratio", "ratio"),
    "speech_rate_words_per_minute": (
        "acoustic.speech_rate",
        "words_per_minute",
    ),
}

_SEGMENT_OUTPUTS = {
    "sentiment": ("acoustic.sentiment.label", None),
    "dominant_emotion": ("acoustic.emotion.label", None),
    "escalation_score": ("acoustic.escalation.score", "ratio"),
    "prediction_confidence": ("acoustic.emotion.top_probability", "ratio"),
    "negative_emotion_probability": (
        "acoustic.emotion.negative_probability",
        "ratio",
    ),
}

_TRAJECTORY_FIELDS = {
    "trajectory_direction": ("candidate.trajectory.direction", None),
    "trajectory_delta": ("candidate.trajectory.delta", "ratio"),
    "start_escalation": ("candidate.trajectory.start_escalation", "ratio"),
    "end_escalation": ("candidate.trajectory.end_escalation", "ratio"),
    "peak_escalation": ("candidate.trajectory.peak_escalation", "ratio"),
    "deescalation_detected": (
        "candidate.trajectory.deescalation_detected",
        None,
    ),
    "unresolved_end_risk": (
        "candidate.trajectory.unresolved_end_risk",
        None,
    ),
}


def _speaker(value: Any) -> Speaker:
    normalized = str(value or "").strip().lower()
    if normalized == "agent":
        return Speaker.AGENT
    if normalized == "customer":
        return Speaker.CUSTOMER
    if normalized == "system":
        return Speaker.SYSTEM
    return Speaker.UNKNOWN


def _transcript_segments(
    transcript: dict[str, Any],
    provenance: SourceProvenance,
) -> list[TranscriptSegment]:
    segments = []
    for index, sentence in enumerate(transcript.get("sentences") or []):
        segment_id = str(
            sentence.get("id")
            or sentence.get("segment_key")
            or f"segment-{index + 1:04d}"
        )
        start = float(sentence.get("start", sentence.get("start_time", 0.0)))
        end = float(sentence.get("end", sentence.get("end_time", start)))
        segments.append(
            TranscriptSegment(
                segment_id=segment_id,
                segment_index=index,
                seq_id=sentence.get("seq_id"),
                speaker=_speaker(sentence.get("speaker")),
                start_seconds=start,
                end_seconds=end,
                text=str(sentence.get("text") or ""),
                provenance=provenance,
            )
        )
    return segments


def _align_sentiment(
    transcript_segments: list[TranscriptSegment],
    sentiment_segments: list[dict[str, Any]],
) -> list[tuple[TranscriptSegment | None, dict[str, Any]]]:
    if len(transcript_segments) == len(sentiment_segments):
        return list(zip(transcript_segments, sentiment_segments))

    by_seq: dict[int, deque[TranscriptSegment]] = defaultdict(deque)
    for segment in transcript_segments:
        if segment.seq_id is not None:
            by_seq[segment.seq_id].append(segment)

    aligned = []
    used = set()
    for sentiment in sentiment_segments:
        match = None
        segment_index = sentiment.get("segment_index")
        if isinstance(segment_index, int):
            possible_indexes = (
                (segment_index - 1, segment_index)
                if segment_index > 0
                else (segment_index,)
            )
            for index in possible_indexes:
                if (
                    0 <= index < len(transcript_segments)
                    and transcript_segments[index].segment_id not in used
                ):
                    match = transcript_segments[index]
                    break

        seq_id = sentiment.get("seq_id")
        if match is None and isinstance(seq_id, int):
            while by_seq[seq_id] and by_seq[seq_id][0].segment_id in used:
                by_seq[seq_id].popleft()
            if by_seq[seq_id]:
                match = by_seq[seq_id].popleft()

        if match is not None:
            used.add(match.segment_id)
        aligned.append((match, sentiment))
    return aligned


def _model_reliability(segment: dict[str, Any]) -> ReliabilityAssessment:
    status = str(segment.get("processing_status") or "unknown")
    reasons = []
    reliability = ReliabilityStatus.USABLE

    if status != "success":
        reliability = ReliabilityStatus.UNAVAILABLE
        reasons.append(f"processing_status:{status}")
    else:
        confidence = segment.get(
            "prediction_confidence",
            segment.get("emotion_confidence"),
        )
        if isinstance(confidence, (int, float)) and confidence < 0.5:
            reliability = ReliabilityStatus.LIMITED
            reasons.append("low_model_probability")

    return ReliabilityAssessment(
        status=reliability,
        reasons=reasons,
    )


def _feature_reliability(segment: dict[str, Any]) -> ReliabilityAssessment:
    features = segment.get("audio_features") or {}
    flags = features.get("audio_quality_flags") or {}
    active_flags = [
        str(name)
        for name, active in flags.items()
        if active is True
    ]
    limiting = {
        "very_short_segment",
        "low_energy_segment",
        "missing_pitch",
        "unrealistic_speech_rate",
    }
    status = (
        ReliabilityStatus.LIMITED
        if limiting.intersection(active_flags)
        else ReliabilityStatus.USABLE
    )
    reasons = (
        ["one_or_more_feature_quality_flags"]
        if status == ReliabilityStatus.LIMITED
        else []
    )
    return ReliabilityAssessment(
        status=status,
        reasons=reasons,
        quality_flags=active_flags,
    )


def _append_segment_signal(
    signals: list[SignalRecord],
    *,
    segment: TranscriptSegment,
    name: str,
    value: Any,
    unit: str | None,
    reliability: ReliabilityAssessment,
    provenance: SourceProvenance,
):
    if value is None or isinstance(value, (dict, list)):
        return
    signals.append(
        SignalRecord(
            signal_id=f"{segment.segment_id}:{name}",
            name=name,
            modality=Modality.ACOUSTIC,
            scope=SignalScope.SEGMENT,
            value=value,
            unit=unit,
            segment_id=segment.segment_id,
            seq_id=segment.seq_id,
            speaker=segment.speaker,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            reliability=reliability,
            provenance=provenance,
            visibility=Visibility.INTERNAL,
        )
    )


def _candidate_call_signals(
    sentiment: dict[str, Any],
    provenance: SourceProvenance,
) -> list[SignalRecord]:
    signals = []
    reliability = ReliabilityAssessment(
        status=ReliabilityStatus.LIMITED,
        reasons=["imported_unvalidated_derived_signal"],
    )
    trajectory = sentiment.get("temporal_emotion_trajectory") or {}

    for field, (name, unit) in _TRAJECTORY_FIELDS.items():
        value = trajectory.get(field)
        if value is None or isinstance(value, (dict, list)):
            continue
        signals.append(
            SignalRecord(
                signal_id=f"call:{name}",
                name=name,
                modality=Modality.MULTIMODAL,
                scope=SignalScope.CALL,
                value=value,
                unit=unit,
                reliability=reliability,
                provenance=provenance,
                visibility=Visibility.INTERNAL,
            )
        )
    return signals


def build_signal_bundle(
    transcript: dict[str, Any],
    sentiment: dict[str, Any] | None = None,
    *,
    transcript_source: str | None = None,
    sentiment_source: str | None = None,
) -> SignalBundle:
    """Map current or Clara-enriched payloads without applying v2 decisions."""
    sentiment = sentiment or {}
    transcript_call_id = str(transcript.get("call_id") or "").strip()
    sentiment_call_id = str(sentiment.get("call_id") or "").strip()
    if sentiment_call_id and sentiment_call_id != transcript_call_id:
        raise ValueError("transcript and sentiment call_id values do not match")

    transcript_provenance = SourceProvenance(
        producer="sentence_segmentation",
        producer_version=str(transcript.get("model") or "unknown"),
        source_artifact=transcript_source,
        method="per_channel_sentence_segmentation",
    )
    sentiment_provenance = SourceProvenance(
        producer=(
            "enriched_acoustic_pipeline"
            if sentiment.get("has_audio_features")
            else "acoustic_sentiment"
        ),
        producer_version=str(sentiment.get("model_version") or "unknown"),
        source_artifact=sentiment_source,
        method="sentence_aligned_audio_inference",
    )
    segments = _transcript_segments(transcript, transcript_provenance)
    sentiment_segments = [
        value
        for value in (sentiment.get("segments") or [])
        if isinstance(value, dict)
    ]
    aligned = _align_sentiment(segments, sentiment_segments)

    signals = []
    usable_model_segments = 0
    feature_segments = 0
    usable_feature_segments = 0
    for segment, sentiment_segment in aligned:
        if segment is None:
            continue

        model_reliability = _model_reliability(sentiment_segment)
        if model_reliability.status == ReliabilityStatus.USABLE:
            usable_model_segments += 1
        for field, (name, unit) in _SEGMENT_OUTPUTS.items():
            _append_segment_signal(
                signals,
                segment=segment,
                name=name,
                value=sentiment_segment.get(field),
                unit=unit,
                reliability=model_reliability,
                provenance=sentiment_provenance,
            )

        features = sentiment_segment.get("audio_features")
        if not isinstance(features, dict):
            continue
        feature_segments += 1
        feature_reliability = _feature_reliability(sentiment_segment)
        if feature_reliability.status == ReliabilityStatus.USABLE:
            usable_feature_segments += 1
        for field, (name, unit) in _AUDIO_FEATURES.items():
            _append_segment_signal(
                signals,
                segment=segment,
                name=name,
                value=features.get(field),
                unit=unit,
                reliability=feature_reliability,
                provenance=sentiment_provenance,
            )

    signals.extend(_candidate_call_signals(sentiment, sentiment_provenance))

    expected = len(segments)
    coverage = [
        ModalityCoverage(
            modality=Modality.TRANSCRIPT,
            source="sentence_segments",
            expected_units=expected,
            usable_units=expected,
            coverage_ratio=1.0 if expected else 0.0,
        )
    ]
    if sentiment:
        coverage.append(
            ModalityCoverage(
                modality=Modality.ACOUSTIC,
                source="emotion_model",
                expected_units=expected,
                usable_units=usable_model_segments,
                coverage_ratio=(
                    round(usable_model_segments / expected, 6)
                    if expected
                    else 0.0
                ),
                limitations=(
                    ["segments_skipped_or_unreliable"]
                    if usable_model_segments < expected
                    else []
                ),
            )
        )
        if feature_segments:
            coverage.append(
                ModalityCoverage(
                    modality=Modality.ACOUSTIC,
                    source="audio_features",
                    expected_units=expected,
                    usable_units=usable_feature_segments,
                    coverage_ratio=(
                        round(usable_feature_segments / expected, 6)
                        if expected
                        else 0.0
                    ),
                    limitations=(
                        ["feature_quality_flags_present"]
                        if usable_feature_segments < feature_segments
                        else []
                    ),
                )
            )

    duration = max((segment.end_seconds for segment in segments), default=0.0)
    sources = [transcript_provenance]
    if sentiment:
        sources.append(sentiment_provenance)
    return SignalBundle(
        call_id=transcript_call_id,
        domain=str(transcript.get("domain") or sentiment.get("domain") or "unknown"),
        accent=transcript.get("accent"),
        duration_seconds=duration,
        transcript_model=transcript.get("model"),
        segments=segments,
        signals=signals,
        coverage=coverage,
        sources=sources,
    )
