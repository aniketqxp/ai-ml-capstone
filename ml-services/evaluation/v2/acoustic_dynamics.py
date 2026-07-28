"""Reliability-gated acoustic normalization and temporal episode extraction."""
from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from .schemas import (
    Modality,
    ModalityCoverage,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SignalEpisode,
    SignalRecord,
    SignalScope,
    SourceProvenance,
    Speaker,
    Visibility,
)

DYNAMICS_VERSION = "0.1.0"
DERIVED_PREFIXES = ("acoustic.normalized.", "acoustic.dynamics.")
COVERAGE_PREFIX = "v2_dynamics:"
EPISODE_TYPE = "acoustic.customer_elevation"


@dataclass(frozen=True)
class DynamicsConfig:
    min_feature_observations: int = 12
    usable_coverage_ratio: float = 0.50
    limited_coverage_ratio: float = 0.35
    corroborating_z: float = 1.0
    strong_z: float = 1.5
    max_customer_position_gap: int = 2
    max_episode_time_gap_seconds: float = 75.0
    min_episode_segments: int = 2
    min_episode_span_seconds: float = 2.0
    trajectory_change_z: float = 0.5


DEFAULT_CONFIG = DynamicsConfig()

_NORMALIZED_FEATURES = {
    "acoustic.escalation.score": "escalation",
    "acoustic.pitch.mean": "pitch",
    "acoustic.volume.mean": "volume",
    "acoustic.energy.rms_mean": "energy",
    "acoustic.pause.ratio": "pause",
    "acoustic.speech_rate": "speech_rate",
}

_BLOCKING_FLAGS = {
    "acoustic.escalation.score": set(),
    "acoustic.pitch.mean": {
        "very_short_segment",
        "low_energy_segment",
        "missing_pitch",
    },
    "acoustic.volume.mean": {
        "very_short_segment",
        "low_energy_segment",
    },
    "acoustic.energy.rms_mean": {
        "very_short_segment",
        "low_energy_segment",
    },
    "acoustic.pause.ratio": {
        "very_short_segment",
    },
    "acoustic.speech_rate": {
        "very_short_segment",
        "unrealistic_speech_rate",
    },
}


def _robust_zscores(values: list[float]) -> list[float] | None:
    """Median/MAD z-scores with an IQR fallback for low-MAD features."""
    if not values:
        return None
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    scale = 1.4826 * statistics.median(deviations)

    if scale <= 1e-12 and len(values) >= 4:
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        lower = ordered[:midpoint]
        upper = ordered[midpoint + (len(ordered) % 2):]
        if lower and upper:
            iqr = statistics.median(upper) - statistics.median(lower)
            scale = iqr / 1.349 if iqr > 0 else 0.0
    if scale <= 1e-12:
        return None
    return [
        round(max(-6.0, min(6.0, (value - median) / scale)), 6)
        for value in values
    ]


def _coverage_reliability(
    usable: int,
    expected: int,
    config: DynamicsConfig,
) -> ReliabilityAssessment:
    ratio = usable / expected if expected else 0.0
    if usable < config.min_feature_observations:
        return ReliabilityAssessment(
            status=ReliabilityStatus.UNAVAILABLE,
            coverage_ratio=ratio,
            reasons=["insufficient_usable_observations"],
        )
    if ratio >= config.usable_coverage_ratio:
        return ReliabilityAssessment(
            status=ReliabilityStatus.USABLE,
            coverage_ratio=ratio,
        )
    if ratio >= config.limited_coverage_ratio:
        return ReliabilityAssessment(
            status=ReliabilityStatus.LIMITED,
            coverage_ratio=ratio,
            reasons=["partial_feature_coverage"],
        )
    return ReliabilityAssessment(
        status=ReliabilityStatus.UNAVAILABLE,
        coverage_ratio=ratio,
        reasons=["feature_coverage_below_engineering_gate"],
    )


def _signal_is_usable(signal: SignalRecord) -> bool:
    if signal.reliability.status in (
        ReliabilityStatus.UNUSABLE,
        ReliabilityStatus.UNAVAILABLE,
    ):
        return False
    blocked = _BLOCKING_FLAGS.get(signal.name, set())
    return not blocked.intersection(signal.reliability.quality_flags)


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.acoustic_dynamics",
        producer_version=DYNAMICS_VERSION,
        method=(
            "within_speaker_robust_normalization_and_provisional_"
            "persistence_rules"
        ),
    )


def _clean_previous(bundle: SignalBundle) -> SignalBundle:
    clean = bundle.model_copy(deep=True)
    clean.episodes = [
        episode
        for episode in clean.episodes
        if episode.episode_type != EPISODE_TYPE
    ]
    clean.signals = [
        signal
        for signal in clean.signals
        if not signal.name.startswith(DERIVED_PREFIXES)
    ]
    clean.coverage = [
        coverage
        for coverage in clean.coverage
        if not coverage.source.startswith(COVERAGE_PREFIX)
    ]
    clean.sources = [
        item
        for item in clean.sources
        if item.producer != "evaluator_v2.acoustic_dynamics"
    ]
    return clean


def _normalize_features(
    bundle: SignalBundle,
    config: DynamicsConfig,
    provenance: SourceProvenance,
) -> tuple[list[SignalRecord], list[ModalityCoverage]]:
    segment_counts = {
        speaker: sum(
            segment.speaker == speaker
            for segment in bundle.segments
        )
        for speaker in (Speaker.AGENT, Speaker.CUSTOMER)
    }
    groups: dict[tuple[Speaker, str], list[SignalRecord]] = {}
    for signal in bundle.signals:
        if (
            signal.name not in _NORMALIZED_FEATURES
            or signal.scope != SignalScope.SEGMENT
            or signal.speaker not in segment_counts
            or not isinstance(signal.value, (int, float))
            or isinstance(signal.value, bool)
            or not _signal_is_usable(signal)
        ):
            continue
        groups.setdefault((signal.speaker, signal.name), []).append(signal)

    normalized = []
    coverage_rows = []
    for speaker in (Speaker.AGENT, Speaker.CUSTOMER):
        for raw_name, short_name in _NORMALIZED_FEATURES.items():
            signals = groups.get((speaker, raw_name), [])
            expected = segment_counts[speaker]
            reliability = _coverage_reliability(
                len(signals),
                expected,
                config,
            )
            limitations = list(reliability.reasons)
            coverage_rows.append(
                ModalityCoverage(
                    modality=Modality.ACOUSTIC,
                    source=f"{COVERAGE_PREFIX}{speaker.value}:{short_name}",
                    expected_units=expected,
                    usable_units=len(signals),
                    coverage_ratio=(
                        round(len(signals) / expected, 6)
                        if expected
                        else 0.0
                    ),
                    limitations=limitations,
                )
            )
            if reliability.status == ReliabilityStatus.UNAVAILABLE:
                continue

            values = [float(signal.value) for signal in signals]
            zscores = _robust_zscores(values)
            if zscores is None:
                coverage_rows[-1].limitations.append(
                    "feature_has_no_usable_within_speaker_variation"
                )
                continue
            for raw, zscore in zip(signals, zscores):
                normalized.append(
                    SignalRecord(
                        signal_id=(
                            f"{raw.segment_id}:acoustic.normalized."
                            f"{short_name}.robust_z"
                        ),
                        name=(
                            f"acoustic.normalized.{short_name}.robust_z"
                        ),
                        modality=Modality.ACOUSTIC,
                        scope=SignalScope.SEGMENT,
                        value=zscore,
                        unit="robust_z",
                        segment_id=raw.segment_id,
                        seq_id=raw.seq_id,
                        speaker=raw.speaker,
                        start_seconds=raw.start_seconds,
                        end_seconds=raw.end_seconds,
                        reliability=reliability,
                        provenance=provenance,
                        visibility=Visibility.INTERNAL,
                    )
                )
    return normalized, coverage_rows


def _normalized_by_segment(
    signals: Iterable[SignalRecord],
) -> dict[str, dict[str, SignalRecord]]:
    out: dict[str, dict[str, SignalRecord]] = {}
    for signal in signals:
        if signal.segment_id:
            out.setdefault(signal.segment_id, {})[signal.name] = signal
    return out


def _segment_elevation(
    values: dict[str, SignalRecord],
    config: DynamicsConfig,
) -> tuple[bool, list[str]]:
    def z(feature: str) -> float | None:
        signal = values.get(
            f"acoustic.normalized.{feature}.robust_z"
        )
        return float(signal.value) if signal else None

    emotion = z("escalation")
    intensity_values = [
        value
        for value in (z("volume"), z("energy"))
        if value is not None
    ]
    families = {
        "pitch": z("pitch"),
        "intensity": max(intensity_values) if intensity_values else None,
        "speech_rate": z("speech_rate"),
        "pause": z("pause"),
    }
    corroborating = [
        name
        for name, value in families.items()
        if value is not None and value >= config.corroborating_z
    ]
    strong = [
        name
        for name, value in families.items()
        if value is not None and value >= config.strong_z
    ]

    elevated = (
        emotion is not None
        and emotion >= config.corroborating_z
        and bool(corroborating)
    ) or len(strong) >= 2
    reasons = list(corroborating)
    if emotion is not None and emotion >= config.corroborating_z:
        reasons.insert(0, "emotion_model")
    return elevated, reasons


def _elevation_signals(
    bundle: SignalBundle,
    normalized: list[SignalRecord],
    config: DynamicsConfig,
    provenance: SourceProvenance,
) -> list[SignalRecord]:
    values_by_segment = _normalized_by_segment(normalized)
    output = []
    for segment in bundle.segments:
        if segment.speaker != Speaker.CUSTOMER:
            continue
        values = values_by_segment.get(segment.segment_id, {})
        if not values:
            continue
        elevated, reasons = _segment_elevation(values, config)
        output.append(
            SignalRecord(
                signal_id=(
                    f"{segment.segment_id}:"
                    "acoustic.dynamics.segment_elevated"
                ),
                name="acoustic.dynamics.segment_elevated",
                modality=Modality.ACOUSTIC,
                scope=SignalScope.SEGMENT,
                value=elevated,
                unit=None,
                segment_id=segment.segment_id,
                seq_id=segment.seq_id,
                speaker=segment.speaker,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                reliability=ReliabilityAssessment(
                    status=ReliabilityStatus.LIMITED,
                    reasons=[
                        "provisional_multisignal_episode_rule",
                        *[f"corroborating:{reason}" for reason in reasons],
                    ],
                ),
                provenance=provenance,
                visibility=Visibility.INTERNAL,
            )
        )
    return output


def _build_episodes(
    bundle: SignalBundle,
    elevation_signals: list[SignalRecord],
    config: DynamicsConfig,
    provenance: SourceProvenance,
) -> list[SignalEpisode]:
    customers = [
        segment
        for segment in bundle.segments
        if segment.speaker == Speaker.CUSTOMER
    ]
    positions = {
        segment.segment_id: index
        for index, segment in enumerate(customers)
    }
    elevated = [
        signal
        for signal in elevation_signals
        if signal.value is True and signal.segment_id in positions
    ]
    elevated.sort(key=lambda signal: positions[signal.segment_id])

    groups: list[list[SignalRecord]] = []
    current: list[SignalRecord] = []
    for signal in elevated:
        if not current:
            current = [signal]
            continue
        previous = current[-1]
        position_gap = (
            positions[signal.segment_id] - positions[previous.segment_id]
        )
        time_gap = (
            (signal.start_seconds or 0.0)
            - (previous.end_seconds or previous.start_seconds or 0.0)
        )
        if (
            position_gap <= config.max_customer_position_gap
            and time_gap <= config.max_episode_time_gap_seconds
        ):
            current.append(signal)
        else:
            groups.append(current)
            current = [signal]
    if current:
        groups.append(current)

    episodes = []
    for group in groups:
        start = min(signal.start_seconds or 0.0 for signal in group)
        end = max(
            signal.end_seconds or signal.start_seconds or 0.0
            for signal in group
        )
        if (
            len(group) < config.min_episode_segments
            or end - start < config.min_episode_span_seconds
        ):
            continue
        episode_number = len(episodes) + 1
        episodes.append(
            SignalEpisode(
                episode_id=f"customer-elevation-{episode_number:02d}",
                episode_type=EPISODE_TYPE,
                modality=Modality.ACOUSTIC,
                speaker=Speaker.CUSTOMER,
                start_seconds=start,
                end_seconds=end,
                segment_ids=[signal.segment_id for signal in group],
                signal_ids=[signal.signal_id for signal in group],
                observation=(
                    f"Persistent acoustic elevation candidate across "
                    f"{len(group)} customer segments."
                ),
                reliability=ReliabilityAssessment(
                    status=ReliabilityStatus.LIMITED,
                    reasons=[
                        "provisional_persistence_rule_not_validated"
                    ],
                ),
                provenance=provenance,
                visibility=Visibility.INTERNAL,
            )
        )
    return episodes


def _phase_rows(
    bundle: SignalBundle,
    elevation_signals: list[SignalRecord],
    normalized: list[SignalRecord],
) -> dict[str, list[tuple[str, float, bool, float | None]]]:
    elevated = {
        signal.segment_id: bool(signal.value)
        for signal in elevation_signals
        if signal.segment_id
    }
    escalation = {
        signal.segment_id: float(signal.value)
        for signal in normalized
        if (
            signal.name == "acoustic.normalized.escalation.robust_z"
            and signal.segment_id
        )
    }
    customer_segments = [
        segment
        for segment in bundle.segments
        if segment.speaker == Speaker.CUSTOMER
        and segment.segment_id in elevated
    ]
    customer_segments.sort(
        key=lambda segment: (segment.start_seconds, segment.segment_index)
    )
    total_seconds = sum(
        max(0.01, segment.end_seconds - segment.start_seconds)
        for segment in customer_segments
    )
    phases = {"early": [], "middle": [], "late": []}
    elapsed = 0.0
    for segment in customer_segments:
        duration = max(0.01, segment.end_seconds - segment.start_seconds)
        midpoint = (elapsed + duration / 2) / total_seconds if total_seconds else 0
        if midpoint < 1 / 3:
            phase = "early"
        elif midpoint < 2 / 3:
            phase = "middle"
        else:
            phase = "late"
        phases[phase].append(
            (
                segment.segment_id,
                duration,
                elevated[segment.segment_id],
                escalation.get(segment.segment_id),
            )
        )
        elapsed += duration
    return phases


def _phase_metrics(
    rows: list[tuple[str, float, bool, float | None]],
) -> tuple[float, float | None]:
    total = sum(duration for _, duration, _, _ in rows)
    elevated = sum(
        duration
        for _, duration, is_elevated, _ in rows
        if is_elevated
    )
    escalation = [
        value
        for _, _, _, value in rows
        if value is not None
    ]
    return (
        round(elevated / total, 6) if total else 0.0,
        round(statistics.median(escalation), 6) if escalation else None,
    )


def _call_signal(
    *,
    name: str,
    value: bool | float | str,
    unit: str | None,
    reliability: ReliabilityAssessment,
    provenance: SourceProvenance,
) -> SignalRecord:
    return SignalRecord(
        signal_id=f"call:{name}",
        name=name,
        modality=Modality.ACOUSTIC,
        scope=SignalScope.CALL,
        value=value,
        unit=unit,
        reliability=reliability,
        provenance=provenance,
        visibility=Visibility.INTERNAL,
    )


def _trajectory_signals(
    bundle: SignalBundle,
    normalized: list[SignalRecord],
    elevation_signals: list[SignalRecord],
    episodes: list[SignalEpisode],
    config: DynamicsConfig,
    provenance: SourceProvenance,
) -> list[SignalRecord]:
    phases = _phase_rows(bundle, elevation_signals, normalized)
    early_ratio, early_median = _phase_metrics(phases["early"])
    late_ratio, late_median = _phase_metrics(phases["late"])
    usable_seconds = {
        phase: round(sum(row[1] for row in rows), 6)
        for phase, rows in phases.items()
    }
    delta = (
        round(late_median - early_median, 6)
        if early_median is not None and late_median is not None
        else None
    )
    late_segment_ids = {row[0] for row in phases["late"]}
    late_elevated = {
        signal.segment_id
        for signal in elevation_signals
        if signal.value is True and signal.segment_id in late_segment_ids
    }
    unresolved_end = any(
        set(episode.segment_ids).intersection(late_segment_ids)
        for episode in episodes
    )
    recovery = bool(
        episodes
        and not unresolved_end
        and not late_elevated
        and (late_median is None or late_median < 0.5)
    )
    if delta is None:
        direction = "unavailable"
    elif delta >= config.trajectory_change_z:
        direction = "worsened"
    elif delta <= -config.trajectory_change_z:
        direction = "improved"
    else:
        direction = "stable"

    reliability = ReliabilityAssessment(
        status=ReliabilityStatus.LIMITED,
        reasons=["provisional_trajectory_rule_not_validated"],
    )
    values = [
        ("acoustic.dynamics.episode_count", len(episodes), "count"),
        (
            "acoustic.dynamics.usable_customer_seconds",
            round(sum(usable_seconds.values()), 6),
            "seconds",
        ),
        (
            "acoustic.dynamics.early_usable_customer_seconds",
            usable_seconds["early"],
            "seconds",
        ),
        (
            "acoustic.dynamics.late_usable_customer_seconds",
            usable_seconds["late"],
            "seconds",
        ),
        (
            "acoustic.dynamics.early_elevated_ratio",
            early_ratio,
            "usable_customer_seconds_ratio",
        ),
        (
            "acoustic.dynamics.late_elevated_ratio",
            late_ratio,
            "usable_customer_seconds_ratio",
        ),
        ("acoustic.dynamics.trajectory_direction", direction, None),
        ("acoustic.dynamics.recovery_candidate", recovery, None),
        (
            "acoustic.dynamics.unresolved_end_candidate",
            unresolved_end,
            None,
        ),
    ]
    if delta is not None:
        values.append(
            ("acoustic.dynamics.trajectory_delta", delta, "robust_z")
        )
    return [
        _call_signal(
            name=name,
            value=value,
            unit=unit,
            reliability=reliability,
            provenance=provenance,
        )
        for name, value, unit in values
    ]


def derive_acoustic_dynamics(
    bundle: SignalBundle,
    config: DynamicsConfig = DEFAULT_CONFIG,
) -> SignalBundle:
    """Return a bundle augmented with internal normalized dynamics."""
    output = _clean_previous(bundle)
    provenance = _source()
    normalized, coverage = _normalize_features(
        output,
        config,
        provenance,
    )
    elevation_signals = _elevation_signals(
        output,
        normalized,
        config,
        provenance,
    )
    episodes = _build_episodes(
        output,
        elevation_signals,
        config,
        provenance,
    )
    trajectory = _trajectory_signals(
        output,
        normalized,
        elevation_signals,
        episodes,
        config,
        provenance,
    )

    output.signals.extend(normalized)
    output.signals.extend(elevation_signals)
    output.signals.extend(trajectory)
    output.episodes.extend(episodes)
    output.coverage.extend(coverage)
    output.sources.append(provenance)
    return SignalBundle.model_validate(output.model_dump())
