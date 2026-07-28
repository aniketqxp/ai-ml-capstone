"""Behavioral tests for provisional acoustic dynamics."""
from __future__ import annotations

import unittest

from v2.acoustic_dynamics import (
    DynamicsConfig,
    _robust_zscores,
    derive_acoustic_dynamics,
)
from v2.schemas import (
    Modality,
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

TEST_CONFIG = DynamicsConfig(
    min_feature_observations=6,
    usable_coverage_ratio=0.5,
    limited_coverage_ratio=0.35,
    corroborating_z=0.8,
    strong_z=1.2,
    max_customer_position_gap=2,
    max_episode_time_gap_seconds=20.0,
    min_episode_segments=2,
    min_episode_span_seconds=2.0,
    trajectory_change_z=0.4,
)


def _provenance(producer: str = "test_fixture") -> SourceProvenance:
    return SourceProvenance(
        producer=producer,
        producer_version="1",
        method="synthetic_test_values",
    )


def _bundle(
    escalation: list[float],
    *,
    energy: list[float] | None = None,
    pitch: list[float] | None = None,
    speaker: Speaker = Speaker.CUSTOMER,
    feature_flags: list[list[str]] | None = None,
) -> SignalBundle:
    size = len(escalation)
    energy = energy or list(escalation)
    pitch = pitch or [100.0 + value for value in escalation]
    feature_flags = feature_flags or [[] for _ in range(size)]
    source = _provenance()
    segments = []
    signals = []
    for index in range(size):
        segment_id = f"segment-{index:02d}"
        start = float(index * 3)
        end = start + 2.0
        segments.append(
            TranscriptSegment(
                segment_id=segment_id,
                segment_index=index,
                seq_id=index,
                speaker=speaker,
                start_seconds=start,
                end_seconds=end,
                text=f"Segment {index}",
                provenance=source,
            )
        )
        model_reliability = ReliabilityAssessment(
            status=ReliabilityStatus.USABLE
        )
        feature_reliability = ReliabilityAssessment(
            status=ReliabilityStatus.USABLE,
            quality_flags=feature_flags[index],
        )
        for name, value, reliability in (
            (
                "acoustic.escalation.score",
                escalation[index],
                model_reliability,
            ),
            (
                "acoustic.energy.rms_mean",
                energy[index],
                feature_reliability,
            ),
            (
                "acoustic.pitch.mean",
                pitch[index],
                feature_reliability,
            ),
        ):
            signals.append(
                SignalRecord(
                    signal_id=f"{segment_id}:{name}",
                    name=name,
                    modality=Modality.ACOUSTIC,
                    scope=SignalScope.SEGMENT,
                    value=value,
                    segment_id=segment_id,
                    seq_id=index,
                    speaker=speaker,
                    start_seconds=start,
                    end_seconds=end,
                    reliability=reliability,
                    provenance=source,
                    visibility=Visibility.INTERNAL,
                )
            )
    return SignalBundle(
        call_id="test-call",
        domain="banking",
        duration_seconds=float(size * 3),
        segments=segments,
        signals=signals,
        sources=[source],
    )


def _call_value(bundle: SignalBundle, name: str):
    return next(
        signal.value
        for signal in bundle.signals
        if signal.name == name
    )


class RobustNormalizationTests(unittest.TestCase):
    def test_shift_and_scale_do_not_change_robust_zscores(self):
        values = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0]
        transformed = [10.0 + value * 7.0 for value in values]

        self.assertEqual(
            _robust_zscores(values),
            _robust_zscores(transformed),
        )

    def test_speaker_baselines_are_normalized_independently(self):
        customer = _bundle([1, 2, 3, 4, 5, 8])
        agent = _bundle(
            [101, 102, 103, 104, 105, 108],
            speaker=Speaker.AGENT,
        )
        combined = customer.model_copy(deep=True)
        combined.segments.extend(agent.segments)
        for index, segment in enumerate(combined.segments):
            segment.segment_id = f"{segment.speaker.value}-{index:02d}"
            segment.segment_index = index
        combined.signals = []
        for source_bundle, offset in ((customer, 0), (agent, 6)):
            for signal_index, signal in enumerate(source_bundle.signals):
                copied = signal.model_copy(deep=True)
                copied.segment_id = combined.segments[
                    offset + signal_index // 3
                ].segment_id
                copied.signal_id = (
                    f"{copied.segment_id}:{copied.name}"
                )
                combined.signals.append(copied)

        output = derive_acoustic_dynamics(combined, TEST_CONFIG)
        escalation = [
            signal
            for signal in output.signals
            if signal.name
            == "acoustic.normalized.escalation.robust_z"
        ]
        by_speaker = {
            speaker: [
                signal.value
                for signal in escalation
                if signal.speaker == speaker
            ]
            for speaker in (Speaker.CUSTOMER, Speaker.AGENT)
        }

        self.assertEqual(
            by_speaker[Speaker.CUSTOMER],
            by_speaker[Speaker.AGENT],
        )


class EpisodeTests(unittest.TestCase):
    def test_isolated_peak_does_not_form_episode(self):
        output = derive_acoustic_dynamics(
            _bundle([0, 0, 0, 10, 0, 0, 0, 0, 0]),
            TEST_CONFIG,
        )

        self.assertEqual(output.episodes, [])
        self.assertEqual(
            _call_value(output, "acoustic.dynamics.episode_count"),
            0,
        )

    def test_adjacent_multisignal_elevation_forms_episode(self):
        values = [0, 0, 0, 8, 9, 0, 0, 0, 0]
        output = derive_acoustic_dynamics(
            _bundle(values),
            TEST_CONFIG,
        )

        self.assertEqual(len(output.episodes), 1)
        self.assertEqual(
            output.episodes[0].segment_ids,
            ["segment-03", "segment-04"],
        )

    def test_middle_episode_followed_by_calm_late_speech_is_recovery(self):
        values = [
            0.0,
            0.1,
            -0.1,
            8.0,
            9.0,
            0.2,
            -0.2,
            0.1,
            0.0,
            -0.1,
            0.2,
            0.0,
        ]
        output = derive_acoustic_dynamics(
            _bundle(values),
            TEST_CONFIG,
        )

        self.assertTrue(
            _call_value(
                output,
                "acoustic.dynamics.recovery_candidate",
            )
        )
        self.assertFalse(
            _call_value(
                output,
                "acoustic.dynamics.unresolved_end_candidate",
            )
        )

    def test_late_episode_is_unresolved_end_candidate(self):
        values = [0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0]
        output = derive_acoustic_dynamics(
            _bundle(values),
            TEST_CONFIG,
        )

        self.assertFalse(
            _call_value(
                output,
                "acoustic.dynamics.recovery_candidate",
            )
        )
        self.assertTrue(
            _call_value(
                output,
                "acoustic.dynamics.unresolved_end_candidate",
            )
        )


class ReliabilityTests(unittest.TestCase):
    def test_quality_flags_can_make_pitch_unavailable(self):
        values = [0, 1, 2, 3, 4, 5, 6, 7]
        flags = [["missing_pitch"] for _ in values]
        output = derive_acoustic_dynamics(
            _bundle(values, feature_flags=flags),
            TEST_CONFIG,
        )

        self.assertFalse(
            any(
                signal.name
                == "acoustic.normalized.pitch.robust_z"
                for signal in output.signals
            )
        )
        pitch_coverage = next(
            row
            for row in output.coverage
            if row.source
            == "v2_dynamics:customer:pitch"
        )
        self.assertEqual(pitch_coverage.usable_units, 0)
        self.assertIn(
            "insufficient_usable_observations",
            pitch_coverage.limitations,
        )

    def test_outputs_remain_internal_and_derivation_is_idempotent(self):
        values = [0, 0, 0, 8, 9, 0, 0, 0, 0]
        first = derive_acoustic_dynamics(
            _bundle(values),
            TEST_CONFIG,
        )
        second = derive_acoustic_dynamics(first, TEST_CONFIG)
        first_derived = [
            signal
            for signal in first.signals
            if signal.name.startswith(
                ("acoustic.normalized.", "acoustic.dynamics.")
            )
        ]
        second_derived = [
            signal
            for signal in second.signals
            if signal.name.startswith(
                ("acoustic.normalized.", "acoustic.dynamics.")
            )
        ]

        self.assertTrue(
            all(
                signal.visibility == Visibility.INTERNAL
                for signal in second_derived
            )
        )
        self.assertTrue(
            all(
                episode.visibility == Visibility.INTERNAL
                for episode in second.episodes
            )
        )
        self.assertEqual(
            [signal.signal_id for signal in first_derived],
            [signal.signal_id for signal in second_derived],
        )
        self.assertEqual(
            [episode.model_dump() for episode in first.episodes],
            [episode.model_dump() for episode in second.episodes],
        )


if __name__ == "__main__":
    unittest.main()
