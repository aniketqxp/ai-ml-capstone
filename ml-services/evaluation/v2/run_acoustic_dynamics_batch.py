"""Run evaluator-v2 acoustic dynamics over Clara's enriched call artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .acoustic_dynamics import (
    DEFAULT_CONFIG,
    DYNAMICS_VERSION,
    DynamicsConfig,
    derive_acoustic_dynamics,
)
from .adapters import build_signal_bundle
from .schemas import SignalBundle, SignalScope

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTIMENT_PREFIX = (
    "ml-services/outputs/backend/sentiment_calls_with_features"
)
TRANSCRIPT_PREFIX = "data/sentence_segments"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    )


def _git_json(ref: str, path: str) -> dict[str, Any]:
    return json.loads(_git("show", f"{ref}:{path}"))


def _sentiment_paths(ref: str, domains: set[str]) -> list[str]:
    paths = _git(
        "ls-tree",
        "-r",
        "--name-only",
        ref,
        SENTIMENT_PREFIX,
    ).splitlines()
    return [
        path
        for path in paths
        if path.endswith(".json")
        and (not domains or Path(path).parent.name in domains)
    ]


def _transcript(
    ref: str,
    domain: str,
    call_id: str,
) -> tuple[dict[str, Any], str]:
    relative = f"{TRANSCRIPT_PREFIX}/{domain}/{call_id}.json"
    return _git_json(ref, relative), f"{ref}:{relative}"


def _call_signal_values(bundle: SignalBundle) -> dict[str, Any]:
    return {
        signal.name.removeprefix("acoustic.dynamics."): signal.value
        for signal in bundle.signals
        if signal.scope == SignalScope.CALL
        and signal.name.startswith("acoustic.dynamics.")
    }


def _feature_availability(bundle: SignalBundle) -> dict[str, str]:
    output = {}
    for feature in (
        "escalation",
        "pitch",
        "volume",
        "energy",
        "pause",
        "speech_rate",
    ):
        normalized = [
            signal
            for signal in bundle.signals
            if signal.name
            == f"acoustic.normalized.{feature}.robust_z"
            and signal.speaker
            and signal.speaker.value == "customer"
        ]
        if not normalized:
            output[feature] = "unavailable"
        else:
            output[feature] = normalized[0].reliability.status.value
    return output


def _call_summary(bundle: SignalBundle) -> dict[str, Any]:
    call_signals = _call_signal_values(bundle)
    return {
        "call_id": bundle.call_id,
        "domain": bundle.domain,
        "feature_availability": _feature_availability(bundle),
        "episode_count": len(bundle.episodes),
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "start_seconds": episode.start_seconds,
                "end_seconds": episode.end_seconds,
                "segment_ids": episode.segment_ids,
            }
            for episode in bundle.episodes
        ],
        "usable_customer_seconds": call_signals[
            "usable_customer_seconds"
        ],
        "early_elevated_ratio": call_signals[
            "early_elevated_ratio"
        ],
        "late_elevated_ratio": call_signals[
            "late_elevated_ratio"
        ],
        "trajectory_direction": call_signals[
            "trajectory_direction"
        ],
        "trajectory_delta": call_signals.get("trajectory_delta"),
        "recovery_candidate": call_signals["recovery_candidate"],
        "unresolved_end_candidate": call_signals[
            "unresolved_end_candidate"
        ],
    }


def build_batch_summary(
    sentiment_ref: str,
    transcript_ref: str,
    domains: set[str],
    config: DynamicsConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    calls = []
    for path in _sentiment_paths(sentiment_ref, domains):
        sentiment = _git_json(sentiment_ref, path)
        domain = Path(path).parent.name
        call_id = str(sentiment["call_id"])
        transcript, transcript_source = _transcript(
            transcript_ref,
            domain,
            call_id,
        )
        bundle = build_signal_bundle(
            transcript,
            sentiment,
            transcript_source=transcript_source,
            sentiment_source=f"{sentiment_ref}:{path}",
        )
        calls.append(
            _call_summary(
                derive_acoustic_dynamics(bundle, config)
            )
        )

    episode_distribution = Counter(
        call["episode_count"]
        for call in calls
    )
    trajectory_distribution = Counter(
        call["trajectory_direction"]
        for call in calls
    )
    feature_availability = {
        feature: dict(
            sorted(
                Counter(
                    call["feature_availability"][feature]
                    for call in calls
                ).items()
            )
        )
        for feature in (
            "escalation",
            "pitch",
            "volume",
            "energy",
            "pause",
            "speech_rate",
        )
    }
    return {
        "schema_version": "1.0",
        "dynamics_version": DYNAMICS_VERSION,
        "sentiment_ref": sentiment_ref,
        "sentiment_commit": _git("rev-parse", sentiment_ref).strip(),
        "transcript_ref": transcript_ref,
        "transcript_commit": _git("rev-parse", transcript_ref).strip(),
        "config": asdict(config),
        "aggregate": {
            "call_count": len(calls),
            "episode_count_distribution": {
                str(key): value
                for key, value in sorted(episode_distribution.items())
            },
            "calls_with_recovery_candidate": sum(
                bool(call["recovery_candidate"])
                for call in calls
            ),
            "calls_with_unresolved_end_candidate": sum(
                bool(call["unresolved_end_candidate"])
                for call in calls
            ),
            "trajectory_direction_distribution": dict(
                sorted(trajectory_distribution.items())
            ),
            "customer_feature_availability": feature_availability,
        },
        "calls": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentiment-ref",
        default="origin/integration/full-pipeline-test",
    )
    parser.add_argument(
        "--transcript-ref",
        default="HEAD",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Restrict the batch to one or more domain directory names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON summary to this path.",
    )
    args = parser.parse_args()
    summary = build_batch_summary(
        args.sentiment_ref,
        args.transcript_ref,
        set(args.domain),
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        output = (
            args.output
            if args.output.is_absolute()
            else REPO_ROOT / args.output
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
