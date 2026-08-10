import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sentiment_payloads import (
    customer_escalation_trend,
    summarize_audio_series,
    summarize_speakers,
)


SERIES = [
    {
        "speaker": "CUSTOMER",
        "sentiment": "Negative",
        "dominant_emotion": "sadness",
        "escalation_score": 0.1,
        "pitch_mean_hz": 180.0,
        "volume_db_mean": -18.0,
        "pause_ratio": 0.2,
        "speech_rate_words_per_minute": 120.0,
    },
    {
        "speaker": "AGENT",
        "sentiment": "Positive",
        "dominant_emotion": "happy",
        "escalation_score": 0.2,
        "pitch_mean_hz": 220.0,
        "volume_db_mean": -14.0,
        "pause_ratio": 0.1,
        "speech_rate_words_per_minute": 180.0,
    },
    {
        "speaker": "CUSTOMER",
        "sentiment": "Negative",
        "dominant_emotion": "sadness",
        "escalation_score": 0.4,
        "pitch_mean_hz": 200.0,
        "volume_db_mean": -16.0,
        "pause_ratio": 0.3,
        "speech_rate_words_per_minute": 160.0,
    },
]


def test_audio_series_uses_clara_metric_names():
    summary = summarize_audio_series(SERIES)
    assert summary["average_pitch_hz"] == 200.0
    assert summary["average_volume_db"] == -16.0
    assert summary["average_pause_ratio"] == 0.2
    assert summary["average_speech_rate_wpm"] == 153.3333


def test_speaker_summary_and_customer_trend_are_separate():
    speakers = summarize_speakers(SERIES)
    assert speakers["customer"]["dominant_sentiment"] == "Negative"
    assert speakers["agent"]["dominant_emotion"] == "happy"
    assert customer_escalation_trend(SERIES) == {}
