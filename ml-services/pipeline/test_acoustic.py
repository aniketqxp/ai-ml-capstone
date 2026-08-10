from types import SimpleNamespace

from acoustic import _audio_features, _feature_summary, _three_class_sentiment


def test_three_class_sentiment_falls_back_to_emotion():
    result = SimpleNamespace(
        overall_audio_sentiment="Unknown",
        dominant_emotion="anger",
    )

    assert _three_class_sentiment(result) == "negative"


def test_audio_features_include_demo_metrics():
    raw = SimpleNamespace(
        duration_seconds=4.0,
        pitch_mean=140.0,
        pitch_std=12.0,
        rms_mean=0.1,
        rms_std=0.02,
        total_silence_duration_seconds=1.0,
        longest_silence_seconds=0.6,
        pause_count=2,
    )

    features = _audio_features(raw, {}, "one two three four five six")
    summary = _feature_summary(
        [{"audio_features": features, "processing_status": "success"}]
    )

    assert features["speech_rate_words_per_minute"] == 90.0
    assert features["pause_ratio"] == 0.25
    assert features["pitch_mean_hz"] == 140.0
    assert summary["average_volume_db"] == -20.0
