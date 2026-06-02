"""
Validation test for Step 1 of the sentiment analysis module.

Run from the project root:

    cd ml-services/src
    python test_sentiment_schema.py

This does not train or run a model yet. It only confirms that the sentiment
JSON contract works correctly.
"""

import json

from sentiment_config import IntensityLevel, SentimentShift
from sentiment_schema import (
    AudioFeatureSummary,
    AudioSentimentResult,
    EmotionProbabilities,
    PeakEmotion,
    SentimentSegment,
    infer_overall_sentiment,
    infer_risk_level,
    seconds_to_timestamp,
)


def build_sample_sentiment_result() -> AudioSentimentResult:
    """Build a sample output that matches the final capstone contract."""

    probabilities = EmotionProbabilities(
        anger=0.71,
        sadness=0.22,
        fear=0.30,
        disgust=0.10,
        happy=0.04,
        neutral=0.18,
    )

    overall_sentiment = infer_overall_sentiment(probabilities)
    dominant_emotion = probabilities.dominant_emotion()
    negative_probability = probabilities.negative_probability()
    stress_probability = probabilities.stress_probability()

    escalation_score = 0.86

    segment = SentimentSegment(
        segment_id=0,
        start_time_seconds=180.0,
        end_time_seconds=190.0,
        dominant_emotion=dominant_emotion,
        overall_audio_sentiment=overall_sentiment,
        emotion_probabilities=probabilities,
        risk_score=escalation_score,
    )

    result = AudioSentimentResult(
        call_id="CALL_001",
        overall_audio_sentiment=overall_sentiment,
        dominant_emotion=dominant_emotion,
        negative_emotion_probability=negative_probability,
        anger_probability=probabilities.anger,
        stress_probability=stress_probability,
        sadness_probability=probabilities.sadness,
        anxiety_probability=probabilities.fear,
        calm_probability=probabilities.calm_probability(),
        audio_features=AudioFeatureSummary(
            vocal_intensity=IntensityLevel.HIGH,
            pitch_level=IntensityLevel.HIGH,
            pitch_variability=IntensityLevel.HIGH,
            speech_rate=IntensityLevel.HIGH,
            pause_frequency=IntensityLevel.MEDIUM,
            long_silence_detected=False,
            total_silence_duration_seconds=4.2,
            overlap_rate=0.19,
        ),
        emotional_volatility=IntensityLevel.HIGH,
        audio_sentiment_shift=SentimentShift.WORSENED,
        audio_escalation_score=escalation_score,
        risk_level=infer_risk_level(escalation_score),
        peak_emotion=PeakEmotion(
            time_seconds=192.0,
            timestamp=seconds_to_timestamp(192.0),
            emotion=dominant_emotion,
            score=escalation_score,
        ),
        sentiment_timeline=[segment],
        model_name="schema-test-placeholder",
        model_version="0.1.0",
        processing_status="success",
        warnings=[
            "This is only a schema validation example. No model inference has been run yet."
        ],
    )

    return result


if __name__ == "__main__":
    sample_result = build_sample_sentiment_result()
    print(json.dumps(sample_result.to_api_response(), indent=2))