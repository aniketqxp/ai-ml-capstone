"""
Batch test for the trained Wav2Vec2 emotion predictor.

Run from ml-services:

    python -m src.test_emotion_predictor_batch
"""

from pathlib import Path

from src.inference.emotion_predictor import EmotionPredictor


TEST_FILES = [
    ("anger", "data/raw/cremad/AudioWAV/1002_DFA_ANG_XX.wav"),
    ("sadness", "data/raw/cremad/AudioWAV/1002_DFA_SAD_XX.wav"),
    ("neutral", "data/raw/cremad/AudioWAV/1002_DFA_NEU_XX.wav"),
    ("happy", "data/raw/cremad/AudioWAV/1002_DFA_HAP_XX.wav"),
    ("fear", "data/raw/cremad/AudioWAV/1002_DFA_FEA_XX.wav"),
    ("disgust", "data/raw/cremad/AudioWAV/1002_DFA_DIS_XX.wav"),
    ("neutral", "data/raw/cremad/AudioWAV/1003_DFA_NEU_XX.wav"),
    ("neutral", "data/raw/cremad/AudioWAV/1004_DFA_NEU_XX.wav"),
]


def main() -> None:
    predictor = EmotionPredictor()

    print("\nBatch Emotion Predictor Test")
    print("-" * 80)

    for expected_label, file_path in TEST_FILES:
        result = predictor.analyze_audio(
            audio_path=Path(file_path),
            call_id=f"TEST_{expected_label.upper()}",
        )

        output = result.to_api_response()

        print(f"File: {Path(file_path).name}")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {output['dominant_emotion']}")
        print(f"Overall sentiment: {output['overall_audio_sentiment']}")
        print(f"Anger: {output['anger_probability']:.3f}")
        print(f"Sadness: {output['sadness_probability']:.3f}")
        print(f"Fear/Anxiety: {output['anxiety_probability']:.3f}")
        print(f"Calm: {output['calm_probability']:.3f}")
        audio_features = output["audio_features"]

        print(f"Escalation score: {output['audio_escalation_score']:.3f}")
        print(f"Risk level: {output['risk_level']}")
        print(f"Vocal intensity: {audio_features['vocal_intensity']}")
        print(f"Pitch level: {audio_features['pitch_level']}")
        print(f"Pitch variability: {audio_features['pitch_variability']}")
        print(f"Speech rate: {audio_features['speech_rate']}")
        print(f"Pause frequency: {audio_features['pause_frequency']}")
        print(f"Long silence detected: {audio_features['long_silence_detected']}")
        print(
            f"Total silence duration: "
            f"{audio_features.get('total_silence_duration_seconds', 0)} sec"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()