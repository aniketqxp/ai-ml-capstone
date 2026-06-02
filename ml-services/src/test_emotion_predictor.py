"""
Test script for the trained Wav2Vec2 emotion predictor.

Run from ml-services:

    python -m src.test_emotion_predictor
"""

import json
from pathlib import Path

from src.inference.emotion_predictor import EmotionPredictor


def run_prediction(audio_path: Path, call_id: str) -> None:
    """
    Run prediction on one audio file and print the sentiment JSON.
    """
    predictor = EmotionPredictor()

    result = predictor.analyze_audio(
        audio_path=audio_path,
        call_id=call_id,
    )

    print(json.dumps(result.to_api_response(), indent=2))


def main() -> None:
    """
    Test prediction on one CREMA-D audio file.
    """
    test_audio_path = Path("data/raw/cremad/AudioWAV/1002_DFA_NEU_XX.wav")

    run_prediction(
        audio_path=test_audio_path,
        call_id="CALL_TEST_001",
    )


if __name__ == "__main__":
    main()