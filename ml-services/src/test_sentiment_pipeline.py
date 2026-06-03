"""
Test script for the clean sentiment pipeline service.

Run from ml-services:

    python -m src.test_sentiment_pipeline
"""

import json
from pathlib import Path

from src.pipelines.sentiment_pipeline import analyze_audio_sentiment


def main() -> None:
    """
    Test the high-level sentiment pipeline on one audio sample.
    """
    result = analyze_audio_sentiment(
        call_id="CALL_PIPELINE_TEST_001",
        audio_path=Path("data/raw/cremad/AudioWAV/1002_DFA_ANG_XX.wav"),
        build_timeline=True,
        save_output=True,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()