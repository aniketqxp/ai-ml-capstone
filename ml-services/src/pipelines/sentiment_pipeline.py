"""
Clean sentiment pipeline service for backend integration.

This module provides a simple, stable entry point for the audio sentiment system.
Backend code should call this pipeline instead of directly using test scripts or
low-level model classes.

Main function:
    analyze_audio_sentiment(call_id, audio_path)

Example:
    result = analyze_audio_sentiment(
        call_id="CALL_001",
        audio_path="data/raw/cremad/AudioWAV/1002_DFA_ANG_XX.wav",
    )

    print(result)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from src.inference.emotion_predictor import EmotionPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_OUTPUT_DIR = ML_SERVICES_ROOT / "outputs" / "sentiment"


class SentimentPipeline:
    """
    High-level sentiment analysis pipeline.

    This class hides the internal model/prediction details and exposes one clean
    interface for backend integration.
    """

    def __init__(
        self,
        predictor: Optional[EmotionPredictor] = None,
        save_outputs: bool = False,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        """
        Initialize the sentiment pipeline.

        Args:
            predictor:
                Optional existing EmotionPredictor instance.
                If not provided, one will be created.
            save_outputs:
                Whether to save JSON outputs to disk.
            output_dir:
                Directory where JSON outputs will be saved.
        """
        self.predictor = predictor or EmotionPredictor()
        self.save_outputs = save_outputs
        self.output_dir = Path(output_dir)

        if self.save_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        call_id: str,
        audio_path: Union[str, Path],
        build_timeline: bool = True,
    ) -> Dict:
        """
        Analyze one call/audio file and return JSON-ready sentiment output.

        Args:
            call_id:
                Unique identifier for the call.
            audio_path:
                Path to the audio file.
            build_timeline:
                Whether to split audio into segments and build sentiment timeline.

        Returns:
            Dictionary matching the AudioSentimentResult schema.
        """
        audio_path = Path(audio_path)

        result = self.predictor.analyze_audio(
            audio_path=audio_path,
            call_id=call_id,
            build_timeline=build_timeline,
        )

        response = result.to_api_response()

        if self.save_outputs:
            self._save_result(call_id=call_id, result=response)

        return response

    def _save_result(self, call_id: str, result: Dict) -> Path:
        """
        Save one sentiment result as JSON.

        Args:
            call_id:
                Call identifier.
            result:
                JSON-ready sentiment result.

        Returns:
            Path to saved JSON file.
        """
        safe_call_id = call_id.replace("/", "_").replace("\\", "_").strip()
        output_path = self.output_dir / f"{safe_call_id}_sentiment.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        return output_path


def analyze_audio_sentiment(
    call_id: str,
    audio_path: Union[str, Path],
    build_timeline: bool = True,
    save_output: bool = False,
) -> Dict:
    """
    Convenience function for backend/API usage.

    This is the simplest function for other parts of the project to call.

    Args:
        call_id:
            Unique call identifier.
        audio_path:
            Path to audio file.
        build_timeline:
            Whether to generate segment-level sentiment timeline.
        save_output:
            Whether to save result JSON under outputs/sentiment.

    Returns:
        JSON-ready dictionary.
    """
    pipeline = SentimentPipeline(save_outputs=save_output)

    return pipeline.analyze(
        call_id=call_id,
        audio_path=audio_path,
        build_timeline=build_timeline,
    )