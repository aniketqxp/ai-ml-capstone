"""
Quick validation script for the baseline emotion model pipeline.

Run from ml-services:

    python -m src.models.test_baseline_model

This uses a small subset from each split to confirm the full baseline pipeline
works before running the full feature extraction and training.
"""

from src.models.baseline_emotion_model import train_baseline_model


def main() -> None:
    """Run a quick baseline training test."""
    train_baseline_model(
        force_rebuild_features=True,
        limit_per_split=50,
    )


if __name__ == "__main__":
    main()