"""
Validation script for the CREMA-D PyTorch audio dataset.

Run from ml-services:

    python -m src.data.test_audio_dataset

This confirms:
    1. Metadata loads correctly
    2. Train/validation/test datasets are created
    3. Audio files can be loaded
    4. Labels are converted to numeric IDs
    5. Waveforms are returned as PyTorch tensors
"""

import json

from src.data.audio_dataset import (
    DEFAULT_SAMPLE_RATE,
    build_cremad_datasets,
    summarize_dataset,
)


def inspect_sample(sample: dict) -> dict:
    """
    Convert one dataset sample into printable debug information.
    """
    waveform = sample["input_values"]

    return {
        "filename": sample["filename"],
        "actor_id": sample["actor_id"],
        "split": sample["split"],
        "label_id": int(sample["label"]),
        "label_name": sample["label_name"],
        "sample_rate": sample["sample_rate"],
        "waveform_shape": list(waveform.shape),
        "duration_seconds": round(waveform.shape[0] / sample["sample_rate"], 3),
        "tensor_dtype": str(waveform.dtype),
    }


def main() -> None:
    """Run dataset validation checks."""
    print("\nLoading CREMA-D metadata summary...")
    summary = summarize_dataset()
    print(json.dumps(summary, indent=2))

    print("\nBuilding emotion classification datasets...")
    emotion_datasets = build_cremad_datasets(
        task="emotion",
        target_sample_rate=DEFAULT_SAMPLE_RATE,
        max_duration_seconds=6.0,
    )

    for split_name, dataset in emotion_datasets.items():
        print(f"{split_name}: {len(dataset)} samples")

    print("\nInspecting one training sample...")
    train_sample = emotion_datasets["train"][0]
    print(json.dumps(inspect_sample(train_sample), indent=2))

    print("\nBuilding sentiment classification datasets...")
    sentiment_datasets = build_cremad_datasets(
        task="sentiment",
        target_sample_rate=DEFAULT_SAMPLE_RATE,
        max_duration_seconds=6.0,
    )

    for split_name, dataset in sentiment_datasets.items():
        print(f"{split_name}: {len(dataset)} samples")

    print("\nInspecting one sentiment training sample...")
    sentiment_sample = sentiment_datasets["train"][0]
    print(json.dumps(inspect_sample(sentiment_sample), indent=2))

    print("\nAudio dataset validation completed successfully.\n")


if __name__ == "__main__":
    main()