"""
Smoke test for the Wav2Vec2 emotion classifier setup.

Run from ml-services:

    python -m src.models.test_wav2vec2_setup

This checks:
    1. Metadata loads correctly
    2. Wav2Vec2 datasets can be built
    3. Processor loads
    4. Model loads
    5. One batch can be collated
"""

from torch.utils.data import DataLoader

from src.models.wav2vec2_emotion_model import (
    Wav2Vec2DataCollator,
    build_model_and_processor,
    build_wav2vec2_datasets,
)


def main() -> None:
    print("\nRunning Wav2Vec2 setup smoke test...")

    datasets = build_wav2vec2_datasets(
        limit_per_split=5,
        max_duration_seconds=6.0,
    )

    print(f"Train samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['validation'])}")
    print(f"Test samples: {len(datasets['test'])}")

    model, processor = build_model_and_processor()
    print(f"Loaded model type: {type(model).__name__}")

    data_collator = Wav2Vec2DataCollator(processor=processor)

    dataloader = DataLoader(
        datasets["train"],
        batch_size=2,
        collate_fn=data_collator,
    )

    batch = next(iter(dataloader))

    print("Batch keys:", list(batch.keys()))
    print("input_values shape:", tuple(batch["input_values"].shape))
    print("labels shape:", tuple(batch["labels"].shape))
    print("labels:", batch["labels"].tolist())

    print("\nWav2Vec2 setup smoke test completed successfully.\n")


if __name__ == "__main__":
    main()