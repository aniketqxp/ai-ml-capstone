"""
Wav2Vec2 emotion classifier for the capstone audio sentiment module.

This script fine-tunes Wav2Vec2 on CREMA-D for 6-class speech emotion
classification.

Run quick smoke test from ml-services:

    python -m src.models.test_wav2vec2_setup

Run a small training test:

    python -m src.models.wav2vec2_emotion_model --limit-per-split 40 --num-epochs 1

Run full training:

    python -m src.models.wav2vec2_emotion_model --num-epochs 5

Outputs:
    outputs/wav2vec2/
    outputs/reports/wav2vec2_emotion_report.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    set_seed,
)

from src.data.audio_dataset import (
    DEFAULT_METADATA_PATH,
    DEFAULT_SAMPLE_RATE,
    EMOTION_LABELS,
    build_label_encoding,
    get_split_dataframe,
    load_audio_file,
    load_metadata,
    resolve_audio_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DEFAULT_MODEL_CHECKPOINT = "facebook/wav2vec2-base"
DEFAULT_OUTPUT_DIR = ML_SERVICES_ROOT / "outputs" / "wav2vec2"
DEFAULT_REPORT_PATH = ML_SERVICES_ROOT / "outputs" / "reports" / "wav2vec2_emotion_report.json"
DEFAULT_CONFUSION_MATRIX_PATH = (
    ML_SERVICES_ROOT / "outputs" / "reports" / "wav2vec2_emotion_confusion_matrix.csv"
)

TASK_NAME = "6-class speech emotion classification"
RANDOM_SEED = 42


class Wav2Vec2EmotionDataset(Dataset):
    """
    PyTorch Dataset for Wav2Vec2 emotion classification.

    Each item returns raw audio values and a numeric label.
    Padding is handled later by Wav2Vec2DataCollator.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        label_to_id: Dict[str, int],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_seconds: Optional[float] = 6.0,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds

        required_columns = {"file_path", "emotion_label", "filename"}
        missing_columns = required_columns - set(self.metadata.columns)
        if missing_columns:
            raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict:
        row = self.metadata.iloc[index]

        audio_path = resolve_audio_path(row["file_path"])
        waveform, sample_rate = load_audio_file(
            audio_path=audio_path,
            target_sample_rate=self.sample_rate,
            max_duration_seconds=self.max_duration_seconds,
        )

        label_name = row["emotion_label"]
        label_id = self.label_to_id[label_name]

        return {
            "input_values": waveform,
            "labels": label_id,
            "filename": row["filename"],
        }


@dataclass
class Wav2Vec2DataCollator:
    """
    Pads variable-length audio inputs for Wav2Vec2 training.

    Audio clips do not all have the same length, so this collator pads each batch
    dynamically using the Wav2Vec2 processor.
    """

    processor: Wav2Vec2Processor
    sampling_rate: int = DEFAULT_SAMPLE_RATE

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Convert a list of dataset samples into one padded training batch.

        The processor expects a list of raw waveform arrays, not a list of
        dictionaries. Labels are added separately after padding.
        """
        input_values = [feature["input_values"] for feature in features]

        batch = self.processor(
            input_values,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(
            [feature["labels"] for feature in features],
            dtype=torch.long,
        )

        return batch

def limit_metadata_per_split(
    metadata: pd.DataFrame,
    limit_per_split: Optional[int],
) -> pd.DataFrame:
    """
    Limit records per split for quick testing.

    This is useful to verify training works before running full fine-tuning.
    """
    if limit_per_split is None:
        return metadata

    limited_parts = []
    for split_name in ["train", "validation", "test"]:
        split_df = metadata[metadata["split"] == split_name].head(limit_per_split)
        limited_parts.append(split_df)

    limited_metadata = pd.concat(limited_parts, ignore_index=True)

    print(f"Using limit_per_split={limit_per_split}")
    print(f"Limited split distribution: {limited_metadata['split'].value_counts().to_dict()}")

    return limited_metadata


def build_wav2vec2_datasets(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    limit_per_split: Optional[int] = None,
    max_duration_seconds: Optional[float] = 6.0,
) -> Dict[str, Wav2Vec2EmotionDataset]:
    """
    Build train, validation, and test datasets for Wav2Vec2.
    """
    metadata = load_metadata(metadata_path)
    metadata = limit_metadata_per_split(metadata, limit_per_split)

    label_encoding = build_label_encoding(task="emotion")

    train_df = get_split_dataframe(metadata, "train")
    validation_df = get_split_dataframe(metadata, "validation")
    test_df = get_split_dataframe(metadata, "test")

    return {
        "train": Wav2Vec2EmotionDataset(
            metadata=train_df,
            label_to_id=label_encoding.label_to_id,
            max_duration_seconds=max_duration_seconds,
        ),
        "validation": Wav2Vec2EmotionDataset(
            metadata=validation_df,
            label_to_id=label_encoding.label_to_id,
            max_duration_seconds=max_duration_seconds,
        ),
        "test": Wav2Vec2EmotionDataset(
            metadata=test_df,
            label_to_id=label_encoding.label_to_id,
            max_duration_seconds=max_duration_seconds,
        ),
    }


def build_model_and_processor(
    model_checkpoint: str = DEFAULT_MODEL_CHECKPOINT,
) -> tuple[Wav2Vec2ForSequenceClassification, Wav2Vec2Processor]:
    """
    Load Wav2Vec2 model and processor for 6-class classification.
    """
    label_encoding = build_label_encoding(task="emotion")

    processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(EMOTION_LABELS),
        label2id=label_encoding.label_to_id,
        id2label=label_encoding.id_to_label,
    )

    # This is safer for laptops and speeds up training.
    # Later we can unfreeze for stronger fine-tuning.
    model.freeze_feature_encoder()

    return model, processor


def compute_metrics(eval_prediction) -> Dict[str, float]:
    """
    Compute evaluation metrics during training.
    """
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
    }


def get_device_note() -> str:
    """
    Return a readable note about available acceleration.
    """
    if torch.cuda.is_available():
        return "CUDA GPU available"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple Silicon MPS available"

    return "CPU only"


def create_training_arguments(
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingArguments:
    """
    Create Hugging Face training arguments.

    Uses eval_strategy because newer Transformers versions replaced the older
    evaluation_strategy argument.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )


def evaluate_on_test_set(
    trainer: Trainer,
    test_dataset: Dataset,
    id_to_label: Dict[int, str],
    confusion_matrix_path: Path,
) -> Dict:
    """
    Evaluate the final model on the held-out test set.
    """
    predictions_output = trainer.predict(test_dataset)

    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = np.argmax(logits, axis=-1)

    target_names = [id_to_label[index] for index in sorted(id_to_label.keys())]

    report = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(labels, predictions, labels=sorted(id_to_label.keys()))
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in target_names],
        columns=[f"predicted_{label}" for label in target_names],
    )

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(confusion_matrix_path)

    return report


def train_wav2vec2_emotion_model(
    model_checkpoint: str = DEFAULT_MODEL_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    confusion_matrix_path: Path = DEFAULT_CONFUSION_MATRIX_PATH,
    limit_per_split: Optional[int] = None,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    max_duration_seconds: Optional[float] = 6.0,
) -> Dict:
    """
    Fine-tune Wav2Vec2 for emotion classification.
    """
    set_seed(RANDOM_SEED)

    print("\nStarting Wav2Vec2 emotion fine-tuning")
    print("-" * 70)
    print(f"Device: {get_device_note()}")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 70)

    datasets = build_wav2vec2_datasets(
        limit_per_split=limit_per_split,
        max_duration_seconds=max_duration_seconds,
    )

    label_encoding = build_label_encoding(task="emotion")
    model, processor = build_model_and_processor(model_checkpoint)

    data_collator = Wav2Vec2DataCollator(processor=processor)

    training_args = create_training_arguments(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_metrics = trainer.evaluate(datasets["validation"])
    test_report = evaluate_on_test_set(
        trainer=trainer,
        test_dataset=datasets["test"],
        id_to_label=label_encoding.id_to_label,
        confusion_matrix_path=confusion_matrix_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "best_model"))
    processor.save_pretrained(str(output_dir / "best_model"))

    full_report = {
        "model_name": "Wav2Vec2 emotion classifier",
        "base_checkpoint": model_checkpoint,
        "task": TASK_NAME,
        "device": get_device_note(),
        "labels": label_encoding.id_to_label,
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "test_samples": len(datasets["test"]),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "validation": validation_metrics,
        "test": test_report,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(full_report, file, indent=2)

    print("\nWav2Vec2 Results")
    print("-" * 70)
    print(f"Validation accuracy: {validation_metrics.get('eval_accuracy'):.4f}")
    print(f"Validation macro F1: {validation_metrics.get('eval_macro_f1'):.4f}")
    print(f"Test accuracy: {test_report['accuracy']:.4f}")
    print(f"Test macro F1: {test_report['macro_f1']:.4f}")
    print("-" * 70)
    print(f"Saved model to: {output_dir / 'best_model'}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")

    return full_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2 for CREMA-D emotion classification."
    )

    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help="Hugging Face Wav2Vec2 checkpoint.",
    )

    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional sample limit per split for quick testing.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )

    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=6.0,
        help="Maximum audio duration per sample.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_wav2vec2_emotion_model(
        model_checkpoint=args.model_checkpoint,
        limit_per_split=args.limit_per_split,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_duration_seconds=args.max_duration_seconds,
    )