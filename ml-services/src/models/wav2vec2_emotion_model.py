"""
Wav2Vec2 emotion classifier for the capstone audio sentiment module.

This script fine-tunes Wav2Vec2 for 6-class speech emotion classification.

Default training uses CREMA-D metadata. For Model V2, pass the combined
CREMA-D + RAVDESS metadata file.

Run quick smoke test from ml-services:

    python -m src.models.test_wav2vec2_setup

Run CREMA-D training:

    python -m src.models.wav2vec2_emotion_model \
      --model-checkpoint Dpngtm/wav2vec2-emotion-recognition \
      --run-name model_v1_cremad \
      --num-epochs 3 \
      --batch-size 2 \
      --learning-rate 1e-5

Run combined CREMA-D + RAVDESS training:

    python -m src.models.wav2vec2_emotion_model \
      --metadata-path data/processed/combined_emotion_metadata.csv \
      --model-checkpoint Dpngtm/wav2vec2-emotion-recognition \
      --run-name model_v2_cremad_ravdess \
      --num-epochs 3 \
      --batch-size 2 \
      --learning-rate 1e-5

Outputs:
    outputs/wav2vec2/<run-name>/best_model
    outputs/reports/<run-name>_report.json
    outputs/reports/<run-name>_confusion_matrix.csv
"""

import argparse
import json
import mlflow
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
DEFAULT_METADATA_PATH = ML_SERVICES_ROOT / "data" / "processed" / "cremad_metadata.csv"
DEFAULT_OUTPUT_ROOT = ML_SERVICES_ROOT / "outputs" / "wav2vec2"
DEFAULT_REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "audio_sentiment_emotion_classification"
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{ML_SERVICES_ROOT / 'mlflow.db'}"

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

def infer_dataset_source(metadata: pd.DataFrame) -> str:
    """
    Infer dataset source from metadata.

    If the metadata has a dataset column, return the joined dataset names.
    Otherwise, assume CREMA-D for backwards compatibility.
    """
    if "dataset" not in metadata.columns:
        return "CREMA-D"

    datasets = sorted(metadata["dataset"].dropna().unique().tolist())

    if not datasets:
        return "Unknown"

    return " + ".join(datasets)


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
    freeze_feature_encoder: bool = True,
    freeze_transformer_layers: int = 0,
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
        ignore_mismatched_sizes=True,
    )

    # This is safer for laptops and speeds up training.
    # Later we can unfreeze for stronger fine-tuning.
    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if freeze_transformer_layers > 0:
        encoder_layers = model.wav2vec2.encoder.layers
        total_layers = len(encoder_layers)

        if freeze_transformer_layers > total_layers:
            raise ValueError(
                f"freeze_transformer_layers={freeze_transformer_layers} is larger "
                f"than total transformer layers={total_layers}"
            )

        for layer_index in range(freeze_transformer_layers):
            for parameter in encoder_layers[layer_index].parameters():
                parameter.requires_grad = False

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

def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total model parameters.
    """
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "frozen_parameters": int(total_parameters - trainable_parameters),
    }


def create_training_arguments(
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
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
        warmup_ratio=warmup_ratio,
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

    label_ids = sorted(id_to_label.keys())

    report = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=label_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in target_names],
        columns=[f"predicted_{label}" for label in target_names],
    )

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(confusion_matrix_path)

    return report

def log_training_run_to_mlflow(
    full_report: Dict,
    report_path: Path,
    confusion_matrix_path: Path,
    run_name: str,
    mlflow_experiment_name: str,
) -> str:
    """
    Log Wav2Vec2 training results to MLflow.

    This tracks model configuration, dataset information, validation/test metrics,
    and report artifacts for experiment comparison.
    """
    mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("module", "audio_sentiment_analysis")
        mlflow.set_tag("model_family", "Wav2Vec2")
        mlflow.set_tag("training_framework", "Hugging Face Transformers")
        mlflow.set_tag("dataset_source", full_report.get("dataset_source", "Unknown"))
        mlflow.set_tag("purpose", "model_training")

        # Parameters
        mlflow.log_param("run_name", full_report.get("run_name"))
        mlflow.log_param("model_name", full_report.get("model_name"))
        mlflow.log_param("base_checkpoint", full_report.get("base_checkpoint"))
        mlflow.log_param("task", full_report.get("task"))
        mlflow.log_param("device", full_report.get("device"))
        mlflow.log_param("dataset_source", full_report.get("dataset_source"))
        mlflow.log_param("metadata_path", full_report.get("metadata_path"))
        mlflow.log_param("output_dir", full_report.get("output_dir"))
        mlflow.log_param("best_model_dir", full_report.get("best_model_dir"))
        mlflow.log_param("train_samples", full_report.get("train_samples"))
        mlflow.log_param("validation_samples", full_report.get("validation_samples"))
        mlflow.log_param("test_samples", full_report.get("test_samples"))
        mlflow.log_param("num_epochs", full_report.get("num_epochs"))
        mlflow.log_param("batch_size", full_report.get("batch_size"))
        mlflow.log_param("learning_rate", full_report.get("learning_rate"))
        mlflow.log_param("weight_decay", full_report.get("weight_decay"))
        mlflow.log_param("warmup_ratio", full_report.get("warmup_ratio"))
        mlflow.log_param(
            "freeze_feature_encoder",
            full_report.get("freeze_feature_encoder"),
        )
        mlflow.log_param(
            "freeze_transformer_layers",
            full_report.get("freeze_transformer_layers"),
        )
        mlflow.log_param("total_parameters", full_report.get("total_parameters"))
        mlflow.log_param("trainable_parameters", full_report.get("trainable_parameters"))
        mlflow.log_param("frozen_parameters", full_report.get("frozen_parameters"))

        # Validation metrics
        validation_metrics = full_report.get("validation", {})
        mlflow.log_metric(
            "validation_accuracy",
            float(validation_metrics.get("eval_accuracy", 0.0)),
        )
        mlflow.log_metric(
            "validation_macro_f1",
            float(validation_metrics.get("eval_macro_f1", 0.0)),
        )
        mlflow.log_metric(
            "validation_weighted_f1",
            float(validation_metrics.get("eval_weighted_f1", 0.0)),
        )
        mlflow.log_metric(
            "validation_loss",
            float(validation_metrics.get("eval_loss", 0.0)),
        )

        # Test metrics
        test_metrics = full_report.get("test", {})
        mlflow.log_metric("test_accuracy", float(test_metrics.get("accuracy", 0.0)))
        mlflow.log_metric("test_macro_f1", float(test_metrics.get("macro_f1", 0.0)))
        mlflow.log_metric(
            "test_weighted_f1",
            float(test_metrics.get("weighted_f1", 0.0)),
        )

        # Artifacts
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="reports")

        if confusion_matrix_path.exists():
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="reports")

        run_id = mlflow.active_run().info.run_id

    return run_id

def train_wav2vec2_emotion_model(
    model_checkpoint: str = DEFAULT_MODEL_CHECKPOINT,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    run_name: str = "wav2vec2_emotion",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    limit_per_split: Optional[int] = None,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    freeze_feature_encoder: bool = True,
    freeze_transformer_layers: int = 0,
    max_duration_seconds: Optional[float] = 6.0,
    enable_mlflow: bool = False,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
) -> Dict:
    """
    Fine-tune Wav2Vec2 for emotion classification.
    """
    set_seed(RANDOM_SEED)
    metadata_path = Path(metadata_path)
    output_root = Path(output_root)

    output_dir = output_root / run_name
    best_model_dir = output_dir / "best_model"
    report_path = DEFAULT_REPORTS_DIR / f"{run_name}_report.json"
    confusion_matrix_path = DEFAULT_REPORTS_DIR / f"{run_name}_confusion_matrix.csv"

    metadata = load_metadata(metadata_path)
    dataset_source = infer_dataset_source(metadata)

    print("\nStarting Wav2Vec2 emotion fine-tuning")
    print("-" * 70)
    print(f"Run name: {run_name}")
    print(f"Device: {get_device_note()}")
    print(f"Dataset source: {dataset_source}")
    print(f"Metadata path: {metadata_path}")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup ratio: {warmup_ratio}")
    print(f"Output directory: {output_dir}")
    print(f"MLflow enabled: {enable_mlflow}")
    if enable_mlflow:
        print(f"MLflow experiment: {mlflow_experiment_name}")
    print("-" * 70)

    datasets = build_wav2vec2_datasets(
        metadata_path=metadata_path,
        limit_per_split=limit_per_split,
        max_duration_seconds=max_duration_seconds,
    )

    label_encoding = build_label_encoding(task="emotion")
    model, processor = build_model_and_processor(
        model_checkpoint=model_checkpoint,
        freeze_feature_encoder=freeze_feature_encoder,
        freeze_transformer_layers=freeze_transformer_layers,
    )
    parameter_counts = count_trainable_parameters(model)
    print("Parameter counts:")
    print(f"  Total parameters: {parameter_counts['total_parameters']}")
    print(f"  Trainable parameters: {parameter_counts['trainable_parameters']}")
    print(f"  Frozen parameters: {parameter_counts['frozen_parameters']}")

    data_collator = Wav2Vec2DataCollator(processor=processor)

    training_args = create_training_arguments(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
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

    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    processor.save_pretrained(str(best_model_dir))

    full_report = {
        "run_name": run_name,
        "model_name": "Wav2Vec2 emotion classifier",
        "base_checkpoint": model_checkpoint,
        "task": TASK_NAME,
        "device": get_device_note(),
        "dataset_source": dataset_source,
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "best_model_dir": str(best_model_dir),
        "mlflow_enabled": enable_mlflow,
        "mlflow_experiment_name": mlflow_experiment_name if enable_mlflow else None,
        "labels": label_encoding.id_to_label,
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "test_samples": len(datasets["test"]),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "freeze_feature_encoder": freeze_feature_encoder,
        "freeze_transformer_layers": freeze_transformer_layers,
        "total_parameters": parameter_counts["total_parameters"],
        "trainable_parameters": parameter_counts["trainable_parameters"],
        "frozen_parameters": parameter_counts["frozen_parameters"],
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "validation": validation_metrics,
        "test": test_report,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(full_report, file, indent=2)
    mlflow_run_id = None

    if enable_mlflow:
        mlflow_run_id = log_training_run_to_mlflow(
            full_report=full_report,
            report_path=report_path,
            confusion_matrix_path=confusion_matrix_path,
            run_name=run_name,
            mlflow_experiment_name=mlflow_experiment_name,
        )

        full_report["mlflow_run_id"] = mlflow_run_id

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(full_report, file, indent=2)

    print("\nWav2Vec2 Results")
    print("-" * 70)
    print(f"Freeze feature encoder: {freeze_feature_encoder}")
    print(f"Freeze transformer layers: {freeze_transformer_layers}")
    print(f"Validation accuracy: {validation_metrics.get('eval_accuracy'):.4f}")
    print(f"Validation macro F1: {validation_metrics.get('eval_macro_f1'):.4f}")
    print(f"Test accuracy: {test_report['accuracy']:.4f}")
    print(f"Test macro F1: {test_report['macro_f1']:.4f}")
    print("-" * 70)
    print(f"Saved model to: {best_model_dir}")
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    if enable_mlflow:
        print(f"Logged MLflow run ID: {mlflow_run_id}")

    return full_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2 for speech emotion classification."
    )

    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help="Hugging Face Wav2Vec2 checkpoint.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata CSV. Defaults to CREMA-D metadata.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="wav2vec2_emotion",
        help="Run name used for model folder and report filenames.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Enable MLflow logging for this training run.",
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for saved Wav2Vec2 models.",
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
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler.",
    )
    parser.add_argument(
        "--freeze-feature-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze Wav2Vec2 CNN feature encoder. Use --no-freeze-feature-encoder to unfreeze.",
    )

    parser.add_argument(
        "--freeze-transformer-layers",
        type=int,
        default=0,
        help="Number of lower Wav2Vec2 transformer layers to freeze.",
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
        metadata_path=args.metadata_path,
        run_name=args.run_name,
        output_root=args.output_root,
        limit_per_split=args.limit_per_split,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        freeze_feature_encoder=args.freeze_feature_encoder,
        freeze_transformer_layers=args.freeze_transformer_layers,
        max_duration_seconds=args.max_duration_seconds,
        enable_mlflow=args.enable_mlflow,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )