"""
Dataset-specific evaluation for Model V2.

This script evaluates the trained Model V2 separately on:
    1. CREMA-D test samples
    2. RAVDESS test samples
    3. Combined test samples

Why:
    Combined accuracy can hide dataset-specific weakness. Since Model V2 was trained
    on CREMA-D + RAVDESS, we need to know whether it performs equally well on both
    datasets or whether one dataset is hurting the score.

Run from ml-services:

    python -m src.evaluation.evaluate_model_by_dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

from src.data.audio_dataset import (
    DEFAULT_SAMPLE_RATE,
    build_label_encoding,
    load_audio_file,
    load_metadata,
    resolve_audio_path,
)
from src.models.wav2vec2_emotion_model import Wav2Vec2DataCollator, Wav2Vec2EmotionDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

MODEL_DIR = ML_SERVICES_ROOT / "outputs" / "wav2vec2" / "model_v2_cremad_ravdess" / "best_model"
METADATA_PATH = ML_SERVICES_ROOT / "data" / "processed" / "combined_emotion_metadata.csv"
REPORTS_DIR = ML_SERVICES_ROOT / "outputs" / "reports"

DATASET_SPECIFIC_REPORT_PATH = REPORTS_DIR / "model_v2_dataset_specific_evaluation.json"
DATASET_SPECIFIC_MD_PATH = REPORTS_DIR / "model_v2_dataset_specific_evaluation.md"

CREMAD_CONFUSION_MATRIX_PATH = REPORTS_DIR / "model_v2_cremad_only_confusion_matrix.csv"
RAVDESS_CONFUSION_MATRIX_PATH = REPORTS_DIR / "model_v2_ravdess_only_confusion_matrix.csv"
COMBINED_CONFUSION_MATRIX_PATH = REPORTS_DIR / "model_v2_combined_recomputed_confusion_matrix.csv"

BATCH_SIZE = 4
MAX_DURATION_SECONDS = 6.0


def get_device() -> torch.device:
    """
    Select available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_note(device: torch.device) -> str:
    if device.type == "cuda":
        return "CUDA GPU available"

    if device.type == "mps":
        return "Apple Silicon MPS available"

    return "CPU only"


def create_subset_dataset(
    metadata: pd.DataFrame,
    dataset_name: Optional[str],
    label_to_id: Dict[str, int],
) -> Wav2Vec2EmotionDataset:
    """
    Create a test dataset subset.

    If dataset_name is None, use all test samples.
    Otherwise, filter by dataset column.
    """
    test_df = metadata[metadata["split"] == "test"].copy()

    if dataset_name is not None:
        test_df = test_df[test_df["dataset"] == dataset_name].copy()

    if test_df.empty:
        raise ValueError(f"No test samples found for dataset_name={dataset_name}")

    return Wav2Vec2EmotionDataset(
        metadata=test_df,
        label_to_id=label_to_id,
        sample_rate=DEFAULT_SAMPLE_RATE,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )


def evaluate_dataset(
    model: Wav2Vec2ForSequenceClassification,
    processor: Wav2Vec2Processor,
    dataset: Wav2Vec2EmotionDataset,
    id_to_label: Dict[int, str],
    device: torch.device,
    confusion_matrix_path: Path,
) -> Dict:
    """
    Evaluate model on one dataset subset.
    """
    collator = Wav2Vec2DataCollator(processor=processor)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()

    all_predictions: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")

            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)

            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    label_ids = sorted(id_to_label.keys())
    target_names = [id_to_label[index] for index in label_ids]

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=label_ids,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in target_names],
        columns=[f"predicted_{label}" for label in target_names],
    )

    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(confusion_matrix_path)

    return {
        "samples": len(dataset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix_path": str(confusion_matrix_path),
    }


def create_markdown_report(results: Dict) -> str:
    lines = []

    lines.append("# Model V2 Dataset-Specific Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This report evaluates Model V2 separately on CREMA-D and RAVDESS test samples. "
        "The goal is to identify whether the combined test score hides dataset-specific weakness."
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Dataset | Samples | Accuracy | Macro F1 | Weighted F1 |")
    lines.append("|---|---:|---:|---:|---:|")

    for dataset_key in ["CREMA-D", "RAVDESS", "Combined"]:
        item = results["dataset_results"][dataset_key]
        lines.append(
            f"| {dataset_key} | {item['samples']} | "
            f"{item['accuracy'] * 100:.2f}% | "
            f"{item['macro_f1'] * 100:.2f}% | "
            f"{item['weighted_f1'] * 100:.2f}% |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    cremad_acc = results["dataset_results"]["CREMA-D"]["accuracy"]
    ravdess_acc = results["dataset_results"]["RAVDESS"]["accuracy"]

    if cremad_acc > ravdess_acc:
        lines.append(
            "Model V2 performs better on CREMA-D than on RAVDESS. This suggests that the model "
            "may still be more adapted to CREMA-D even after adding RAVDESS."
        )
    elif ravdess_acc > cremad_acc:
        lines.append(
            "Model V2 performs better on RAVDESS than on CREMA-D. This suggests that adding "
            "RAVDESS helped the model learn patterns that transfer well to that dataset, but "
            "we need to check if CREMA-D performance dropped."
        )
    else:
        lines.append(
            "Model V2 performs similarly on CREMA-D and RAVDESS, which suggests balanced generalization."
        )

    lines.append("")
    lines.append("## Next Improvement Direction")
    lines.append("")
    lines.append(
        "Use this dataset-specific result to decide whether Model V3 should focus on "
        "hyperparameter tuning, augmentation, partial unfreezing, or dataset balancing."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing trained model directory: {MODEL_DIR}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing combined metadata: {METADATA_PATH}")

    device = get_device()

    print("\nLoading Model V2 for dataset-specific evaluation")
    print("-" * 80)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Device: {get_device_note(device)}")
    print("-" * 80)

    metadata = load_metadata(METADATA_PATH)
    label_encoding = build_label_encoding(task="emotion")

    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)

    cremad_dataset = create_subset_dataset(
        metadata=metadata,
        dataset_name="CREMA-D",
        label_to_id=label_encoding.label_to_id,
    )
    ravdess_dataset = create_subset_dataset(
        metadata=metadata,
        dataset_name="RAVDESS",
        label_to_id=label_encoding.label_to_id,
    )
    combined_dataset = create_subset_dataset(
        metadata=metadata,
        dataset_name=None,
        label_to_id=label_encoding.label_to_id,
    )

    results = {
        "model_version": "model_v2",
        "run_name": "model_v2_cremad_ravdess",
        "model_dir": str(MODEL_DIR),
        "metadata_path": str(METADATA_PATH),
        "device": get_device_note(device),
        "dataset_results": {
            "CREMA-D": evaluate_dataset(
                model=model,
                processor=processor,
                dataset=cremad_dataset,
                id_to_label=label_encoding.id_to_label,
                device=device,
                confusion_matrix_path=CREMAD_CONFUSION_MATRIX_PATH,
            ),
            "RAVDESS": evaluate_dataset(
                model=model,
                processor=processor,
                dataset=ravdess_dataset,
                id_to_label=label_encoding.id_to_label,
                device=device,
                confusion_matrix_path=RAVDESS_CONFUSION_MATRIX_PATH,
            ),
            "Combined": evaluate_dataset(
                model=model,
                processor=processor,
                dataset=combined_dataset,
                id_to_label=label_encoding.id_to_label,
                device=device,
                confusion_matrix_path=COMBINED_CONFUSION_MATRIX_PATH,
            ),
        },
    }

    with DATASET_SPECIFIC_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    markdown = create_markdown_report(results)

    with DATASET_SPECIFIC_MD_PATH.open("w", encoding="utf-8") as file:
        file.write(markdown)

    print("\nModel V2 Dataset-Specific Evaluation")
    print("-" * 80)

    for dataset_key, result in results["dataset_results"].items():
        print(
            f"{dataset_key}: "
            f"samples={result['samples']}, "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"weighted_f1={result['weighted_f1']:.4f}"
        )

    print("-" * 80)
    print(f"Saved JSON report to: {DATASET_SPECIFIC_REPORT_PATH}")
    print(f"Saved Markdown report to: {DATASET_SPECIFIC_MD_PATH}")


if __name__ == "__main__":
    main()