"""
Prepare AppTek Call Center Dialogues sample for emotion inference.

This script loads the saved Hugging Face AppTek sample and exports audio files
to local WAV files so the selected V5 emotion model can run on them.

Run from ml-services:

    python -m src.data.apptek_dataset
"""

import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

APPTEK_SAMPLE_DIR = ML_SERVICES_ROOT / "data" / "raw" / "apptek" / "test_sample_20"
APPTEK_AUDIO_DIR = ML_SERVICES_ROOT / "data" / "processed" / "apptek" / "audio"
APPTEK_METADATA_PATH = ML_SERVICES_ROOT / "data" / "processed" / "apptek" / "apptek_metadata.csv"
APPTEK_SUMMARY_PATH = ML_SERVICES_ROOT / "data" / "processed" / "apptek" / "apptek_summary.json"


def save_audio_as_wav(audio_array: np.ndarray, sampling_rate: int, output_path: Path) -> None:
    """
    Save a mono audio array as a WAV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_array = np.asarray(audio_array)

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0)

    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_int16.tobytes())


def prepare_apptek_sample() -> None:
    """
    Export AppTek sample audio files and metadata.
    """
    if not APPTEK_SAMPLE_DIR.exists():
        raise FileNotFoundError(
            f"AppTek sample not found: {APPTEK_SAMPLE_DIR}\n"
            "Download/save test_sample_20 first."
        )

    dataset = load_from_disk(str(APPTEK_SAMPLE_DIR))

    records = []

    APPTEK_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(dataset):
        call_id = f"APPTEK_{index:04d}"
        audio = row["audio"]

        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]

        output_audio_path = APPTEK_AUDIO_DIR / f"{call_id}.wav"

        save_audio_as_wav(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            output_path=output_audio_path,
        )

        records.append(
            {
                "call_id": call_id,
                "audio_path": str(output_audio_path.relative_to(ML_SERVICES_ROOT)),
                "text": row.get("text", ""),
                "domain": row.get("domain", ""),
                "gender": row.get("gender", ""),
                "accent": row.get("accent", ""),
                "sampling_rate": sampling_rate,
                "duration_seconds": round(len(audio_array) / sampling_rate, 3),
            }
        )

    metadata = pd.DataFrame(records)

    APPTEK_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(APPTEK_METADATA_PATH, index=False)

    summary = {
        "dataset": "AppTek Call Center Dialogues",
        "source": "apptek-com/apptek_callcenter_dialogues",
        "sample_rows": len(metadata),
        "domains": metadata["domain"].value_counts().to_dict(),
        "accents": metadata["accent"].value_counts().to_dict(),
        "genders": metadata["gender"].value_counts().to_dict(),
        "total_duration_seconds": float(metadata["duration_seconds"].sum()),
        "notes": [
            "This AppTek sample is used for realistic call-center inference/demo.",
            "It is not used for supervised emotion accuracy unless manually labeled.",
            "Audio files are exported locally so the selected V5 emotion model can process them.",
        ],
    }

    with APPTEK_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nAppTek sample preparation completed successfully.")
    print("-" * 70)
    print(f"Rows: {len(metadata)}")
    print(f"Saved metadata to: {APPTEK_METADATA_PATH}")
    print(f"Saved summary to: {APPTEK_SUMMARY_PATH}")
    print(f"Saved audio files to: {APPTEK_AUDIO_DIR}")
    print("-" * 70)


if __name__ == "__main__":
    prepare_apptek_sample()