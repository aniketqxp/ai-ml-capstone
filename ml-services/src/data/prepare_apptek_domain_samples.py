"""
Prepare AppTek samples for the selected capstone domains.

Target domains:
    - banking
    - healthcare
    - telecommunications / telecom if available

This script:
    1. Loads AppTek test split
    2. Selects a small number of calls from chosen domains
    3. Exports audio to WAV
    4. Creates metadata for inference

Run from ml-services:

    python -m src.data.prepare_apptek_domain_samples
"""

import json
import wave
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

DATASET_NAME = "apptek-com/apptek_callcenter_dialogues"

OUTPUT_ROOT = ML_SERVICES_ROOT / "data" / "processed" / "apptek_selected_domains"
AUDIO_DIR = OUTPUT_ROOT / "audio"
METADATA_PATH = OUTPUT_ROOT / "apptek_selected_domain_metadata.csv"
SUMMARY_PATH = OUTPUT_ROOT / "apptek_selected_domain_summary.json"

SAMPLES_PER_DOMAIN = 3

# These are flexible aliases. The script will match by lowercase text.
TARGET_DOMAIN_ALIASES: Dict[str, List[str]] = {
    "banking": ["banking", "bank", "finance", "financial"],
    "healthcare": ["healthcare", "health", "medical", "clinic", "hospital"],
    "telecommunications": [
        "telecommunications",
        "telecom",
        "communication",
        "mobile",
        "phone",
        "internet",
        "broadband",
    ],
}


def save_audio_as_wav(audio_array: np.ndarray, sampling_rate: int, output_path: Path) -> None:
    """
    Save mono audio array as WAV.
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


def match_target_domain(domain: str) -> str | None:
    """
    Return normalized target domain name if the raw AppTek domain matches.
    """
    domain_lower = str(domain).lower()

    for normalized_domain, aliases in TARGET_DOMAIN_ALIASES.items():
        for alias in aliases:
            if alias in domain_lower:
                return normalized_domain

    return None


def main() -> None:
    print("\nLoading AppTek test split...")
    print("-" * 80)

    ds = load_dataset(DATASET_NAME, split="test")

    selected_rows = []
    counts = defaultdict(int)
    raw_domain_matches = defaultdict(list)

    print("Selecting domain samples...")
    for index, row in enumerate(ds):
        raw_domain = row.get("domain", "")
        normalized_domain = match_target_domain(raw_domain)

        if normalized_domain is None:
            continue

        if counts[normalized_domain] >= SAMPLES_PER_DOMAIN:
            continue

        selected_rows.append((index, normalized_domain, row))
        counts[normalized_domain] += 1
        raw_domain_matches[normalized_domain].append(raw_domain)

        if all(
            counts[target_domain] >= SAMPLES_PER_DOMAIN
            for target_domain in TARGET_DOMAIN_ALIASES.keys()
        ):
            break

    print("\nSelected counts:")
    for domain in TARGET_DOMAIN_ALIASES.keys():
        print(f"{domain}: {counts[domain]}")

    missing_domains = [
        domain
        for domain in TARGET_DOMAIN_ALIASES.keys()
        if counts[domain] == 0
    ]

    if missing_domains:
        print("\nMissing target domains:")
        for domain in missing_domains:
            print(f"- {domain}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for selected_index, normalized_domain, row in selected_rows:
        call_id = f"APPTEK_{normalized_domain.upper()}_{counts[normalized_domain]:02d}_{selected_index:04d}"

        # Better readable sequential ID
        domain_existing_count = sum(
            1 for record in records if record["selected_domain"] == normalized_domain
        )
        call_id = f"APPTEK_{normalized_domain.upper()}_{domain_existing_count + 1:02d}"

        audio = row["audio"]
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]

        output_audio_path = AUDIO_DIR / f"{call_id}.wav"

        save_audio_as_wav(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            output_path=output_audio_path,
        )

        records.append(
            {
                "call_id": call_id,
                "selected_domain": normalized_domain,
                "raw_domain": row.get("domain", ""),
                "audio_path": str(output_audio_path.relative_to(ML_SERVICES_ROOT)),
                "text": row.get("text", ""),
                "gender": row.get("gender", ""),
                "accent": row.get("accent", ""),
                "sampling_rate": sampling_rate,
                "duration_seconds": round(len(audio_array) / sampling_rate, 3),
            }
        )

    metadata = pd.DataFrame(records)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(METADATA_PATH, index=False)

    summary = {
        "dataset": "AppTek Call Center Dialogues",
        "source": DATASET_NAME,
        "target_domains": list(TARGET_DOMAIN_ALIASES.keys()),
        "samples_per_domain_requested": SAMPLES_PER_DOMAIN,
        "selected_rows": len(metadata),
        "selected_domain_counts": metadata["selected_domain"].value_counts().to_dict()
        if len(metadata) > 0
        else {},
        "raw_domain_counts": metadata["raw_domain"].value_counts().to_dict()
        if len(metadata) > 0
        else {},
        "missing_target_domains": missing_domains,
        "notes": [
            "Selected domain samples are used for realistic call-center inference/demo.",
            "They are not used as supervised emotion accuracy labels.",
            "Telecommunications is included only if a matching AppTek domain exists.",
        ],
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSelected AppTek domain sample preparation completed.")
    print("-" * 80)
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved audio: {AUDIO_DIR}")
    print("-" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()