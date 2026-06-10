"""
Prepare AppTek calls for selected call-center domains without decoding audio through Hugging Face.

This avoids the torchcodec dependency by casting the audio column with decode=False
and copying the original audio file/bytes into our processed folder.

Examples:

    # Small demo
    python -m src.data.prepare_apptek_domain_samples --samples-per-domain 3 --overwrite

    # Full 3-domain subset
    python -m src.data.prepare_apptek_domain_samples --samples-per-domain all --overwrite
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset


DATASET_NAME = "apptek-com/apptek_callcenter_dialogues"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SERVICES_ROOT = PROJECT_ROOT / "ml-services"

OUTPUT_DIR = ML_SERVICES_ROOT / "data" / "processed" / "apptek_selected_domains"
AUDIO_DIR = OUTPUT_DIR / "audio"
METADATA_PATH = OUTPUT_DIR / "apptek_selected_domain_metadata.csv"
SUMMARY_PATH = OUTPUT_DIR / "apptek_selected_domain_summary.json"


DOMAIN_MAPPING = {
    "banking": "banking",
    "health": "healthcare",
    "telecom": "telecommunications",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-domain",
        default="3",
        help="Number of calls per selected domain, or 'all'. Example: 3, 10, all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output metadata/audio files.",
    )
    return parser.parse_args()


def export_audio_without_decoding(audio_obj, output_path: Path) -> None:
    """
    Export HF audio object without decoding with torchcodec.

    With decode=False, the audio object usually has:
    - path: local cached file path
    - bytes: optional audio bytes

    We copy the path if available, otherwise write bytes.
    """
    audio_path = audio_obj.get("path")
    audio_bytes = audio_obj.get("bytes")

    if audio_path:
        shutil.copyfile(audio_path, output_path)
        return

    if audio_bytes:
        with output_path.open("wb") as file:
            file.write(audio_bytes)
        return

    raise ValueError(f"Could not export audio. Audio object keys: {audio_obj.keys()}")


def get_audio_info(audio_path: Path):
    """
    Get duration and sample rate from exported audio file.
    """
    info = sf.info(str(audio_path))
    duration_seconds = round(float(info.duration), 3)
    sampling_rate = int(info.samplerate)
    return duration_seconds, sampling_rate


def main():
    args = parse_args()

    samples_arg = str(args.samples_per_domain).lower().strip()

    if samples_arg == "all":
        samples_per_domain = None
    else:
        samples_per_domain = int(samples_arg)
        if samples_per_domain <= 0:
            raise ValueError("--samples-per-domain must be positive or 'all'.")

    if OUTPUT_DIR.exists() and args.overwrite:
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading AppTek test split without audio decoding...")
    print("-" * 80)

    ds = load_dataset(DATASET_NAME, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    selected_rows = []
    selected_counts = Counter()
    raw_counts = Counter()
    call_number_by_domain = defaultdict(int)

    print("Selecting and exporting selected domain calls...")
    print("-" * 80)

    for row in ds:
        raw_domain = str(row.get("domain", "")).lower().strip()

        if raw_domain not in DOMAIN_MAPPING:
            continue

        selected_domain = DOMAIN_MAPPING[raw_domain]

        if samples_per_domain is not None and selected_counts[selected_domain] >= samples_per_domain:
            continue

        call_number_by_domain[selected_domain] += 1
        selected_counts[selected_domain] += 1
        raw_counts[raw_domain] += 1

        call_id = f"APPTEK_{selected_domain.upper()}_{call_number_by_domain[selected_domain]:04d}"
        audio_filename = f"{call_id}.wav"
        output_audio_path = AUDIO_DIR / audio_filename

        export_audio_without_decoding(row["audio"], output_audio_path)
        duration_seconds, sampling_rate = get_audio_info(output_audio_path)

        selected_rows.append(
            {
                "call_id": call_id,
                "selected_domain": selected_domain,
                "raw_domain": raw_domain,
                "domain": selected_domain,
                "gender": row.get("gender", ""),
                "accent": row.get("accent", ""),
                "duration_seconds": duration_seconds,
                "sampling_rate": sampling_rate,
                "audio_path": str(output_audio_path.relative_to(ML_SERVICES_ROOT)),
                "text": row.get("text", ""),
            }
        )

        if selected_counts[selected_domain] % 25 == 0:
            print(
                f"Exported {selected_counts[selected_domain]} calls for {selected_domain}..."
            )

    metadata_df = pd.DataFrame(selected_rows)
    metadata_df.to_csv(METADATA_PATH, index=False)

    summary = {
        "dataset": "AppTek Call Center Dialogues",
        "source": DATASET_NAME,
        "target_domains": ["banking", "healthcare", "telecommunications"],
        "samples_per_domain_requested": "all" if samples_per_domain is None else samples_per_domain,
        "selected_rows": int(len(metadata_df)),
        "selected_domain_counts": dict(selected_counts),
        "raw_domain_counts": dict(raw_counts),
        "missing_target_domains": [
            domain
            for domain in ["banking", "healthcare", "telecommunications"]
            if selected_counts[domain] == 0
        ],
        "notes": [
            "Selected domain samples are used for realistic call-center inference/demo.",
            "They are not used as supervised emotion accuracy labels.",
            "AppTek raw domains are mapped as banking -> banking, health -> healthcare, telecom -> telecommunications.",
            "WAV audio files should not be committed to GitHub because they are large.",
            "Audio was exported with decode=False to avoid requiring torchcodec.",
        ],
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSelected AppTek domain preparation completed.")
    print("-" * 80)
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved audio: {AUDIO_DIR}")
    print("-" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
