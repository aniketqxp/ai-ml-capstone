"""
Batch audio feature extraction for all processed AppTek calls.

This script:
1. Finds backend sentiment JSON files.
2. Finds the matching transcript JSON using call_id.
3. Finds the matching local AppTek audio file.
4. Extracts pitch, volume, energy, pauses, and speech rate for each segment.
5. Merges the audio features into the sentiment JSON.
6. Saves dashboard-ready sentiment JSON with audio features.
7. Creates a batch summary CSV.

Note:
This version uses one matching audio file per call. Speaker/channel-specific
audio selection can be improved later.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_transcript_path(transcripts_dir: Path, call_id: str) -> Optional[Path]:
    matches = list(transcripts_dir.rglob(f"{call_id}.json"))
    if matches:
        return matches[0]
    return None


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def find_audio_from_metadata(metadata_dir: Path, audio_dir: Path, call_id: str) -> Optional[Path]:
    """
    Try to find the local audio file for a call_id using available metadata CSV files.

    This is intentionally flexible because metadata column names may differ.
    """
    csv_files = list(metadata_dir.glob("*.csv"))

    possible_id_columns = [
        "call_id",
        "source_apptek_id",
        "source_id",
        "original_id",
        "apptek_id",
        "id",
    ]

    possible_file_columns = [
        "audio_path",
        "local_audio_path",
        "local_path",
        "file_path",
        "filename",
        "file_name",
        "audio_file",
        "local_filename",
        "wav_file",
    ]

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        columns = list(df.columns)

        id_columns = [c for c in possible_id_columns if c in columns]
        file_columns = [c for c in possible_file_columns if c in columns]

        if not id_columns:
            # Also try any column that contains id-ish text
            id_columns = [c for c in columns if "id" in c.lower() or "source" in c.lower()]

        if not file_columns:
            # Also try any column that contains file/path/audio-ish text
            file_columns = [
                c for c in columns
                if "file" in c.lower() or "path" in c.lower() or "audio" in c.lower() or "wav" in c.lower()
            ]

        for _, row in df.iterrows():
            row_values = [normalize_text(row.get(c)) for c in id_columns]

            # Match call_id directly or inside source IDs like en_CA_Banking_1586889_channel1
            if not any(call_id in value for value in row_values):
                continue

            for file_col in file_columns:
                file_value = normalize_text(row.get(file_col))
                if not file_value:
                    continue

                candidate = Path(file_value)

                if candidate.exists():
                    return candidate

                candidate = audio_dir / file_value
                if candidate.exists():
                    return candidate

                candidate = metadata_dir / file_value
                if candidate.exists():
                    return candidate

                # If only filename without extension/name appears, search audio dir
                matches = list(audio_dir.rglob(Path(file_value).name))
                if matches:
                    return matches[0]

    return None


def fallback_find_audio(audio_dir: Path, call_id: str, domain: Optional[str]) -> Optional[Path]:
    """
    Last-resort fallback:
    If metadata matching fails, pick an audio file based on domain.

    This is not ideal, but it prevents full failure.
    For final version, metadata mapping should be used.
    """
    domain = (domain or "").lower()

    if "bank" in domain:
        pattern = "*BANKING*.wav"
    elif "health" in domain:
        pattern = "*HEALTH*.wav"
    elif "telecom" in domain or "telecommunication" in domain:
        pattern = "*TELECOMMUNICATIONS*.wav"
    else:
        pattern = "*.wav"

    matches = sorted(audio_dir.glob(pattern))
    if matches:
        return matches[0]

    return None

def valid_existing_feature_file(feature_path, transcript_path):
    """
    Check if the audio feature file already exists and looks valid.

    This lets us skip expensive audio feature extraction when we only need
    to rerun the merge step.
    """
    feature_path = Path(feature_path)
    transcript_path = Path(transcript_path)

    if not feature_path.exists():
        return False

    try:
        feature_data = load_json(feature_path)
        transcript_data = load_json(transcript_path)

        feature_segments = feature_data.get("segments", [])
        transcript_segments = transcript_data.get("sentences", [])

        if len(feature_segments) != len(transcript_segments):
            return False

        return True

    except Exception:
        return False


def run_command(command: List[str]) -> bool:
    print("\nRunning:")
    print(" ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print("Command failed.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recalculate audio features even if feature files already exist.",
    )

    parser.add_argument(
        "--sentiment-dir",
        default="outputs/backend/sentiment_calls",
        help="Directory containing backend sentiment JSON files.",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="../data/sentence_segments",
        help="Directory containing transcript sentence JSON files.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/processed/apptek_selected_domains",
        help="Directory containing AppTek metadata CSV files.",
    )
    parser.add_argument(
        "--audio-dir",
        default="data/processed/apptek_selected_domains/audio",
        help="Directory containing selected AppTek audio files.",
    )
    parser.add_argument(
        "--features-output-dir",
        default="outputs/features/calls",
        help="Output directory for extracted audio feature JSON files.",
    )
    parser.add_argument(
        "--merged-output-dir",
        default="outputs/backend/sentiment_calls_with_features",
        help="Output directory for merged sentiment + audio feature JSON files.",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/features/batch_audio_features_summary.csv",
        help="Batch summary CSV output path.",
    )

    args = parser.parse_args()

    sentiment_dir = Path(args.sentiment_dir)
    transcripts_dir = Path(args.transcripts_dir)
    metadata_dir = Path(args.metadata_dir)
    audio_dir = Path(args.audio_dir)
    features_output_dir = Path(args.features_output_dir)
    merged_output_dir = Path(args.merged_output_dir)
    summary_output = Path(args.summary_output)

    features_output_dir.mkdir(parents=True, exist_ok=True)
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    sentiment_files = sorted(sentiment_dir.rglob("*_backend_sentiment.json"))

    print(f"Found sentiment files: {len(sentiment_files)}")

    summary_rows = []

    for sentiment_path in sentiment_files:
        try:
            sentiment = load_json(sentiment_path)
            call_id = sentiment.get("call_id")
            domain = sentiment.get("domain")

            if not call_id:
                print(f"Skipping file without call_id: {sentiment_path}")
                continue

            print("\n" + "=" * 80)
            print(f"Processing call_id: {call_id}")
            print(f"Domain: {domain}")

            transcript_path = find_transcript_path(transcripts_dir, call_id)

            if transcript_path is None:
                print(f"Transcript not found for {call_id}")
                summary_rows.append({
                    "call_id": call_id,
                    "domain": domain,
                    "status": "failed_missing_transcript",
                    "transcript_path": None,
                    "audio_path": None,
                    "total_segments": None,
                    "matched_segments": None,
                })
                continue

            audio_path = find_audio_from_metadata(metadata_dir, audio_dir, call_id)

            if audio_path is None:
                print("Metadata audio match not found. Trying fallback domain audio match.")
                audio_path = fallback_find_audio(audio_dir, call_id, domain)

            if audio_path is None:
                print(f"Audio not found for {call_id}")
                summary_rows.append({
                    "call_id": call_id,
                    "domain": domain,
                    "status": "failed_missing_audio",
                    "transcript_path": str(transcript_path),
                    "audio_path": None,
                    "total_segments": None,
                    "matched_segments": None,
                })
                continue

            print(f"Transcript: {transcript_path}")
            print(f"Audio: {audio_path}")

            domain_folder = str(domain or "unknown").lower()
            feature_output_path = features_output_dir / domain_folder / f"{call_id}_audio_features.json"
            merged_output_path = merged_output_dir / domain_folder / f"{call_id}_backend_sentiment_with_features.json"

            call_start_time = time.time()

            should_extract = args.force_extract or not valid_existing_feature_file(
                feature_output_path,
                transcript_path,
            )

            if should_extract:
                print("Audio features: extracting")

                extract_ok = run_command([
                    "python",
                    "-m",
                    "src.features.audio_feature_extractor",
                    "--transcript-path",
                    str(transcript_path),
                    "--audio-path",
                    str(audio_path),
                    "--output-path",
                    str(feature_output_path),
                ])

                if not extract_ok:
                    summary_rows.append({
                        "call_id": call_id,
                        "domain": domain,
                        "status": "failed_feature_extraction",
                        "transcript_path": str(transcript_path),
                        "audio_path": str(audio_path),
                        "total_segments": None,
                        "matched_segments": None,
                    })
                    continue

            else:
                print(f"Audio features: using cached file {feature_output_path}")

            merge_ok = run_command([
                "python",
                "-m",
                "src.integration.merge_sentiment_audio_features",
                "--sentiment-path",
                str(sentiment_path),
                "--features-path",
                str(feature_output_path),
                "--output-path",
                str(merged_output_path),
            ])

            if not merge_ok:
                summary_rows.append({
                    "call_id": call_id,
                    "domain": domain,
                    "status": "failed_merge",
                    "transcript_path": str(transcript_path),
                    "audio_path": str(audio_path),
                    "total_segments": None,
                    "matched_segments": None,
                })
                continue

            call_elapsed_time = round(time.time() - call_start_time, 2)
            print(f"Call processing time: {call_elapsed_time} seconds")

            merged = load_json(merged_output_path)
            match_summary = merged.get("audio_feature_match_summary", {})

            summary_rows.append({
                "call_id": call_id,
                "domain": domain,
                "status": "success",
                "transcript_path": str(transcript_path),
                "audio_path": str(audio_path),
                "feature_source": "extracted" if should_extract else "cached",
                "processing_time_seconds": call_elapsed_time,
                "total_segments": match_summary.get("total_sentiment_segments"),
                "matched_segments": match_summary.get("matched_segments"),
                "has_audio_features": merged.get("has_audio_features"),
                "output_path": str(merged_output_path),
            })

        except Exception as e:
            print(f"Failed processing {sentiment_path}: {e}")
            summary_rows.append({
                "call_id": sentiment_path.stem,
                "domain": None,
                "status": "failed_exception",
                "error": str(e),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_output, index=False)

    print("\n" + "=" * 80)
    print("Batch audio feature extraction complete.")
    print(f"Summary saved to: {summary_output}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()