import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ML_SERVICES_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ML_SERVICES_ROOT.parent

DEFAULT_TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "sentence_segments"
DEFAULT_METADATA_PATH = ML_SERVICES_ROOT / "data" / "processed" / "apptek_selected_domains" / "apptek_selected_domain_metadata_with_source_ids.csv"

CONVERTED_DIR = ML_SERVICES_ROOT / "data" / "processed" / "transcription_segments" / "converted_batch"
SIMPLE_OUTPUT_DIR = ML_SERVICES_ROOT / "outputs" / "apptek" / "simple_sentiment_results"
TIMESTAMPED_OUTPUT_DIR = ML_SERVICES_ROOT / "outputs" / "apptek" / "timestamped_sentiment_results"
BATCH_SUMMARY_PATH = ML_SERVICES_ROOT / "outputs" / "apptek" / "batch_transcript_sentiment_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-dir", default=str(DEFAULT_TRANSCRIPT_DIR))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_metadata(metadata_path):
    mapping = {}

    with open(metadata_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            source_id = row.get("source_apptek_id")
            call_id = row.get("call_id")

            if source_id and call_id:
                mapping[source_id] = call_id

    return mapping


def get_channel_audio_paths(call_id, metadata_mapping):
    channel1_source_id = f"{call_id}_channel1"
    channel2_source_id = f"{call_id}_channel2"

    if channel1_source_id not in metadata_mapping:
        raise ValueError(f"Missing channel1 mapping for {channel1_source_id}")

    if channel2_source_id not in metadata_mapping:
        raise ValueError(f"Missing channel2 mapping for {channel2_source_id}")

    agent_call_id = metadata_mapping[channel1_source_id]
    customer_call_id = metadata_mapping[channel2_source_id]

    agent_audio_path = ML_SERVICES_ROOT / "data" / "processed" / "apptek_selected_domains" / "audio" / f"{agent_call_id}.wav"
    customer_audio_path = ML_SERVICES_ROOT / "data" / "processed" / "apptek_selected_domains" / "audio" / f"{customer_call_id}.wav"

    if not agent_audio_path.exists():
        raise FileNotFoundError(f"Missing agent audio file: {agent_audio_path}")

    if not customer_audio_path.exists():
        raise FileNotFoundError(f"Missing customer audio file: {customer_audio_path}")

    internal_call_id = f"{agent_call_id}_{customer_call_id.split('_')[-1]}"

    return internal_call_id, agent_audio_path, customer_audio_path


def run_command(command):
    print()
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, cwd=ML_SERVICES_ROOT, check=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = parse_args()

    transcript_dir = Path(args.transcript_dir)
    metadata_path = Path(args.metadata_path)

    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    SIMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTAMPED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata_mapping = load_metadata(metadata_path)
    transcript_files = sorted(transcript_dir.rglob("*.json"))

    if args.limit:
        transcript_files = transcript_files[: args.limit]

    print("Batch timestamped sentiment started")
    print("-" * 80)
    print(f"Transcript files found: {len(transcript_files)}")
    print("-" * 80)

    summary_rows = []

    for index, transcript_path in enumerate(transcript_files, start=1):
        print()
        print("=" * 80)
        print(f"[{index}/{len(transcript_files)}] Processing {transcript_path}")
        print("=" * 80)

        transcript_data = read_json(transcript_path)
        source_call_id = transcript_data.get("call_id") or transcript_data.get("call")
        domain = transcript_data.get("domain", transcript_path.parent.name)

        try:
            internal_call_id, agent_audio_path, customer_audio_path = get_channel_audio_paths(
                source_call_id,
                metadata_mapping,
            )

            converted_path = CONVERTED_DIR / domain / f"{source_call_id}_segments.json"
            converted_path.parent.mkdir(parents=True, exist_ok=True)

            simple_output_path = SIMPLE_OUTPUT_DIR / domain / f"{source_call_id}_simple_sentiment.json"
            simple_output_path.parent.mkdir(parents=True, exist_ok=True)

            timestamped_output_path = TIMESTAMPED_OUTPUT_DIR / f"{internal_call_id}_timestamped_sentiment.json"

            run_command([
                sys.executable,
                "-m",
                "src.data.convert_stereo_transcription_segments",
                "--input-path",
                str(transcript_path),
                "--output-path",
                str(converted_path),
                "--call-id",
                internal_call_id,
                "--source-apptek-id",
                source_call_id,
                "--agent-audio-path",
                str(agent_audio_path),
                "--customer-audio-path",
                str(customer_audio_path),
                "--include-text",
                "--min-duration-seconds",
                "0",
            ])

            run_command([
                sys.executable,
                "-m",
                "src.inference.run_timestamped_sentiment",
                "--segments-path",
                str(converted_path),
            ])

            run_command([
                sys.executable,
                "-m",
                "src.inference.export_simple_sentiment_schema",
                "--input-path",
                str(timestamped_output_path),
                "--output-path",
                str(simple_output_path),
                "--call-id",
                source_call_id,
                "--include-skipped",
            ])

            timestamped_data = read_json(timestamped_output_path)
            simple_data = read_json(simple_output_path)

            summary_rows.append({
                "source_call_id": source_call_id,
                "domain": domain,
                "status": "success",
                "total_segments": timestamped_data.get("total_segments"),
                "successful_segments": timestamped_data.get("successful_segments"),
                "returned_simple_segments": len(simple_data.get("segments", [])),
                "overall_audio_sentiment": timestamped_data.get("overall_audio_sentiment"),
                "dominant_emotion": timestamped_data.get("dominant_emotion"),
                "audio_escalation_score": timestamped_data.get("audio_escalation_score"),
                "risk_level": timestamped_data.get("risk_level"),
                "simple_output_path": str(simple_output_path.relative_to(ML_SERVICES_ROOT)),
            })

        except Exception as error:
            print(f"FAILED for {source_call_id}: {error}")

            summary_rows.append({
                "source_call_id": source_call_id,
                "domain": domain,
                "status": f"failed: {error}",
                "total_segments": "",
                "successful_segments": "",
                "returned_simple_segments": "",
                "overall_audio_sentiment": "",
                "dominant_emotion": "",
                "audio_escalation_score": "",
                "risk_level": "",
                "simple_output_path": "",
            })

    fieldnames = [
        "source_call_id",
        "domain",
        "status",
        "total_segments",
        "successful_segments",
        "returned_simple_segments",
        "overall_audio_sentiment",
        "dominant_emotion",
        "audio_escalation_score",
        "risk_level",
        "simple_output_path",
    ]

    with open(BATCH_SUMMARY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    successful = sum(1 for row in summary_rows if row["status"] == "success")
    failed = len(summary_rows) - successful

    print()
    print("Batch timestamped sentiment completed")
    print("-" * 80)
    print(f"Total files: {len(summary_rows)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Summary CSV: {BATCH_SUMMARY_PATH}")
    print("-" * 80)


if __name__ == "__main__":
    main()
