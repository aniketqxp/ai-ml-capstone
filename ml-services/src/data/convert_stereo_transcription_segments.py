"""
Convert stereo transcription sentence_segments.json into sentiment timestamp schema.

This supports separate audio files for:
- agent channel
- customer channel

Example mapping:
source_id = en_CA_Banking_1586889
agent audio = APPTEK_BANKING_0035.wav / channel1
customer audio = APPTEK_BANKING_0036.wav / channel2
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--source-apptek-id", required=True)
    parser.add_argument("--agent-audio-path", required=True)
    parser.add_argument("--customer-audio-path", required=True)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--min-duration-seconds", type=float, default=0.5)
    return parser.parse_args()


def normalize_speaker(speaker):
    speaker = str(speaker).strip().lower()

    if speaker == "agent":
        return "agent"
    if speaker == "customer":
        return "customer"

    return "unknown"


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    converted_segments = []
    skipped = 0

    for sentence in payload.get("sentences", []):
        start = float(sentence["start"])
        end = float(sentence["end"])
        duration = end - start
        speaker = normalize_speaker(sentence.get("speaker", "unknown"))

        if duration < args.min_duration_seconds:
            skipped += 1
            continue

        segment = {
            "segment_id": sentence.get("id", sentence.get("seq_id")),
            "seq_id": sentence.get("seq_id"),
            "speaker": speaker,
            "start_time_seconds": round(start, 3),
            "end_time_seconds": round(end, 3),
            "duration_seconds": round(duration, 3),
        }

        if args.include_text:
            segment["text"] = sentence.get("text", "")

        converted_segments.append(segment)

    output_payload = {
        "call_id": args.call_id,
        "source_apptek_id": args.source_apptek_id,
        "source_call_id": payload.get("call"),
        "source_model": payload.get("model"),
        "analysis_input_type": "speaker_timestamped_stereo_channels",
        "audio_paths": {
            "agent": args.agent_audio_path,
            "customer": args.customer_audio_path,
        },
        "channel_mapping": {
            "agent": "channel1",
            "customer": "channel2",
        },
        "total_source_sentences": len(payload.get("sentences", [])),
        "total_converted_segments": len(converted_segments),
        "skipped_short_segments": skipped,
        "segments": converted_segments,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, indent=2)

    print("Converted stereo transcription segments successfully.")
    print("-" * 80)
    print("Input:", input_path)
    print("Output:", output_path)
    print("Call ID:", args.call_id)
    print("Source AppTek ID:", args.source_apptek_id)
    print("Agent audio:", args.agent_audio_path)
    print("Customer audio:", args.customer_audio_path)
    print("Source sentences:", len(payload.get("sentences", [])))
    print("Converted segments:", len(converted_segments))
    print("Skipped short segments:", skipped)


if __name__ == "__main__":
    main()
