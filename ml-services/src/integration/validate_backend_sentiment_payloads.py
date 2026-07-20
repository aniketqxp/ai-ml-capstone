import json
from pathlib import Path
from collections import Counter


ML_SERVICES_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = ML_SERVICES_ROOT / "outputs" / "backend" / "sentiment_calls"


def main():
    files = sorted(PAYLOAD_DIR.rglob("*backend_sentiment.json"))

    print("Backend sentiment payload validation")
    print("-" * 80)
    print("Files:", len(files))

    status_counts = Counter()
    risk_counts = Counter()
    domain_counts = Counter()

    bad_files = []
    duplicate_seq_warnings = []

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        required_top = [
            "call_id",
            "domain",
            "model_version",
            "join_keys",
            "call_summary",
            "segments",
        ]

        missing_top = [field for field in required_top if field not in data]

        if missing_top:
            bad_files.append((str(file), f"Missing top fields: {missing_top}"))
            continue

        segments = data["segments"]
        total_segments = data["call_summary"].get("total_segments")

        if len(segments) != total_segments:
            bad_files.append((str(file), f"Segment count mismatch: {len(segments)} != {total_segments}"))

        seen_segment_indexes = set()
        seen_segment_keys = set()
        seq_ids = []

        for seg in segments:
            required_segment_fields = [
                "segment_index",
                "segment_key",
                "seq_id",
                "sentiment",
                "dominant_emotion",
                "escalation_score",
                "processing_status",
            ]

            for field in required_segment_fields:
                if field not in seg:
                    bad_files.append((str(file), f"Missing segment field: {field}"))

            segment_index = seg.get("segment_index")
            segment_key = seg.get("segment_key")

            if segment_index in seen_segment_indexes:
                bad_files.append((str(file), f"Duplicate segment_index: {segment_index}"))

            if segment_key in seen_segment_keys:
                bad_files.append((str(file), f"Duplicate segment_key: {segment_key}"))

            seen_segment_indexes.add(segment_index)
            seen_segment_keys.add(segment_key)
            seq_ids.append(seg.get("seq_id"))

            status_counts[seg.get("processing_status")] += 1

        seq_counts = Counter(seq_ids)
        duplicates = [seq for seq, count in seq_counts.items() if seq is not None and count > 1]

        if duplicates:
            duplicate_seq_warnings.append((str(file), duplicates))

        risk_counts[data["call_summary"].get("risk_level")] += 1
        domain_counts[data["domain"]] += 1

    print()
    print("Domain counts:", dict(domain_counts))
    print("Risk counts:", dict(risk_counts))
    print("Segment status counts:", dict(status_counts))
    print()

    if duplicate_seq_warnings:
        print("Duplicate seq_id warnings:")
        for file, duplicates in duplicate_seq_warnings:
            print(file, "=>", duplicates)
        print()

    if bad_files:
        print("Validation failed")
        for file, error in bad_files[:20]:
            print(file, "=>", error)
        raise SystemExit(1)

    print("Validation passed: all backend sentiment payloads are valid.")
    print("Note: duplicate seq_id values are warnings only because segment_index and segment_key are unique.")


if __name__ == "__main__":
    main()
