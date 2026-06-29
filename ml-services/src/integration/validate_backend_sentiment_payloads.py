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

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        required_top = ["call_id", "domain", "model_version", "call_summary", "segments"]
        missing_top = [field for field in required_top if field not in data]

        if missing_top:
            bad_files.append((str(file), f"Missing top fields: {missing_top}"))
            continue

        segments = data["segments"]
        total_segments = data["call_summary"].get("total_segments")

        if len(segments) != total_segments:
            bad_files.append((str(file), f"Segment count mismatch: {len(segments)} != {total_segments}"))

        seen_seq_ids = set()

        for seg in segments:
            for field in ["seq_id", "sentiment", "dominant_emotion", "escalation_score", "processing_status"]:
                if field not in seg:
                    bad_files.append((str(file), f"Missing segment field: {field}"))

            seq_id = seg.get("seq_id")
            if seq_id in seen_seq_ids:
                bad_files.append((str(file), f"Duplicate seq_id: {seq_id}"))
            seen_seq_ids.add(seq_id)

            status_counts[seg.get("processing_status")] += 1

        risk_counts[data["call_summary"].get("risk_level")] += 1
        domain_counts[data["domain"]] += 1

    print()
    print("Domain counts:", dict(domain_counts))
    print("Risk counts:", dict(risk_counts))
    print("Segment status counts:", dict(status_counts))
    print()

    if bad_files:
        print("Validation failed")
        for file, error in bad_files[:20]:
            print(file, "=>", error)
        raise SystemExit(1)

    print("Validation passed: all backend sentiment payloads are valid.")


if __name__ == "__main__":
    main()
