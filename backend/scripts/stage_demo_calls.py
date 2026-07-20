"""Stage short manifest calls in Supabase for catalog-driven live analysis.

The script uploads the original channel WAVs and inserts Call rows without an
index_summary. Those rows appear as available in GET /calls/catalog and are not
processed until POST /calls/analyze is called.

  python backend/scripts/stage_demo_calls.py --count 3
  python backend/scripts/stage_demo_calls.py --only <call_id> [<call_id> ...]
"""
import argparse
import json
import sys
import wave
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from app import storage
from app.database import SessionLocal, init_db
from app.models import Agent, Call

DATA = REPO / "data" / "na_testset"
MANIFEST = DATA / "manifest.json"
DEFAULT_AGENT = "Demo Agent"


def _local_path(relative):
    return DATA / Path(relative.replace("\\", "/"))


def _duration_seconds(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def _get_or_create_agent(db):
    agent = db.query(Agent).filter(Agent.name == DEFAULT_AGENT).first()
    if not agent:
        agent = Agent(name=DEFAULT_AGENT, team="demo")
        db.add(agent)
        db.flush()
    return agent


def _candidate_rows(rows, selected, count):
    by_id = {row["call_id"]: row for row in rows}
    if selected:
        missing = [call_id for call_id in selected if call_id not in by_id]
        if missing:
            raise SystemExit(f"unknown manifest call id(s): {', '.join(missing)}")
        return [by_id[call_id] for call_id in selected]

    candidates = []
    for row in rows:
        agent_wav = _local_path(row["agent_wav"])
        customer_wav = _local_path(row["customer_wav"])
        transcript = DATA / "results" / row["accent"] / f"{row['call_id']}.json"
        if agent_wav.exists() and customer_wav.exists() and not transcript.exists():
            candidates.append((_duration_seconds(agent_wav), row))
    candidates.sort(key=lambda item: item[0])
    return [row for _, row in candidates[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3,
                        help="number of shortest untranscribed calls to stage")
    parser.add_argument("--only", nargs="+", help="specific manifest call ids")
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not storage.is_configured():
        raise SystemExit("SUPABASE_URL + a storage write key must be set")

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = _candidate_rows(rows, args.only, args.count)
    if not selected:
        raise SystemExit("no eligible calls found")

    init_db()
    db = SessionLocal()
    try:
        agent = _get_or_create_agent(db)
        staged = 0
        for row in selected:
            public_id = row["call_id"]
            existing = db.query(Call).filter(
                Call.call_metadata["public_call_id"].astext == public_id
            ).first()
            if existing:
                print(f"  skipped {public_id} (already in catalog)")
                continue

            agent_wav = _local_path(row["agent_wav"])
            customer_wav = _local_path(row["customer_wav"])
            duration = _duration_seconds(agent_wav)
            storage.upload_file(
                f"uploads/{public_id}/agent.wav", str(agent_wav), "audio/wav"
            )
            storage.upload_file(
                f"uploads/{public_id}/customer.wav", str(customer_wav), "audio/wav"
            )
            db.add(Call(
                agent_id=agent.agent_id,
                audio_path=f"uploads/{public_id}/",
                call_date=datetime.utcnow(),
                duration_seconds=int(duration),
                call_metadata={
                    "public_call_id": public_id,
                    "domain": row["domain"],
                    "accent": row["accent"],
                },
            ))
            db.commit()
            staged += 1
            print(f"  staged {public_id} ({duration:.1f}s, {row['domain']})")
        print(f"\nstaged {staged} call(s)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
