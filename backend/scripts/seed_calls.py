"""
Idempotent seeder: push the curated demo calls' artifacts to Supabase Storage
and upsert their Call rows (with index_summary) so the hosted dashboard shows
the same set the local frontend/public did.

Runs from the DEV machine (frontend/public/* is gitignored but present there),
not in the Space. Re-running updates in place -- storage uploads use upsert and
Call rows are matched by public_call_id in call_metadata.

  python backend/scripts/seed_calls.py                # all calls in public/calls
  python backend/scripts/seed_calls.py --only en_CA_Banking_1586889 ...
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from app.database import SessionLocal, init_db
from app.models import Agent, Call, EvaluationRun
from app import storage

PUBLIC = REPO / "frontend" / "public"
CALLS = PUBLIC / "calls"
AUDIO = PUBLIC / "audio"
SEG = PUBLIC / "sentence_segments"
SENT = PUBLIC / "sentiment"
EVAL_V2 = PUBLIC / "evaluation-v2"
INDEX = PUBLIC / "calls_index.json"

DEFAULT_AGENT = "Demo Agent"


def get_or_create_agent(db):
    a = db.query(Agent).filter(Agent.name == DEFAULT_AGENT).first()
    if not a:
        a = Agent(name=DEFAULT_AGENT, team="demo")
        db.add(a)
        db.flush()
    return a


def upload_artifacts(cid):
    """Upload every present artifact for a call; return the storage key map."""
    keys = {}
    pairs = [
        (CALLS / f"{cid}.json", f"calls/{cid}.json", "application/json"),
        (AUDIO / f"{cid}.mp3", f"audio/{cid}.mp3", "audio/mpeg"),
        (SEG / f"{cid}.json", f"sentence_segments/{cid}.json", "application/json"),
        (SENT / f"{cid}.json", f"sentiment/{cid}.json", "application/json"),
        (EVAL_V2 / f"{cid}.json", f"evaluation-v2/{cid}.json", "application/json"),
    ]
    for local, key, ctype in pairs:
        if local.exists():
            storage.upload_file(key, str(local), ctype)
            keys[key.split("/", 1)[0]] = key
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="seed only these call_ids")
    args = ap.parse_args()

    if not storage.is_configured():
        raise SystemExit("SUPABASE_URL + a storage key must be set to seed")

    init_db()
    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []
    summaries = {s["call_id"]: s for s in index}

    call_files = sorted(CALLS.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        call_files = [c for c in call_files if c.stem in wanted]

    db = SessionLocal()
    agent = get_or_create_agent(db)
    seeded = 0
    for cj in call_files:
        cid = cj.stem
        summary = summaries.get(cid)
        call_json = json.loads(cj.read_text(encoding="utf-8"))
        ev_meta = (call_json.get("evaluation") or {}).get("metadata") or {}
        domain = (summary or {}).get("domain") or ev_meta.get("domain")
        accent = (summary or {}).get("accent") or ev_meta.get("accent")
        duration = (summary or {}).get("duration") or call_json.get("duration")

        keys = upload_artifacts(cid)

        meta = {"public_call_id": cid, "domain": domain, "accent": accent,
                "index_summary": summary, "artifact_keys": keys}
        call = db.query(Call).filter(
            Call.call_metadata["public_call_id"].astext == cid).first()
        if call:
            call.call_metadata = meta
            call.duration_seconds = int(duration) if duration else None
        else:
            call = Call(
                agent_id=agent.agent_id, audio_path=f"audio/{cid}.mp3",
                call_date=datetime.utcnow(),
                duration_seconds=int(duration) if duration else None,
                call_metadata=meta,
            )
            db.add(call)
            db.flush()

        v2_path = EVAL_V2 / f"{cid}.json"
        if v2_path.exists():
            run = json.loads(v2_path.read_text(encoding="utf-8"))
            db.query(EvaluationRun).filter(
                EvaluationRun.runtime_run_id == run["run_id"],
            ).delete(synchronize_session=False)
            decision = run.get("decision") or {}
            db.add(EvaluationRun(
                call_id=call.call_id,
                public_call_id=cid,
                runtime_run_id=run["run_id"],
                evaluator_version=run["evaluator_version"],
                mode=run["mode"],
                status=run["status"],
                decision_sha256=run.get("decision_sha256"),
                attention_required=decision.get("attention_required"),
                payload=run,
            ))
        seeded += 1
        print(f"  seeded {cid}  ({domain})")

    db.commit()
    db.close()
    print(f"\nseeded {seeded} call(s)")


if __name__ == "__main__":
    main()
