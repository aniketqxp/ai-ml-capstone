"""
Configurable background worker pool draining a shared job queue.

The worker count is controlled by PIPELINE_WORKERS. Keep it conservative: each
call runs CPU-heavy Whisper/acoustic inference and rate-limited LLM requests.

Restart recovery: the Space's disk is ephemeral and it sleeps/restarts freely.
On startup we re-enqueue every job still marked queued/processing and
re-download its original channels from storage; the orchestrator is idempotent
(an existing transcript is reused), so a re-run resumes rather than restarts.
"""
import os
import uuid
import queue
import threading
import traceback
import time
from datetime import datetime

from app.database import SessionLocal
from app import storage
from app.models import (Call, Job, Transcript, Evaluation,
                        SentimentSegment, CallAudioSummary)
from app.routing import apply_action_routing

_q = queue.Queue()
_threads = []
_started = False
WORKER_COUNT = max(1, int(os.environ.get("PIPELINE_WORKERS", "1")))

# text escalation tier -> Asma's 0-10 escalation_risk scale
ESCALATION_MAP = {"none": 0, "review": 5, "escalate": 8}

STAGE_PROGRESS = {
    "uploaded": 0, "starting": 2, "transcribing": 5,
    "registering": 28, "segmenting": 32, "acoustic": 35,
    "evaluating": 75, "exporting": 90, "notifying": 92, "uploading": 94,
    "persisting": 97, "done": 100,
}
STAGE_RANGES = {"transcribing": (5, 27), "acoustic": (35, 73)}


class JobCancelled(Exception):
    cancelled = True


def _uuid(v):
    return v if isinstance(v, uuid.UUID) else uuid.UUID(str(v))


def enqueue(job_id):
    _q.put(str(job_id))


def start():
    """Launch the worker pool and recover interrupted jobs (idempotent)."""
    global _started
    if _started:
        return
    _started = True
    for number in range(WORKER_COUNT):
        thread = threading.Thread(
            target=_loop, name=f"pipeline-worker-{number + 1}", daemon=True)
        thread.start()
        _threads.append(thread)
    print(f"[worker] started {WORKER_COUNT} pipeline worker(s)")
    _recover()


def _recover():
    """Re-enqueue jobs left queued/processing by a previous (crashed) run --
    but ONLY jobs from this pipeline's ingest path (call_metadata carries a
    public_call_id). Legacy/foreign jobs are left untouched."""
    db = SessionLocal()
    try:
        stuck = db.query(Job).filter(Job.status.in_(["queued", "processing"])).all()
        recovered_job_ids = []
        for j in stuck:
            call = db.query(Call).filter(Call.call_id == j.call_id).first()
            meta = (call.call_metadata or {}) if call else {}
            if meta.get("public_call_id"):
                # A recovered job is waiting until the single worker actually
                # picks it up. Leaving it as "processing" makes every recovered
                # job appear active even though they run sequentially.
                j.status = "queued"
                j.updated_at = datetime.utcnow()
                recovered_job_ids.append(j.job_id)
        db.commit()
        for job_id in recovered_job_ids:
            enqueue(job_id)
        if recovered_job_ids:
            print(f"[worker] recovered {len(recovered_job_ids)} interrupted job(s)")
    except Exception as e:
        print(f"[worker] recovery skipped (db unreachable?): {e}")
    finally:
        db.close()


def _loop():
    while True:
        job_id = _q.get()
        try:
            _run_job(job_id)
        except Exception:
            traceback.print_exc()
        finally:
            _q.task_done()


def _set(db, job, status=None, stage=None, error=None, percent=None,
         current=None, total=None, message=None, eta=None):
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    if error is not None:
        job.error = error[:2000]
    if percent is not None:
        job.progress_percent = max(0, min(100, int(percent)))
    job.progress_current = current
    job.progress_total = total
    if message is not None:
        job.progress_message = message[:200]
    job.estimated_seconds_remaining = eta
    job.updated_at = datetime.utcnow()
    db.commit()


def _ensure_local_wavs(public_id, meta):
    """Return local (agent_wav, customer_wav), re-downloading from storage if
    the ephemeral disk was wiped (restart recovery)."""
    from app.pipeline_bridge import install
    install()
    import paths
    dest = paths.DATA_ROOT / "_incoming" / public_id
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for role in ("agent", "customer"):
        local = dest / f"{role}.wav"
        if not local.exists():
            data = storage.download_bytes(f"uploads/{public_id}/{role}.wav",
                                          use_cache=False)
            local.write_bytes(data)
        out[role] = str(local)
    return out["agent"], out["customer"]


def _upload_artifacts(public_id, artifacts):
    """Push every produced artifact to storage; return the key map."""
    p = artifacts.get("paths", {})
    keys = {}
    plan = [
        ("call_json", f"calls/{public_id}.json", "application/json"),
        ("audio", f"audio/{public_id}.mp3", "audio/mpeg"),
        ("evaluation", f"evaluation/{public_id}.json", "application/json"),
        ("sentence_segments", f"sentence_segments/{public_id}.json", "application/json"),
        ("transcript", f"transcripts/{public_id}.json", "application/json"),
    ]
    if artifacts.get("acoustic") and p.get("sentiment"):
        plan.append(("sentiment", f"sentiment/{public_id}.json", "application/json"))
    for local_key, obj_key, ctype in plan:
        local = p.get(local_key)
        if local and os.path.exists(local):
            storage.upload_file(obj_key, local, ctype)
            keys[local_key] = obj_key
    return keys


def _write_db_rows(db, call, public_id, artifacts):
    """Populate the relational tables from the produced artifacts. The frontend
    reads artifacts from storage; these rows back the flags/summary endpoints."""
    import json
    p = artifacts["paths"]

    # transcripts: one row per sentence segment
    with open(p["sentence_segments"], encoding="utf-8") as f:
        seg = json.load(f)
    db.query(Transcript).filter(Transcript.source_call_id == public_id).delete()
    for s in seg.get("sentences", []):
        db.add(Transcript(
            call_id=call.call_id, source_call_id=public_id,
            turn_id=s.get("seq_id", 0), speaker=(s.get("speaker") or "").lower(),
            start_time=s.get("start"), end_time=s.get("end"),
            text=s.get("text"), avg_confidence=1.0, low_confidence=False))

    # sentiment segments (acoustic only) + max escalation for routing
    max_escalation = None
    if artifacts.get("acoustic") and p.get("sentiment"):
        with open(p["sentiment"], encoding="utf-8") as f:
            sent = json.load(f)
        db.query(SentimentSegment).filter(
            SentimentSegment.call_id == public_id).delete()
        # model_version column is varchar(100); guard against long values
        model_version = (sent.get("model_version") or "")[:100]
        scores = []
        for i, s in enumerate(sent.get("segments", [])):
            db.add(SentimentSegment(
                call_id=public_id, segment_index=i,
                segment_key=f"{public_id}_{s.get('seq_id')}",
                seq_id=s.get("seq_id"), sentiment=s.get("sentiment"),
                dominant_emotion=s.get("dominant_emotion"),
                escalation_score=s.get("escalation_score"),
                processing_status=s.get("processing_status"),
                domain=artifacts.get("domain"),
                model_version=model_version))
            if s.get("escalation_score") is not None:
                scores.append(s["escalation_score"])
        max_escalation = max(scores) if scores else None

    # evaluation: full graph.json as the scorecard, tier -> 0-10 risk
    with open(p["evaluation"], encoding="utf-8") as f:
        ev = json.load(f)
    risk_level = (ev.get("escalation") or {}).get("risk_level", "none")
    db.query(Evaluation).filter(Evaluation.call_id == call.call_id).delete()
    evaluation = Evaluation(
        call_id=call.call_id, agent_id=call.agent_id,
        scorecard=ev, compliance_flags=ev.get("compliance"),
        escalation_risk=ESCALATION_MAP.get(risk_level, 0), llm_scored=True)
    apply_action_routing(evaluation, max_escalation)
    db.add(evaluation)


def _run_job(job_id):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == _uuid(job_id)).first()
        if not job:
            return
        if job.cancel_requested:
            _set(db, job, status="cancelled", stage="cancelled", percent=0,
                 message="Cancelled before processing", eta=0)
            return
        call = db.query(Call).filter(Call.call_id == job.call_id).first()
        meta = dict(call.call_metadata or {}) if call else {}
        public_id = meta.get("public_call_id")
        if not public_id:
            # not one of ours (e.g. a legacy job) -- refuse rather than crash
            _set(db, job, status="failed",
                 error="call has no public_call_id; not an ingest-pipeline call")
            return

        started = time.monotonic()
        job.started_at = datetime.utcnow()
        _set(db, job, status="processing", stage="starting", error="",
             percent=2, message="Preparing audio", eta=None)

        agent_wav, customer_wav = _ensure_local_wavs(public_id, meta)
        spec = {"call_id": public_id, "domain": meta["domain"],
                "accent": meta["accent"], "agent_wav": agent_wav,
                "customer_wav": customer_wav}

        def progress(stage, current=None, total=None, message=None):
            db.refresh(job)
            if job.cancel_requested:
                raise JobCancelled("Analysis cancelled by user")
            percent = STAGE_PROGRESS.get(stage, job.progress_percent or 0)
            if stage in STAGE_RANGES and total:
                lower, upper = STAGE_RANGES[stage]
                percent = lower + (upper - lower) * min(1, current / total)
            elapsed = time.monotonic() - started
            eta = None
            if percent >= 3:
                eta = max(0, round(elapsed * (100 - percent) / percent))
            _set(db, job, stage=stage, percent=percent, current=current,
                 total=total, message=message or stage.replace("_", " ").title(),
                 eta=eta)

        from app.pipeline_bridge import get_process_call
        process_call = get_process_call()
        artifacts = process_call(spec, progress=progress)

        progress("notifying", message="Preparing automation notifications")
        from app.email_notifications import process_email_notifications
        artifacts["email_notifications"] = process_email_notifications(
            db, call, public_id, artifacts)

        progress("uploading", message="Publishing analysis artifacts")
        keys = _upload_artifacts(public_id, artifacts)

        progress("persisting", message="Saving dashboard results")
        _write_db_rows(db, call, public_id, artifacts)

        summary = artifacts.get("index_summary") or {}
        meta["index_summary"] = summary
        meta["artifact_keys"] = keys
        call.call_metadata = meta
        if summary.get("duration"):
            call.duration_seconds = int(summary["duration"])
        db.commit()

        _set(db, job, status="succeeded", stage="done", error="", percent=100,
             message="Analysis complete", eta=0)
        print(f"[worker] job {job_id} succeeded ({public_id})")
    except JobCancelled:
        try:
            job = db.query(Job).filter(Job.job_id == _uuid(job_id)).first()
            if job:
                _set(db, job, status="cancelled", stage="cancelled", error="",
                     percent=job.progress_percent or 0, message="Analysis cancelled", eta=0)
        except Exception:
            pass
    except Exception as e:
        traceback.print_exc()
        try:
            job = db.query(Job).filter(Job.job_id == _uuid(job_id)).first()
            if job:
                _set(db, job, status="failed", error=str(e))
        except Exception:
            pass
    finally:
        db.close()
