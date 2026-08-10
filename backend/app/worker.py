"""
Single background worker: one daemon thread draining a job queue.

Why one thread (not a pool): the evaluation graph fires many rate-limited LLM
calls per call and CPU whisper/torch already saturate the 2-vCPU Space.
Sequential processing keeps us under provider limits and off the CPU cliff.
Jobs are cheap to enqueue, slow to run (~5-15 min), so queue + one worker is
the right shape.

Restart recovery: the Space's disk is ephemeral and it sleeps/restarts freely.
On startup we re-enqueue every job still marked queued/processing and
re-download its original channels from storage; the orchestrator is idempotent
(an existing transcript is reused), so a re-run resumes rather than restarts.
"""
import hashlib
import json
import os
import queue
import re
import threading
import traceback
import uuid
from datetime import datetime

from app import storage
from app.database import SessionLocal
from app.models import (
    Call,
    CallAudioSummary,
    Evaluation,
    EvaluationRun,
    Job,
    SentimentSegment,
    Transcript,
)
from app.routing import apply_action_routing

_q = queue.Queue()
_thread = None
_started = False

# text escalation tier -> Asma's 0-10 escalation_risk scale
ESCALATION_MAP = {"none": 0, "review": 5, "escalate": 8}


def _uuid(v):
    return v if isinstance(v, uuid.UUID) else uuid.UUID(str(v))


def enqueue(job_id):
    _q.put(str(job_id))


def start():
    """Launch the worker thread and recover interrupted jobs (idempotent)."""
    global _thread, _started
    if _started:
        return
    _started = True
    _thread = threading.Thread(target=_loop, name="pipeline-worker", daemon=True)
    _thread.start()
    _recover()


def _recover():
    """Re-enqueue jobs left queued/processing by a previous (crashed) run --
    but ONLY jobs from this pipeline's ingest path (call_metadata carries a
    public_call_id). Legacy/foreign jobs are left untouched."""
    db = SessionLocal()
    try:
        stuck = db.query(Job).filter(Job.status.in_(["queued", "processing"])).all()
        recovered = 0
        for j in stuck:
            call = db.query(Call).filter(Call.call_id == j.call_id).first()
            meta = (call.call_metadata or {}) if call else {}
            if meta.get("public_call_id"):
                enqueue(j.job_id)
                recovered += 1
        if recovered:
            print(f"[worker] recovered {recovered} interrupted job(s)")
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


def _set(db, job, status=None, stage=None, error=None):
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    if error is not None:
        job.error = error[:2000]
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
        ("sentence_segments", f"sentence_segments/{public_id}.json", "application/json"),
        ("transcript", f"transcripts/{public_id}.json", "application/json"),
    ]
    if artifacts.get("acoustic") and p.get("sentiment"):
        plan.append(("sentiment", f"sentiment/{public_id}.json", "application/json"))
    if p.get("evaluation_v2"):
        plan.append((
            "evaluation_v2",
            f"evaluation-v2/{public_id}.json",
            "application/json",
        ))
    for local_key, obj_key, ctype in plan:
        local = p.get(local_key)
        if local and os.path.exists(local):
            storage.upload_file(obj_key, local, ctype)
            keys[local_key] = obj_key

    if p.get("evaluation"):
        with open(p["evaluation"], encoding="utf-8") as f:
            legacy = json.load(f)
        marker = hashlib.sha256(
            json.dumps(
                legacy,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        key = f"evaluation-runs/{public_id}/v1-{marker}.json"
        storage.upload_file(key, p["evaluation"], "application/json")
        keys["evaluation_v1_run"] = key
    if p.get("evaluation_v2"):
        with open(p["evaluation_v2"], encoding="utf-8") as f:
            shadow = json.load(f)
        marker = str(shadow.get("decision_sha256") or shadow["run_id"])
        marker = re.sub(r"[^a-zA-Z0-9_-]", "-", marker)[:64]
        key = f"evaluation-runs/{public_id}/v2-{marker}.json"
        storage.upload_file(key, p["evaluation_v2"], "application/json")
        keys["evaluation_v2_run"] = key
    return keys


def _write_db_rows(db, call, job, public_id, artifacts):
    """Populate the relational tables from the produced artifacts. The frontend
    reads artifacts from storage; these rows back the flags/summary endpoints."""
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
                seq_id=s.get("seq_id"), speaker=s.get("speaker"),
                start_time=s.get("start_time"), end_time=s.get("end_time"),
                text=s.get("text"),
                sentiment=s.get("sentiment_class") or s.get("sentiment"),
                dominant_emotion=s.get("dominant_emotion"),
                escalation_score=s.get("escalation_score"),
                processing_status=s.get("processing_status"),
                audio_features=s.get("audio_features"),
                has_audio_features=bool(s.get("audio_features")),
                audio_feature_version=sent.get("audio_feature_version"),
                domain=artifacts.get("domain"),
                model_version=model_version))
            if s.get("escalation_score") is not None:
                scores.append(s["escalation_score"])
        max_escalation = max(scores) if scores else None
        db.query(CallAudioSummary).filter(
            CallAudioSummary.call_id == public_id
        ).delete()
        db.add(CallAudioSummary(
            call_id=public_id,
            domain=artifacts.get("domain"),
            model_version=model_version,
            has_audio_features=bool(sent.get("has_audio_features")),
            audio_feature_version=sent.get("audio_feature_version"),
            audio_feature_match_summary={
                "matched_segments": sum(
                    bool(row.get("audio_features"))
                    for row in sent.get("segments", [])
                ),
                "total_sentiment_segments": len(sent.get("segments", [])),
            },
            dashboard_audio_feature_series=sent.get(
                "dashboard_audio_feature_series"
            ),
            call_summary={
                **(sent.get("call_summary") or {}),
                "audio_features": sent.get("audio_feature_summary") or {},
                "speaker_audio_features": sent.get(
                    "speaker_audio_feature_summary"
                ) or {},
            },
        ))

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

    from v2.runtime import legacy_attention_proxy

    legacy_proxy = legacy_attention_proxy(ev)
    db.query(EvaluationRun).filter(
        EvaluationRun.job_id == job.job_id,
        EvaluationRun.evaluator_version == "v1",
    ).delete()
    db.add(EvaluationRun(
        job_id=job.job_id,
        call_id=call.call_id,
        public_call_id=public_id,
        runtime_run_id=f"{public_id}:v1:{job.job_id}",
        evaluator_version="v1",
        mode="primary",
        status="succeeded",
        attention_required=(
            legacy_proxy.attention_required if legacy_proxy else None
        ),
        payload=ev,
    ))

    shadow = artifacts.get("evaluation_v2")
    if shadow:
        db.query(EvaluationRun).filter(
            EvaluationRun.job_id == job.job_id,
            EvaluationRun.evaluator_version.like("v2%"),
        ).delete(synchronize_session=False)
        decision = shadow.get("decision") or {}
        db.add(EvaluationRun(
            job_id=job.job_id,
            call_id=call.call_id,
            public_call_id=public_id,
            runtime_run_id=shadow["run_id"],
            evaluator_version=shadow["evaluator_version"],
            mode=shadow["mode"],
            status=shadow["status"],
            decision_sha256=shadow.get("decision_sha256"),
            attention_required=decision.get("attention_required"),
            payload=shadow,
        ))


def _run_job(job_id):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == _uuid(job_id)).first()
        if not job:
            return
        if job.status not in {"queued", "processing"}:
            print(
                f"[worker] skipped duplicate queue entry {job_id} "
                f"({job.status})"
            )
            return
        call = db.query(Call).filter(Call.call_id == job.call_id).first()
        meta = dict(call.call_metadata or {}) if call else {}
        public_id = meta.get("public_call_id")
        if not public_id:
            # not one of ours (e.g. a legacy job) -- refuse rather than crash
            _set(db, job, status="failed",
                 error="call has no public_call_id; not an ingest-pipeline call")
            return

        force_transcription = job.stage == "force_uploaded"
        _set(db, job, status="processing", stage="starting", error="")

        agent_wav, customer_wav = _ensure_local_wavs(public_id, meta)
        spec = {"call_id": public_id, "domain": meta["domain"],
                "accent": meta["accent"], "agent_wav": agent_wav,
                "customer_wav": customer_wav}

        def progress(stage):
            _set(db, job, stage=stage)

        from app.pipeline_bridge import get_process_call
        process_call = get_process_call()
        artifacts = process_call(
            spec,
            progress=progress,
            enable_acoustic=True,
            reuse_transcript=not force_transcription,
        )

        _set(db, job, stage="uploading")
        keys = _upload_artifacts(public_id, artifacts)

        _set(db, job, stage="persisting")
        _write_db_rows(db, call, job, public_id, artifacts)

        _set(db, job, stage="notifying")
        try:
            from app.email_notifications import process_recommended_email

            shadow = artifacts.get("evaluation_v2")
            if shadow:
                process_recommended_email(
                    db,
                    call,
                    public_id,
                    shadow,
                    approved=False,
                )
        except Exception as exc:
            print(f"[worker] email action unavailable: {type(exc).__name__}: {exc}")

        summary = artifacts.get("index_summary") or {}
        meta["index_summary"] = summary
        meta["artifact_keys"] = keys
        call.call_metadata = meta
        if summary.get("duration"):
            call.duration_seconds = int(summary["duration"])
        db.commit()

        _set(db, job, status="succeeded", stage="done", error="")
        print(f"[worker] job {job_id} succeeded ({public_id})")
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
