import os
import shutil
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app import storage, worker
from app.database import get_db
from app.models import (
    Call,
    Evaluation,
    EvaluationFeedback,
    EvaluationRun,
    Job,
    SentimentSegment,
    Transcript,
)
from app.routing import apply_action_routing

router = APIRouter(prefix="/calls", tags=["calls"])

PIPELINE_STAGES = (
    {
        "id": "queued",
        "label": "Queued",
        "worker_stages": {"uploaded", "starting"},
    },
    {
        "id": "transcript",
        "label": "Transcript",
        "worker_stages": {"transcribing", "registering"},
    },
    {
        "id": "segments",
        "label": "Sentence segments",
        "worker_stages": {"segmenting"},
    },
    {
        "id": "acoustic",
        "label": "Audio signals",
        "worker_stages": {"acoustic"},
        "feature_flag": "ENABLE_ACOUSTIC",
    },
    {
        "id": "evaluation",
        "label": "LLM evaluation",
        "worker_stages": {"evaluating"},
    },
    {
        "id": "evaluation_v2",
        "label": "V2 shadow",
        "worker_stages": {"evaluating_v2_shadow"},
        "feature_flag": "EVALUATOR_V2_SHADOW",
    },
    {
        "id": "publish",
        "label": "Publish results",
        "worker_stages": {"exporting", "uploading", "persisting"},
    },
    {
        "id": "ready",
        "label": "Ready",
        "worker_stages": {"done"},
    },
)


class AnalyzeRequest(BaseModel):
    call_ids: list[uuid.UUID] = Field(min_length=1, max_length=10)
    force: bool = False


class EvaluationFeedbackRequest(BaseModel):
    feedback_type: Literal[
        "approve_action",
        "dismiss_decision",
        "dismiss_finding",
        "confirm_finding",
        "action_completed",
        "action_failed",
    ]
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finding_id: str | None = None
    action_type: str | None = None
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_target(self):
        if self.feedback_type in {
            "dismiss_finding",
            "confirm_finding",
        } and not self.finding_id:
            raise ValueError(
                f"{self.feedback_type} requires finding_id"
            )
        if self.feedback_type in {
            "approve_action",
            "action_completed",
            "action_failed",
        } and not self.action_type:
            raise ValueError(
                f"{self.feedback_type} requires action_type"
            )
        return self


def _latest_job(db, call_id):
    return (
        db.query(Job)
        .filter(Job.call_id == call_id)
        .order_by(Job.created_at.desc())
        .first()
    )


def _flag_enabled(name):
    value = os.environ.get(name, "0").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _pipeline_progress(job):
    raw_stage = (job.stage or "uploaded").strip().lower()
    current_index = next(
        (
            index
            for index, step in enumerate(PIPELINE_STAGES)
            if raw_stage in step["worker_stages"]
        ),
        0,
    )
    enabled = [
        (
            not step.get("feature_flag")
            or _flag_enabled(step["feature_flag"])
            or raw_stage in step["worker_stages"]
        )
        for step in PIPELINE_STAGES
    ]
    terminal_success = job.status in {"succeeded", "complete"}
    failed = job.status == "failed"
    stages = []
    completed = 0

    for index, step in enumerate(PIPELINE_STAGES):
        if not enabled[index]:
            state = "skipped"
        elif terminal_success or index < current_index:
            state = "completed"
            completed += 1
        elif index == current_index:
            state = "failed" if failed else "active"
        else:
            state = "pending"
        stages.append({
            "id": step["id"],
            "label": step["label"],
            "state": state,
        })

    total = sum(enabled)
    percent = 100 if terminal_success else round(completed / total * 100)
    current = stages[current_index]
    return {
        "raw_stage": raw_stage,
        "current_stage_id": current["id"],
        "current_stage_label": current["label"],
        "percent": percent,
        "stages": stages,
    }


def _call_by_public_id(db, public_id):
    return (
        db.query(Call)
        .filter(
            Call.call_metadata["public_call_id"].astext == public_id
        )
        .first()
    )


def _latest_v2_run(db, public_id):
    return (
        db.query(EvaluationRun)
        .filter(
            EvaluationRun.public_call_id == public_id,
            EvaluationRun.evaluator_version.like("v2%"),
        )
        .order_by(EvaluationRun.created_at.desc())
        .first()
    )


def _incoming_dir(public_id):
    """Local scratch dir for this call's channel wavs (under CAPSTONE_DATA_ROOT)."""
    from app.pipeline_bridge import install
    install()
    import paths
    d = paths.DATA_ROOT / "_incoming" / public_id
    d.mkdir(parents=True, exist_ok=True)
    return d

@router.post("/ingest", status_code=201)
async def ingest_call(
    agent_id: str = Form(...),
    domain: str = Form(...),
    accent: str = Form("en-US_General"),
    call_date: str = Form(None),
    duration_seconds: int = Form(None),
    file: UploadFile = File(...),
    file2: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Ingest one fresh call. Accepts either a single stereo file (ch0=agent,
    ch1=customer) or two per-channel mono files. Splits/normalizes to two mono
    16 kHz WAVs, stores the originals for restart recovery, and enqueues the
    worker. A single mono file with no second channel is rejected (400).
    """
    from app.pipeline_bridge import install
    install()
    from audio_io import AudioError, prepare_channels

    public_id = f"{domain.lower()}_{datetime.utcnow():%Y%m%d}_{uuid.uuid4().hex[:8]}"
    inc = _incoming_dir(public_id)

    raw1 = inc / ("upload1_" + os.path.basename(file.filename or "a.wav"))
    with open(raw1, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    raw2 = None
    if file2 is not None:
        raw2 = inc / ("upload2_" + os.path.basename(file2.filename or "b.wav"))
        with open(raw2, "wb") as buf:
            shutil.copyfileobj(file2.file, buf)

    agent_wav = inc / "agent.wav"
    customer_wav = inc / "customer.wav"
    try:
        prepare_channels(str(raw1), str(agent_wav), str(customer_wav),
                         secondary=str(raw2) if raw2 else None)
    except AudioError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # originals to storage so a restart mid-job can re-download and resume
    storage.upload_file(f"uploads/{public_id}/agent.wav", str(agent_wav), "audio/wav")
    storage.upload_file(f"uploads/{public_id}/customer.wav", str(customer_wav), "audio/wav")

    call = Call(
        agent_id=uuid.UUID(agent_id),
        audio_path=f"uploads/{public_id}/",
        call_date=datetime.fromisoformat(call_date) if call_date else datetime.utcnow(),
        duration_seconds=duration_seconds,
        call_metadata={"public_call_id": public_id,
                       "domain": domain.lower(), "accent": accent},
    )
    db.add(call)
    db.flush()

    job = Job(call_id=call.call_id, status="queued", stage="uploaded")
    db.add(job)
    db.commit()
    db.refresh(job)

    worker.enqueue(job.job_id)

    return {
        "job_id": str(job.job_id),
        "call_id": str(call.call_id),
        "public_call_id": public_id,
        "status": "queued",
        "created_at": job.created_at.isoformat()
    }


@router.get("/catalog")
def get_catalog(db: Session = Depends(get_db)):
    """Return analyzed and staged calls in one dashboard-facing shape."""
    rows = db.query(Call).order_by(Call.created_at.desc()).all()
    catalog = []
    for call in rows:
        meta = dict(call.call_metadata or {})
        public_id = meta.get("public_call_id")
        if not public_id:
            continue

        summary = dict(meta.get("index_summary") or {})
        job = _latest_job(db, call.call_id)
        analyzed = bool(summary)
        item = {
            **summary,
            "call_id": public_id,
            "db_call_id": str(call.call_id),
            "domain": summary.get("domain") or meta.get("domain"),
            "accent": summary.get("accent") or meta.get("accent"),
            "duration": summary.get("duration") or call.duration_seconds,
            "analyzed": analyzed,
            "status": job.status if job else ("analyzed" if analyzed else "available"),
            "stage": job.stage if job else ("done" if analyzed else "uploaded"),
            "error": job.error if job else None,
            "job_id": str(job.job_id) if job else None,
            "pipeline": _pipeline_progress(job) if job else None,
        }
        catalog.append(item)
    return catalog


@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze_calls(payload: AnalyzeRequest, db: Session = Depends(get_db)):
    """Queue staged calls, reusing an active job when a request is repeated."""
    calls = {
        call.call_id: call
        for call in db.query(Call).filter(Call.call_id.in_(payload.call_ids)).all()
    }
    missing = [str(call_id) for call_id in payload.call_ids if call_id not in calls]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "call not found", "call_ids": missing},
        )

    jobs = []
    for call_id in payload.call_ids:
        call = calls[call_id]
        meta = dict(call.call_metadata or {})
        public_id = meta.get("public_call_id")
        if not public_id:
            raise HTTPException(status_code=409, detail=f"call {call_id} is not pipeline-managed")

        active = _latest_job(db, call_id)
        if active and active.status in ("queued", "processing"):
            jobs.append({
                "call_id": str(call_id),
                "public_call_id": public_id,
                "job_id": str(active.job_id),
                "status": active.status,
                "created": False,
            })
            continue

        if meta.get("index_summary") and not payload.force:
            jobs.append({
                "call_id": str(call_id),
                "public_call_id": public_id,
                "job_id": str(active.job_id) if active else None,
                "status": "already_analyzed",
                "created": False,
            })
            continue

        job = Job(call_id=call_id, status="queued", stage="uploaded")
        db.add(job)
        db.flush()
        jobs.append({
            "call_id": str(call_id),
            "public_call_id": public_id,
            "job_id": str(job.job_id),
            "status": job.status,
            "created": True,
        })

    db.commit()
    for item in jobs:
        if item["created"]:
            worker.enqueue(item["job_id"])
    return {"jobs": jobs}

@router.get("/{call_id}/status")
def get_call_status(call_id: str, db: Session = Depends(get_db)):
    try:
        db_call_id = uuid.UUID(call_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid call id")
    job = _latest_job(db, db_call_id)
    if not job:
        raise HTTPException(status_code=404, detail="call not found")
    call = db.query(Call).filter(Call.call_id == db_call_id).first()
    public_id = (call.call_metadata or {}).get("public_call_id") if call else None
    return {
        "call_id": call_id,
        "job_id": str(job.job_id),
        "public_call_id": public_id,
        "status": job.status,
        "stage": job.stage,
        "pipeline": _pipeline_progress(job),
        "error": job.error,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None
    }


@router.get("/{public_call_id}/evaluation-runs")
def get_evaluation_runs(
    public_call_id: str,
    db: Session = Depends(get_db),
):
    if not _call_by_public_id(db, public_call_id):
        raise HTTPException(status_code=404, detail="call not found")
    rows = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.public_call_id == public_call_id)
        .order_by(EvaluationRun.created_at.desc())
        .all()
    )
    return [
        {
            "evaluation_run_id": str(row.evaluation_run_id),
            "runtime_run_id": row.runtime_run_id,
            "evaluator_version": row.evaluator_version,
            "mode": row.mode,
            "status": row.status,
            "decision_sha256": row.decision_sha256,
            "attention_required": row.attention_required,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@router.post("/{public_call_id}/feedback", status_code=201)
def record_evaluation_feedback(
    public_call_id: str,
    payload: EvaluationFeedbackRequest,
    db: Session = Depends(get_db),
):
    call = _call_by_public_id(db, public_call_id)
    if not call:
        raise HTTPException(status_code=404, detail="call not found")
    run = _latest_v2_run(db, public_call_id)
    if not run or run.decision_sha256 != payload.decision_sha256:
        raise HTTPException(
            status_code=409,
            detail="feedback decision does not match the latest v2 run",
        )

    decision = (run.payload or {}).get("decision") or {}
    known_findings = {
        item.get("finding_id")
        for item in (
            (decision.get("triggered_findings") or [])
            + (decision.get("positive_findings") or [])
        )
    }
    if (
        payload.finding_id
        and payload.finding_id not in known_findings
    ):
        raise HTTPException(
            status_code=422,
            detail="feedback references an unknown finding",
        )
    action = decision.get("recommended_action") or {}
    if (
        payload.action_type
        and payload.action_type != action.get("action_type")
    ):
        raise HTTPException(
            status_code=422,
            detail="feedback references an unknown action",
        )
    if (
        payload.feedback_type == "approve_action"
        and not action.get("requires_human_approval")
    ):
        raise HTTPException(
            status_code=422,
            detail="this action does not require manager approval",
        )

    feedback = EvaluationFeedback(
        evaluation_run_id=run.evaluation_run_id,
        public_call_id=public_call_id,
        decision_sha256=payload.decision_sha256,
        feedback_type=payload.feedback_type,
        finding_id=payload.finding_id,
        action_type=payload.action_type,
        note=payload.note,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {
        "feedback_id": str(feedback.feedback_id),
        "public_call_id": public_call_id,
        "feedback_type": feedback.feedback_type,
        "created_at": feedback.created_at.isoformat(),
    }


@router.get("/{public_call_id}/feedback")
def get_evaluation_feedback(
    public_call_id: str,
    db: Session = Depends(get_db),
):
    if not _call_by_public_id(db, public_call_id):
        raise HTTPException(status_code=404, detail="call not found")
    rows = (
        db.query(EvaluationFeedback)
        .filter(
            EvaluationFeedback.public_call_id == public_call_id
        )
        .order_by(EvaluationFeedback.created_at.desc())
        .all()
    )
    return [
        {
            "feedback_id": str(row.feedback_id),
            "feedback_type": row.feedback_type,
            "finding_id": row.finding_id,
            "action_type": row.action_type,
            "note": row.note,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]

@router.post("/transcripts/ingest", status_code=201)
async def ingest_transcript(
    payload: dict,
    db: Session = Depends(get_db)
):
    call_id_str = payload.get("call_id")
    sentences = payload.get("sentences", [])

    inserted = 0
    for sentence in sentences:
        transcript = Transcript(
            call_id=uuid.UUID(call_id_str) if len(call_id_str) == 36 else None,
            source_call_id=call_id_str,
            turn_id=sentence.get("seq_id", 0),
            speaker=sentence.get("speaker", "").lower(),
            start_time=sentence.get("start"),
            end_time=sentence.get("end"),
            text=sentence.get("text"),
            avg_confidence=1.0,
            low_confidence=False
        )
        db.add(transcript)
        inserted += 1

    db.commit()

    return {
        "call_id": call_id_str,
        "inserted": inserted,
        "status": "complete"
    }

@router.post("/evaluate", status_code=201)
async def ingest_evaluation(
    payload: dict,
    db: Session = Depends(get_db)
):
    """
    Receives Llama/Aniket evaluation output and applies action routing logic.
    """
    call_id_str = payload.get("call_id")
    agent_id_str = payload.get("agent_id")

    # Get max escalation score from sentiment table
    sentiment_segments = db.query(SentimentSegment).filter(
        SentimentSegment.call_id == call_id_str
    ).all()

    scores = [s.escalation_score for s in sentiment_segments if s.escalation_score is not None]
    max_escalation = max(scores) if scores else None

    # Create evaluation record
    evaluation = Evaluation(
        call_id=uuid.UUID(call_id_str) if len(call_id_str) == 36 else None,
        agent_id=uuid.UUID(agent_id_str) if agent_id_str else None,
        overall_grade=payload.get("overall_grade"),
        weighted_score=payload.get("weighted_score"),
        scorecard=payload.get("scorecard"),
        compliance_flags=payload.get("compliance_flags"),
        escalation_risk=payload.get("escalation_risk"),
        llm_scored=payload.get("llm_scored", True)
    )

    # Apply action routing
    evaluation = apply_action_routing(evaluation, max_escalation)

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    return {
        "call_id": call_id_str,
        "evaluation_id": str(evaluation.evaluation_id),
        "overall_grade": evaluation.overall_grade,
        "escalation_flag": evaluation.escalation_flag,
        "coaching_required": evaluation.coaching_required,
        "manual_review_required": evaluation.manual_review_required,
        "status": "complete"
    }

@router.get("/{call_id}/flags")
def get_call_flags(call_id: str, db: Session = Depends(get_db)):
    """
    Returns the action flags for a call — useful for dashboard alerts.
    """
    try:
        evaluation = db.query(Evaluation).filter(
            Evaluation.call_id == uuid.UUID(call_id)
        ).first()
    except Exception:
        return {"error": "invalid call_id format"}

    if not evaluation:
        return {"error": "no evaluation found for this call"}

    return {
        "call_id": call_id,
        "overall_grade": evaluation.overall_grade,
        "escalation_flag": evaluation.escalation_flag,
        "coaching_required": evaluation.coaching_required,
        "manual_review_required": evaluation.manual_review_required,
        "escalation_risk": evaluation.escalation_risk,
        "weighted_score": float(evaluation.weighted_score) if evaluation.weighted_score else None
    }
