import uuid
import os
import shutil
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Call, Job, Transcript, Evaluation, SentimentSegment
from app import worker, storage
from app.routing import apply_action_routing

router = APIRouter(prefix="/calls", tags=["calls"])


class AnalyzeRequest(BaseModel):
    call_ids: list[uuid.UUID] = Field(min_length=1, max_length=10)


def _latest_job(db, call_id):
    return (
        db.query(Job)
        .filter(Job.call_id == call_id)
        .order_by(Job.created_at.desc())
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
    from audio_io import prepare_channels, AudioError

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

        if meta.get("index_summary"):
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
        "error": job.error,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None
    }

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
