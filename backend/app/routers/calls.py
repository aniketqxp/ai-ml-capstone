import uuid
import os
import shutil
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Call, Job, Transcript, Evaluation, SentimentSegment

router = APIRouter(prefix="/calls", tags=["calls"])

def process_job(job_id: str, call_id: str):
    print(f"[worker] Job {job_id} queued for call {call_id}")

def apply_action_routing(evaluation: Evaluation, max_escalation_score: float = None):
    """
    Task 4.3 — Deterministic action routing logic.
    Reads evaluation scores and sets flags automatically.
    """
    # Escalation risk from Llama (0-10 scale)
    if evaluation.escalation_risk is not None:
        if evaluation.escalation_risk >= 7:
            evaluation.escalation_flag = True
            evaluation.coaching_required = True
        elif evaluation.escalation_risk >= 4:
            evaluation.coaching_required = True

    # Grade-based routing
    if evaluation.overall_grade == "F":
        evaluation.manual_review_required = True
        evaluation.coaching_required = True

    # Sentiment-based escalation (Clara's Wav2Vec2 score 0.0-1.0)
    if max_escalation_score is not None:
        if max_escalation_score >= 0.7:
            evaluation.escalation_flag = True
            evaluation.coaching_required = True
        elif max_escalation_score >= 0.4:
            evaluation.coaching_required = True

    return evaluation

@router.post("/ingest", status_code=201)
async def ingest_call(
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    call_date: str = Form(...),
    duration_seconds: int = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    os.makedirs("/data/audio/calls", exist_ok=True)
    safe_filename = os.path.basename(file.filename)
    audio_path = f"/data/audio/calls/{uuid.uuid4()}_{safe_filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    call = Call(
        agent_id=uuid.UUID(agent_id),
        audio_path=audio_path,
        call_date=datetime.fromisoformat(call_date),
        duration_seconds=duration_seconds
    )
    db.add(call)
    db.flush()

    job = Job(call_id=call.call_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(process_job, str(job.job_id), str(call.call_id))

    return {
        "job_id": str(job.job_id),
        "call_id": str(call.call_id),
        "status": "queued",
        "audio_path": audio_path,
        "created_at": job.created_at.isoformat()
    }

@router.get("/{call_id}/status")
def get_call_status(call_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.call_id == uuid.UUID(call_id)).first()
    if not job:
        return {"error": "call not found"}
    return {
        "call_id": call_id,
        "job_id": str(job.job_id),
        "status": job.status,
        "stage": job.stage,
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