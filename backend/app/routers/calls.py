import uuid
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Call, Job

router = APIRouter(prefix="/calls", tags=["calls"])
REPO_ROOT = Path(__file__).resolve().parents[3]
SENTIMENT_OUTPUT_DIR = REPO_ROOT / "ml-services" / "outputs" / "backend" / "sentiment_calls_with_features"


def find_sentiment_file(call_id: str) -> Path | None:
    matches = list(SENTIMENT_OUTPUT_DIR.rglob(f"{call_id}_backend_sentiment_with_features.json"))
    if matches:
        return matches[0]
    return None

def process_job(job_id: str, call_id: str):
    # Pipeline workers plug in here
    # Rodrigo's diarization, Aniket's transcription etc
    print(f"[worker] Job {job_id} queued for call {call_id}")

@router.post("/ingest", status_code=201)
async def ingest_call(
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    call_date: str = Form(...),
    duration_seconds: int = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Create directory and save audio file safely
    os.makedirs("/data/audio/calls", exist_ok=True)
    safe_filename = os.path.basename(file.filename)
    audio_path = f"/data/audio/calls/{uuid.uuid4()}_{safe_filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create call record in database
    call = Call(
        agent_id=uuid.UUID(agent_id),
        audio_path=audio_path,
        call_date=datetime.fromisoformat(call_date),
        duration_seconds=duration_seconds
    )
    db.add(call)
    db.flush()

    # Create job record to track pipeline progress
    job = Job(call_id=call.call_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    # Queue background processing
    background_tasks.add_task(process_job, str(job.job_id), str(call.call_id))

    return {
        "job_id": str(job.job_id),
        "call_id": str(call.call_id),
        "status": "queued",
        "audio_path": audio_path,
        "created_at": job.created_at.isoformat()
    }




@router.get("/{call_id}/sentiment")
def get_call_sentiment(call_id: str):
    sentiment_file = find_sentiment_file(call_id)

    if not sentiment_file:
        raise HTTPException(
            status_code=404,
            detail=f"No backend sentiment file found for call_id: {call_id}"
        )

    with open(sentiment_file, "r", encoding="utf-8") as f:
        return json.load(f)

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


@router.get("/sentiment/available")
def list_available_sentiment_calls():
    if not SENTIMENT_OUTPUT_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sentiment output directory not found: {SENTIMENT_OUTPUT_DIR}"
        )

    calls = []

    for file_path in sorted(SENTIMENT_OUTPUT_DIR.rglob("*_backend_sentiment_with_features.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        call_summary = payload.get("call_summary", {})
        audio_summary = payload.get("audio_feature_summary", {})
        customer_trend = payload.get("customer_escalation_trend", {})
        speaker_summary = payload.get("speaker_audio_feature_summary", {})

        calls.append({
            "call_id": payload.get("call_id"),
            "domain": payload.get("domain"),
            "model_version": payload.get("model_version"),

            "risk_level": call_summary.get("risk_level"),
            "dominant_sentiment": call_summary.get("dominant_sentiment"),
            "dominant_emotion": call_summary.get("dominant_emotion"),
            "total_segments": call_summary.get("total_segments"),
            "successful_segments": call_summary.get("successful_segments"),
            "skipped_segments": call_summary.get("skipped_segments"),

            "has_audio_features": payload.get("has_audio_features"),
            "average_pitch_hz": audio_summary.get("average_pitch_hz"),
            "average_volume_db": audio_summary.get("average_volume_db"),
            "average_speech_rate_wpm": audio_summary.get("average_speech_rate_wpm"),
            "average_pause_ratio": audio_summary.get("average_pause_ratio"),

            "customer_escalation_trend": customer_trend.get("trend"),
            "customer_trend_delta": customer_trend.get("trend_delta"),

            "has_speaker_summary": bool(speaker_summary),
            "dashboard_series_count": len(payload.get("dashboard_audio_feature_series", [])),
        })

    return {
        "total_calls": len(calls),
        "calls": calls
    }