import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Call, Job

router = APIRouter(prefix="/calls", tags=["calls"])

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
    # Save audio file path
    audio_path = f"/data/audio/calls/{uuid.uuid4()}_{file.filename}"

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