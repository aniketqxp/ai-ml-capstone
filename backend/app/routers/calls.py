import uuid
import os
import shutil
import json
import wave
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Agent, Call, Job, Transcript, Evaluation, SentimentSegment
from app import worker, storage
from app.routing import apply_action_routing

router = APIRouter(prefix="/calls", tags=["calls"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTIMENT_OUTPUT_DIR = (
    REPO_ROOT
    / "ml-services"
    / "outputs"
    / "backend"
    / "sentiment_calls_with_features"
)


def find_sentiment_file(call_id: str):
    """Locate the generated backend sentiment payload for a public call ID."""
    if not SENTIMENT_OUTPUT_DIR.exists():
        return None

    matches = list(
        SENTIMENT_OUTPUT_DIR.rglob(
            f"{call_id}_backend_sentiment_with_features.json"
        )
    )
    return matches[0] if matches else None



class AnalyzeRequest(BaseModel):
    call_ids: list[uuid.UUID] = Field(min_length=1, max_length=10)


def _latest_job(db, call_id):
    return (
        db.query(Job)
        .filter(Job.call_id == call_id)
        .order_by(Job.created_at.desc())
        .first()
    )


def _artifact_dashboard_summary(public_id):
    """Build missing catalog fields from the published analysis artifacts.

    Older seeded rows only stored identity fields in ``index_summary``. The
    detail artifacts are authoritative, so use them to keep the dashboard
    useful without requiring a database reseed.
    """
    result = {}

    try:
        evaluation = storage.stream_json(f"evaluation/{public_id}.json")
    except (FileNotFoundError, storage.StorageError, ValueError):
        evaluation = {}

    compliance = evaluation.get("compliance") or {}
    compliance_results = [
        value.get("passed")
        for value in compliance.values()
        if isinstance(value, dict) and "passed" in value
    ]
    if compliance_results:
        result["compliance_passed"] = sum(
            value is True for value in compliance_results
        )
        result["compliance_applicable"] = sum(
            value is not None for value in compliance_results
        )

    workflow = evaluation.get("workflow") or {}
    steps = workflow.get("expected_steps") or []
    if steps:
        result["workflow_met"] = sum(
            step.get("met") is True for step in steps
        )
        result["workflow_total"] = len(steps)
    if workflow.get("subject"):
        result["subject"] = workflow["subject"]

    escalation = evaluation.get("escalation") or {}
    if escalation.get("risk_level"):
        result["risk_level"] = escalation["risk_level"]

    metadata = evaluation.get("metadata") or {}
    if metadata.get("duration_seconds") is not None:
        result["duration"] = metadata["duration_seconds"]

    try:
        sentiment = storage.stream_json(f"sentiment/{public_id}.json")
    except (FileNotFoundError, storage.StorageError, ValueError):
        sentiment = {}

    sentiment_summary = sentiment.get("call_summary") or {}
    audio_summary = sentiment.get("audio_feature_summary") or {}
    review = sentiment.get("manager_review_recommendation") or {}
    if sentiment:
        result.update({
            "dominant_sentiment": sentiment_summary.get("dominant_sentiment"),
            "dominant_emotion": sentiment_summary.get("dominant_emotion"),
            "average_escalation_score": sentiment_summary.get(
                "average_escalation_score"
            ),
            "has_audio_features": bool(
                sentiment.get("dashboard_audio_feature_series")
                or audio_summary.get("total_segments_with_audio_features")
            ),
            "manager_review_required": review.get("review_required"),
        })

    return {key: value for key, value in result.items() if value is not None}


def _incoming_dir(public_id):
    """Local scratch dir for this call's channel wavs (under CAPSTONE_DATA_ROOT)."""
    from app.pipeline_bridge import install
    install()
    import paths
    d = paths.DATA_ROOT / "_incoming" / public_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _upload_agent(db, agent_id=None):
    if agent_id:
        try:
            parsed_id = uuid.UUID(agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid agent_id") from exc
        agent = db.query(Agent).filter(Agent.agent_id == parsed_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="agent not found")
        return agent

    agent = db.query(Agent).filter(Agent.name == "Dashboard Upload").first()
    if not agent:
        agent = Agent(name="Dashboard Upload", team="self-service")
        db.add(agent)
        db.flush()
    return agent


def _staged_duration(public_id):
    audio_path = storage.local_path(f"uploads/{public_id}/agent.wav")
    if not audio_path:
        return None
    try:
        with wave.open(str(audio_path), "rb") as audio:
            return round(audio.getnframes() / audio.getframerate(), 2)
    except (OSError, wave.Error, ZeroDivisionError):
        return None

@router.post("/ingest", status_code=201)
async def ingest_call(
    agent_id: str = Form(None),
    domain: str = Form(...),
    accent: str = Form("en-US_General"),
    call_date: str = Form(None),
    duration_seconds: int = Form(None),
    file: UploadFile = File(...),
    file2: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Stage one fresh call for later analysis. Stereo and paired-channel files
    preserve speaker attribution; ordinary mono recordings are accepted with
    limited speaker attribution.
    """
    from app.pipeline_bridge import install
    install()
    from audio_io import prepare_channels, AudioError

    normalized_domain = "".join(
        character if character.isalnum() else "_"
        for character in domain.strip().lower()
    ).strip("_") or "general"
    public_id = f"{normalized_domain}_{datetime.utcnow():%Y%m%d}_{uuid.uuid4().hex[:8]}"
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

    detected_duration = duration_seconds or _staged_duration(public_id)

    upload_agent = _upload_agent(db, agent_id)
    call = Call(
        agent_id=upload_agent.agent_id,
        audio_path=f"uploads/{public_id}/",
        call_date=datetime.fromisoformat(call_date) if call_date else datetime.utcnow(),
        duration_seconds=int(detected_duration) if detected_duration else None,
        call_metadata={"public_call_id": public_id,
                       "domain": normalized_domain, "accent": accent,
                       "source_filename": file.filename},
    )
    db.add(call)
    db.flush()

    db.commit()

    return {
        "job_id": None,
        "call_id": str(call.call_id),
        "public_call_id": public_id,
        "status": "available",
        "created_at": call.created_at.isoformat() if call.created_at else None,
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

        summary = {
            **dict(meta.get("index_summary") or {}),
            **_artifact_dashboard_summary(public_id),
        }
        job = _latest_job(db, call.call_id)
        analyzed = bool(summary)
        item = {
            **summary,
            "call_id": public_id,
            "db_call_id": str(call.call_id),
            "domain": summary.get("domain") or meta.get("domain"),
            "accent": summary.get("accent") or meta.get("accent"),
            "duration": (
                summary.get("duration")
                or call.duration_seconds
                or _staged_duration(public_id)
            ),
            "analyzed": analyzed,
            "status": job.status if job else ("analyzed" if analyzed else "available"),
            "stage": job.stage if job else ("done" if analyzed else "uploaded"),
            "error": job.error if job else None,
            "job_id": str(job.job_id) if job else None,
            "progress_percent": job.progress_percent if job else (100 if analyzed else 0),
            "progress_current": job.progress_current if job else None,
            "progress_total": job.progress_total if job else None,
            "progress_message": job.progress_message if job else None,
            "estimated_seconds_remaining": job.estimated_seconds_remaining if job else None,
            "cancel_requested": bool(job.cancel_requested) if job else False,
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

        job = Job(call_id=call_id, status="queued", stage="uploaded",
                  progress_percent=0, progress_message="Waiting in queue")
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
        "progress_percent": job.progress_percent,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "progress_message": job.progress_message,
        "estimated_seconds_remaining": job.estimated_seconds_remaining,
        "cancel_requested": bool(job.cancel_requested),
        "updated_at": job.updated_at.isoformat() if job.updated_at else None
    }


@router.post("/{call_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
def cancel_call(call_id: str, db: Session = Depends(get_db)):
    try:
        db_call_id = uuid.UUID(call_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid call id") from exc
    job = _latest_job(db, db_call_id)
    if not job:
        raise HTTPException(status_code=404, detail="call not found")
    if job.status not in ("queued", "processing"):
        raise HTTPException(status_code=409, detail=f"job is already {job.status}")
    job.cancel_requested = True
    job.progress_message = "Cancellation requested"
    job.updated_at = datetime.utcnow()
    if job.status == "queued":
        job.status = "cancelled"
        job.stage = "cancelled"
        job.estimated_seconds_remaining = 0
    db.commit()
    return {"call_id": call_id, "job_id": str(job.job_id),
            "status": job.status, "cancel_requested": True}

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

@router.get("/sentiment/available")
def list_available_sentiment_calls():
    """
    Return a dashboard-friendly summary of all generated sentiment payloads.
    """
    if not SENTIMENT_OUTPUT_DIR.exists():
        return {
            "total_calls": 0,
            "calls": [],
            "output_directory": str(SENTIMENT_OUTPUT_DIR),
        }

    calls = []

    for file_path in sorted(
        SENTIMENT_OUTPUT_DIR.rglob(
            "*_backend_sentiment_with_features.json"
        )
    ):
        try:
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[sentiment] Skipping invalid file {file_path}: {exc}")
            continue

        call_summary = payload.get("call_summary") or {}
        audio_summary = payload.get("audio_feature_summary") or {}
        customer_trend = payload.get("customer_escalation_trend") or {}
        speaker_summary = payload.get("speaker_audio_feature_summary") or {}

        calls.append(
            {
                "call_id": payload.get("call_id"),
                "domain": payload.get("domain"),
                "model_version": payload.get("model_version"),
                "risk_level": call_summary.get("risk_level"),
                "dominant_sentiment": call_summary.get(
                    "dominant_sentiment"
                ),
                "dominant_emotion": call_summary.get(
                    "dominant_emotion"
                ),
                "total_segments": call_summary.get("total_segments"),
                "successful_segments": call_summary.get(
                    "successful_segments"
                ),
                "skipped_segments": call_summary.get(
                    "skipped_segments"
                ),
                "has_audio_features": payload.get(
                    "has_audio_features"
                ),
                "average_pitch_hz": audio_summary.get(
                    "average_pitch_hz"
                ),
                "average_volume_db": audio_summary.get(
                    "average_volume_db"
                ),
                "average_speech_rate_wpm": audio_summary.get(
                    "average_speech_rate_wpm"
                ),
                "average_pause_ratio": audio_summary.get(
                    "average_pause_ratio"
                ),
                "customer_escalation_trend": customer_trend.get(
                    "trend"
                ),
                "customer_trend_delta": customer_trend.get(
                    "trend_delta"
                ),
                "has_speaker_summary": bool(speaker_summary),
                "dashboard_series_count": len(
                    payload.get("dashboard_audio_feature_series") or []
                ),
            }
        )

    return {
        "total_calls": len(calls),
        "calls": calls,
    }


@router.get("/{call_id}/sentiment")
def get_call_sentiment(call_id: str):
    """
    Return the full sentiment and acoustic-feature payload for one call.
    """
    sentiment_file = find_sentiment_file(call_id)

    if sentiment_file is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No backend sentiment file found for call_id: "
                f"{call_id}"
            ),
        )

    try:
        with sentiment_file.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid sentiment JSON for call {call_id}: {exc}",
        ) from exc
