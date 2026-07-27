import uuid
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Call, Job, Transcript, Evaluation, SentimentSegment

router = APIRouter(prefix="/calls", tags=["calls"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTIMENT_OUTPUT_DIR = REPO_ROOT / "ml-services" / "outputs" / "backend" / "sentiment_calls_with_features"

def find_sentiment_file(call_id: str) -> Path | None:
    matches = list(SENTIMENT_OUTPUT_DIR.rglob(f"{call_id}_backend_sentiment_with_features.json"))
    if matches:
        return matches[0]
    return None

def process_job(job_id: str, call_id: str):
    print(f"[worker] Job {job_id} queued for call {call_id}")

def apply_action_routing(evaluation: Evaluation, max_escalation_score: float = None):
    if evaluation.escalation_risk is not None:
        if evaluation.escalation_risk >= 7:
            evaluation.escalation_flag = True
            evaluation.coaching_required = True
        elif evaluation.escalation_risk >= 4:
            evaluation.coaching_required = True

    if evaluation.overall_grade == "F":
        evaluation.manual_review_required = True
        evaluation.coaching_required = True

    if max_escalation_score is not None:
        if max_escalation_score >= 0.7:
            evaluation.escalation_flag = True
            evaluation.coaching_required = True
        elif max_escalation_score >= 0.5:
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

@router.get("/summary")
def get_calls_summary(db: Session = Depends(get_db)):
    all_calls = db.query(Call).all()

    sentiment_data = db.query(
        SentimentSegment.call_id,
        SentimentSegment.domain,
        func.max(SentimentSegment.escalation_score).label("max_escalation"),
        func.avg(SentimentSegment.escalation_score).label("avg_escalation"),
        func.count(SentimentSegment.id).label("total_segments")
    ).group_by(
        SentimentSegment.call_id,
        SentimentSegment.domain
    ).all()
    sentiment_map = {row.call_id: row for row in sentiment_data}

    jobs = db.query(Job).all()
    job_map = {str(j.call_id): j for j in jobs}

    evaluations = db.query(Evaluation).all()
    eval_map = {str(e.call_id): e for e in evaluations if e.call_id}

    calls_list = []
    total = len(all_calls)
    escalated = 0
    coaching = 0
    clean = 0
    processing = 0
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    domain_counts = {}

    for call in all_calls:
        call_id_str = str(call.call_id)
        sentiment_row = sentiment_map.get(call_id_str)
        job = job_map.get(call_id_str)
        eval_obj = eval_map.get(call_id_str)

        if not sentiment_row:
            pipeline_status = job.status if job else "unknown"
            processing += 1
            calls_list.append({
                "call_id": call_id_str,
                "domain": None,
                "grade": None,
                "dominant_sentiment": None,
                "dominant_emotion": None,
                "max_escalation_score": None,
                "avg_escalation_score": None,
                "total_segments": None,
                "escalation_flag": False,
                "coaching_required": False,
                "manual_review_required": False,
                "action": "Processing",
                "pipeline_status": pipeline_status
            })
            continue

        max_esc = round(float(sentiment_row.max_escalation), 4) if sentiment_row.max_escalation else None
        avg_esc = round(float(sentiment_row.avg_escalation), 4) if sentiment_row.avg_escalation else None

        successful = db.query(SentimentSegment).filter(
            SentimentSegment.call_id == call_id_str,
            SentimentSegment.processing_status == "success"
        ).all()

        sentiments = [s.sentiment for s in successful if s.sentiment]
        dominant_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else None

        emotions = [s.dominant_emotion for s in successful if s.dominant_emotion]
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else None

        grade = eval_obj.overall_grade if eval_obj else None
        escalation_flag = eval_obj.escalation_flag if eval_obj else False
        coaching_flag = eval_obj.coaching_required if eval_obj else False
        manual_review = eval_obj.manual_review_required if eval_obj else False

        if escalation_flag:
            action = "Escalate"
            escalated += 1
        elif coaching_flag:
            action = "Coaching"
            coaching += 1
        else:
            action = "No action"
            clean += 1

        if grade and grade in grade_counts:
            grade_counts[grade] += 1

        domain = sentiment_row.domain or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        calls_list.append({
            "call_id": call_id_str,
            "domain": domain,
            "grade": grade,
            "dominant_sentiment": dominant_sentiment,
            "dominant_emotion": dominant_emotion,
            "max_escalation_score": max_esc,
            "avg_escalation_score": avg_esc,
            "total_segments": sentiment_row.total_segments,
            "escalation_flag": escalation_flag,
            "coaching_required": coaching_flag,
            "manual_review_required": manual_review,
            "action": action,
            "pipeline_status": "complete"
        })

    return {
        "total_calls": total,
        "escalated": escalated,
        "coaching_required": coaching,
        "clean": clean,
        "processing": processing,
        "grade_distribution": grade_counts,
        "domain_distribution": domain_counts,
        "calls": calls_list
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
    call_id_str = payload.get("call_id")
    agent_id_str = payload.get("agent_id")

    sentiment_segments = db.query(SentimentSegment).filter(
        SentimentSegment.call_id == call_id_str
    ).all()

    scores = [s.escalation_score for s in sentiment_segments if s.escalation_score is not None]
    max_escalation = max(scores) if scores else None

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