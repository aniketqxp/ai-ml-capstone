from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import SentimentSegment, CallAudioSummary

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

@router.post("/ingest", status_code=201)
async def ingest_sentiment(
    payload: dict,
    db: Session = Depends(get_db)
):
    call_id_str = payload.get("call_id")
    domain = payload.get("domain")
    model_version = payload.get("model_version")
    has_audio_features = payload.get("has_audio_features", False)
    audio_feature_version = payload.get("audio_feature_version")
    segments = payload.get("segments", [])
    call_summary = payload.get("call_summary")
    audio_feature_match_summary = payload.get("audio_feature_match_summary")
    dashboard_audio_feature_series = payload.get("dashboard_audio_feature_series")

    # Save or update call-level audio summary
    existing_summary = db.query(CallAudioSummary).filter(
        CallAudioSummary.call_id == call_id_str
    ).first()

    if not existing_summary:
        summary_row = CallAudioSummary(
            call_id=call_id_str,
            domain=domain,
            model_version=model_version,
            has_audio_features=has_audio_features,
            audio_feature_version=audio_feature_version,
            audio_feature_match_summary=audio_feature_match_summary,
            dashboard_audio_feature_series=dashboard_audio_feature_series,
            call_summary=call_summary
        )
        db.add(summary_row)

    # Save segment-level data
    inserted = 0
    skipped = 0

    for segment in segments:
        existing = db.query(SentimentSegment).filter(
            SentimentSegment.segment_key == segment.get("segment_key")
        ).first()

        if existing:
            skipped += 1
            continue

        audio_features = segment.get("audio_features")

        sentiment_row = SentimentSegment(
            call_id=call_id_str,
            segment_index=segment.get("segment_index"),
            segment_key=segment.get("segment_key"),
            seq_id=segment.get("seq_id"),
            speaker=segment.get("speaker"),
            start_time=segment.get("start_time"),
            end_time=segment.get("end_time"),
            text=segment.get("text"),
            sentiment=segment.get("sentiment"),
            dominant_emotion=segment.get("dominant_emotion"),
            escalation_score=segment.get("escalation_score"),
            processing_status=segment.get("processing_status"),
            audio_features=audio_features,
            has_audio_features=audio_features is not None,
            audio_feature_version=audio_feature_version,
            domain=domain,
            model_version=model_version
        )
        db.add(sentiment_row)
        inserted += 1

    db.commit()

    return {
        "call_id": call_id_str,
        "inserted": inserted,
        "skipped_duplicates": skipped,
        "has_audio_features": has_audio_features,
        "status": "complete"
    }

@router.get("/{call_id}/summary")
def get_sentiment_summary(call_id: str, db: Session = Depends(get_db)):
    segments = db.query(SentimentSegment).filter(
        SentimentSegment.call_id == call_id
    ).all()

    if not segments:
        return {"error": "no sentiment data found for this call"}

    successful = [s for s in segments if s.processing_status == "success"]
    skipped = [s for s in segments if s.processing_status == "skipped_too_short"]

    sentiments = [s.sentiment for s in successful if s.sentiment]
    dominant_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else None

    emotions = [s.dominant_emotion for s in successful if s.dominant_emotion]
    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else None

    scores = [s.escalation_score for s in successful if s.escalation_score is not None]
    avg_escalation = round(sum(scores) / len(scores), 4) if scores else None
    max_escalation = round(max(scores), 4) if scores else None

    return {
        "call_id": call_id,
        "total_segments": len(segments),
        "successful_segments": len(successful),
        "skipped_segments": len(skipped),
        "dominant_sentiment": dominant_sentiment,
        "dominant_emotion": dominant_emotion,
        "average_escalation_score": avg_escalation,
        "max_escalation_score": max_escalation
    }

@router.get("/{call_id}/dashboard")
def get_dashboard_series(call_id: str, db: Session = Depends(get_db)):
    summary = db.query(CallAudioSummary).filter(
        CallAudioSummary.call_id == call_id
    ).first()

    if not summary:
        return {"error": "no audio summary found for this call"}

    return {
        "call_id": call_id,
        "domain": summary.domain,
        "has_audio_features": summary.has_audio_features,
        "call_summary": summary.call_summary,
        "dashboard_audio_feature_series": summary.dashboard_audio_feature_series
    }