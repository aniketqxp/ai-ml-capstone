import uuid
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import SentimentSegment

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

@router.post("/ingest", status_code=201)
async def ingest_sentiment(
    payload: dict,
    db: Session = Depends(get_db)
):
    call_id_str = payload.get("call_id")
    domain = payload.get("domain")
    model_version = payload.get("model_version")
    segments = payload.get("segments", [])

    inserted = 0
    skipped = 0

    for segment in segments:
        # Check if this segment_key already exists to avoid duplicates
        existing = db.query(SentimentSegment).filter(
            SentimentSegment.segment_key == segment.get("segment_key")
        ).first()

        if existing:
            skipped += 1
            continue

        sentiment_row = SentimentSegment(
            call_id=call_id_str,
            segment_index=segment.get("segment_index"),
            segment_key=segment.get("segment_key"),
            seq_id=segment.get("seq_id"),
            sentiment=segment.get("sentiment"),
            dominant_emotion=segment.get("dominant_emotion"),
            escalation_score=segment.get("escalation_score"),
            processing_status=segment.get("processing_status"),
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

    # Calculate dominant sentiment
    sentiments = [s.sentiment for s in successful if s.sentiment]
    dominant_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else None

    # Calculate dominant emotion
    emotions = [s.dominant_emotion for s in successful if s.dominant_emotion]
    dominant_emotion = max(set(emotions), key=emotions.count) if emotions else None

    # Calculate average escalation score
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