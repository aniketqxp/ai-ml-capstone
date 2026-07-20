"""
Serves the frontend's static-fetch contract from Supabase Storage + the DB.

The React app fetches /calls_index.json, /calls/{id}.json,
/sentence_segments/{id}.json, /sentiment/{id}.json and /audio/{id}.mp3. These
were static files under frontend/public; here the same paths become API routes
so the Vercel-hosted SPA reads seeded and freshly-ingested calls identically.

JSON artifacts are proxied (small; keeps the API's CORS headers in front of the
Vercel origin); audio 307-redirects to the Supabase CDN, which owns Range +
bandwidth. calls_index is built live from the Call rows' index_summary.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Call
from app import storage

router = APIRouter(tags=["artifacts"])


@router.get("/calls_index.json")
def calls_index(db: Session = Depends(get_db)):
    """Dashboard list: every call's index_summary, newest first."""
    rows = db.query(Call).order_by(Call.created_at.desc()).all()
    index = [(c.call_metadata or {}).get("index_summary") for c in rows]
    return JSONResponse([s for s in index if s])


def _json_or_404(key):
    try:
        return JSONResponse(storage.stream_json(key))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="artifact not found")


@router.get("/calls/{call_id}.json")
def call_json(call_id: str):
    return _json_or_404(f"calls/{call_id}.json")


@router.get("/sentence_segments/{call_id}.json")
def sentence_segments(call_id: str):
    return _json_or_404(f"sentence_segments/{call_id}.json")


@router.get("/sentiment/{call_id}.json")
def sentiment(call_id: str):
    # optional overlay -> empty (not 404) so CallDetail degrades to plain turns
    try:
        return JSONResponse(storage.stream_json(f"sentiment/{call_id}.json"))
    except FileNotFoundError:
        return JSONResponse({"segments": []})


@router.get("/audio/{call_id}.mp3")
def audio(call_id: str):
    return RedirectResponse(
        storage.public_url(f"audio/{call_id}.mp3"), status_code=307)
