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
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
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

@router.get("/evaluation/{call_id}.json")
def evaluation(call_id: str):
    return _json_or_404(f"evaluation/{call_id}.json")


def _range_file(path, range_header):
    size = path.stat().st_size
    try:
        value = range_header.removeprefix("bytes=").split(",", 1)[0]
        start_text, end_text = value.split("-", 1)
        start = int(start_text) if start_text else 0
        end = min(int(end_text), size - 1) if end_text else size - 1
        if start < 0 or start > end or start >= size:
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=416, detail="invalid audio range")

    def chunks():
        remaining = end - start + 1
        with path.open("rb") as source:
            source.seek(start)
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(chunks(), status_code=206, media_type="audio/mpeg", headers={
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Content-Length": str(end - start + 1),
    })


@router.get("/audio/{call_id}.mp3")
def audio(call_id: str, request: Request):
    local_audio = storage.local_path(f"audio/{call_id}.mp3")
    if local_audio:
        range_header = request.headers.get("range")
        if range_header:
            return _range_file(local_audio, range_header)
        return FileResponse(local_audio, media_type="audio/mpeg")
    if not storage.is_configured():
        raise HTTPException(status_code=404, detail="audio artifact not found")
    return RedirectResponse(
        storage.public_url(f"audio/{call_id}.mp3"), status_code=307)
