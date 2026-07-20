import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load the repo-root .env for local dev BEFORE importing anything that reads
# env at import time (database.py builds the engine from DATABASE_URL). On the
# Space there is no .env; the real environment (Space secrets) is used and
# always takes precedence -- load_dotenv never overrides existing vars.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import calls, sentiment, artifacts
from app import worker


def _boot_model_prewarm():
    """Pre-download the acoustic weights so the first job doesn't pay for it.
    Best-effort, off-thread; only when acoustic is enabled."""
    if os.environ.get("ENABLE_ACOUSTIC", "0").strip().lower() in ("", "0", "false", "no"):
        return
    try:
        from app.pipeline_bridge import install
        install()
        from acoustic import resolve_model_dir
        resolve_model_dir()
        print("[startup] acoustic weights ready")
    except Exception as e:
        print(f"[startup] acoustic pre-warm skipped: {e}")


@asynccontextmanager
async def lifespan(app):
    try:
        init_db()
    except Exception as e:
        print(f"[startup] init_db skipped (db unreachable?): {e}")
    worker.start()
    threading.Thread(target=_boot_model_prewarm, name="model-prewarm",
                     daemon=True).start()
    yield


app = FastAPI(
    title="AI/ML Capstone API",
    description="Fresh-call ingest, pipeline worker, and artifact serving",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: the Vercel frontend calls this API cross-origin. FRONTEND_ORIGIN pins
# the allowed origin(s) in production (comma-separated); "*" for local dev.
# Credentials are disabled under wildcard (the spec forbids "*" + credentials);
# the frontend uses plain fetch with no cookies, so this is fine either way.
_origins = os.environ.get("FRONTEND_ORIGIN", "*")
_wild = _origins.strip() == "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _wild else [o.strip() for o in _origins.split(",")],
    allow_credentials=not _wild,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calls.router)
app.include_router(sentiment.router)
app.include_router(artifacts.router)


@app.get("/")
def read_root():
    return {"status": "online", "service": "backend-orchestration"}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "capstone_api"}
