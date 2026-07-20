"""
The one place the backend reaches into the ml-services pipeline.

The evaluation/ and pipeline/ modules import each other by bare top-level
names (`import paths`, `from graph import ...`) and were only ever run with
those dirs on sys.path. This shim inserts them exactly once and is imported
LAZILY from inside the worker thread -- never at app import time -- so a plain
`import app.main` (health checks, artifact routes) never drags in torch,
langgraph, or faster-whisper.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]          # backend/app -> backend -> repo
_ML = _REPO / "ml-services"
# order matters: pipeline/evaluation/scripts hold the bare-name modules;
# _ML itself makes `import src.inference...` (the vendored acoustic pkg) resolve
_PATHS = [_ML / "pipeline", _ML / "evaluation", _ML / "scripts", _ML]

_installed = False


def install():
    """Idempotently put the ml-services module dirs on sys.path."""
    global _installed
    if _installed:
        return
    for p in _PATHS:
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)
    _installed = True


def get_process_call():
    """Return orchestrator.process_call (heavy imports; worker-thread only)."""
    install()
    from orchestrator import process_call
    return process_call
