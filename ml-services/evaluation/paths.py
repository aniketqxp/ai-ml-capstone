"""Central path configuration for the ml-services pipeline.

Every location the pipeline reads or writes derives from two roots:

  REPO_ROOT  -- the repository checkout, derived from this file's location.
  DATA_ROOT  -- the data directory; env-overridable (CAPSTONE_DATA_ROOT) so a
                hosted container can point it at a writable scratch volume.
                Defaults to {REPO_ROOT}/data, the local dev layout.

This module also loads .env (repo root) into os.environ at import time,
via setdefault -- real environment variables always win over .env entries.
Doing the load here (rather than in env_util) guarantees CAPSTONE_* keys in
.env are honored regardless of module import order; env_util.load_env stays
as a re-export for existing callers.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(os.environ.get("CAPSTONE_ENV_FILE", REPO_ROOT / ".env"))


def load_env(path=None):
    """Load KEY=VALUE lines from a .env file into os.environ (setdefault)."""
    path = Path(path) if path else ENV_FILE
    vals = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
            os.environ.setdefault(k.strip(), v.strip())
    return vals


load_env()

DATA_ROOT = Path(os.environ.get("CAPSTONE_DATA_ROOT", str(REPO_ROOT / "data")))

NA_TESTSET = DATA_ROOT / "na_testset"
MANIFEST = NA_TESTSET / "manifest.json"
# Registry for calls ingested at runtime (fresh uploads); merged over MANIFEST
# by assemble.load_manifest so every downstream consumer sees them.
RUNTIME_MANIFEST = NA_TESTSET / "manifest_runtime.json"

SENTIMENT_ROOT = DATA_ROOT / "sentiment"
SENTENCE_SEG_ROOT = DATA_ROOT / "sentence_segments"

EVAL_RESULTS = Path(os.environ.get(
    "CAPSTONE_EVAL_RESULTS", str(Path(__file__).resolve().parent / "results")))

FRONTEND_ROOT = REPO_ROOT / "frontend"
FRONTEND_PUBLIC = Path(os.environ.get(
    "CAPSTONE_FRONTEND_PUBLIC", str(FRONTEND_ROOT / "public")))
