"""Minimal .env loader (no external dependency).

The actual loading lives in paths.py (which also runs it at import time so
CAPSTONE_* path overrides in .env take effect before any path constant is
read). This re-export keeps the existing call sites working unchanged.
"""
from paths import load_env  # noqa: F401
