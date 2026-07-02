"""Minimal .env loader (no external dependency)."""
import os

def load_env(path=None):
    if path is None:
        path = r"d:\Desktop\Main\Projects\ai-ml-capstone\.env"
    vals = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
                os.environ.setdefault(k.strip(), v.strip())
    return vals
