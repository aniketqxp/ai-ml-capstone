"""
Supabase Storage access over plain REST (no SDK dependency).

Artifacts (call JSON, mp3, sentence_segments, sentiment, raw transcript, and
the original upload for restart recovery) live in one PUBLIC bucket so reads
need no auth and the Supabase CDN serves audio Range requests directly. Writes
use the service-role key.

Layout in the bucket (keys):
  calls/{id}.json  audio/{id}.mp3  sentence_segments/{id}.json
  sentiment/{id}.json  transcripts/{id}.json  uploads/{id}/<name>
"""
import os
import json
import mimetypes
from pathlib import Path

import requests

BUCKET = os.environ.get("SUPABASE_BUCKET", "call-artifacts")
_TIMEOUT = 30


class StorageError(RuntimeError):
    pass


def _base():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        raise StorageError("SUPABASE_URL is not set")
    return url


def _service_key():
    # secret/service-role key preferred (storage writes bypass RLS); falls back
    # to SUPABASE_KEY, which only authorizes writes if the bucket grants anon
    # insert. Reads from a public bucket work with either.
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
    if not key:
        raise StorageError("neither SUPABASE_SERVICE_KEY nor SUPABASE_KEY is set")
    return key


def is_configured():
    return bool(os.environ.get("SUPABASE_URL")
                and (os.environ.get("SUPABASE_SERVICE_KEY")
                     or os.environ.get("SUPABASE_KEY")))


def object_url(key):
    """Authenticated object endpoint (used for writes)."""
    return f"{_base()}/storage/v1/object/{BUCKET}/{key}"


def public_url(key):
    """Public read URL served by the Supabase CDN (Range-capable)."""
    return f"{_base()}/storage/v1/object/public/{BUCKET}/{key}"


# ── writes ──────────────────────────────────────────────────────────────────

def upload_bytes(key, data, content_type="application/octet-stream"):
    # `apikey` header (not just Authorization: Bearer) is required for the new
    # sb_secret_ key format -- the object endpoint parses Bearer tokens as JWTs
    # and rejects the non-JWT key with "Invalid Compact JWS". Sending both is
    # compatible with legacy JWT service_role keys too.
    k = _service_key()
    headers = {
        "apikey": k,
        "Authorization": f"Bearer {k}",
        "Content-Type": content_type,
        "x-upsert": "true",          # overwrite -> idempotent re-runs
    }
    r = requests.post(object_url(key), headers=headers, data=data, timeout=_TIMEOUT)
    if r.status_code not in (200, 201):
        raise StorageError(f"upload {key} failed: {r.status_code} {r.text[:200]}")
    return key


def upload_file(key, path, content_type=None):
    if content_type is None:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return upload_bytes(key, f.read(), content_type)


def upload_json(key, obj):
    return upload_bytes(key, json.dumps(obj).encode("utf-8"), "application/json")


# ── reads (public bucket; small local cache since artifacts are immutable) ──

_CACHE_DIR = Path(os.environ.get("CAPSTONE_DATA_ROOT", "/tmp")) / "_storage_cache"


def _cache_path(key):
    return _CACHE_DIR / key


def exists(key):
    try:
        r = requests.head(public_url(key), timeout=_TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False


def download_bytes(key, use_cache=True):
    cp = _cache_path(key)
    if use_cache and cp.exists():
        return cp.read_bytes()
    r = requests.get(public_url(key), timeout=_TIMEOUT)
    missing = r.status_code == 404
    if r.status_code == 400:
        try:
            payload = r.json()
            missing = (str(payload.get("statusCode")) == "404"
                       or payload.get("error") == "not_found")
        except ValueError:
            pass
    if missing:
        raise FileNotFoundError(key)
    if r.status_code != 200:
        raise StorageError(f"download {key} failed: {r.status_code}")
    if use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(r.content)
    return r.content


def stream_json(key, use_cache=True):
    return json.loads(download_bytes(key, use_cache=use_cache).decode("utf-8"))
