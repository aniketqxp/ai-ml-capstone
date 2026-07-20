# Integration Handoff — Hosted Call-QA Demo Site

This document lets a fresh agent (or human) continue the hosting-integration work
without prior context. It is self-contained; read it top to bottom.

Branch: `feat/transcription-eval-pipeline`. Push directly (clean push, **never**
open a PR on this repo). Commit style: `feat/fix/docs/chore:` prefixes, stoic
technical descriptions, no references to people/tasks, no `git add .` (stage
selectively).

---

## 1. What we are building

Package the existing offline pipeline into a **hosted site for a live demo**.
The pipeline (already working offline) is: per-channel faster-whisper
transcription → sentence segmentation → gated wav2vec2 acoustic sentiment →
LangGraph LLM evaluation (compliance/quality/escalation, rubric v0.4.1, with
deterministic acoustic-text fusion) → export to the React dashboard + call
analysis page.

### The demo UX (this is the target — build toward it)

1. A **catalog** of calls is shown. Most are already analyzed (seeded in
   advance). A few are **staged but not yet analyzed**.
2. The user **selects** one or more un-analyzed calls and clicks **Analyze**.
3. Each selected call runs the full pipeline **verbosely** — a per-call
   progress bar/stepper shows how far along it is
   (`transcribing → segmenting → acoustic → evaluating → exporting`).
4. On completion the call joins the dashboard; clicking it opens the existing
   call-analysis page (audio player, chapter strip, compliance scorecard,
   sentiment overlay).

For the demo, **~3 fresh calls** are analyzed live (keep them SHORT — see
§7 timing). Everything else is pre-seeded. This is NOT a file-upload form; it is
catalog → select → analyze. (A real file-upload ingest endpoint also exists and
works, but the demo uses the catalog flow.)

Neither the backend branch nor the sentiment branch is authoritative — they are
inputs folded into this integration. Foundation first; analysis-logic and
UI-polish improvements are a later iteration.

---

## 2. Architecture (split hosting)

- **Hugging Face Spaces (Docker, free 2-vCPU/16GB)** runs the **FastAPI API +
  one background worker thread**. No SPA is served from here.
- **Vercel** hosts the **React (Vite) frontend** as a separate static deploy.
  It calls the HF API cross-origin via a build-time `VITE_API_BASE`.
- **Supabase** provides **Postgres** (via the session-pooler URI) and **Storage**
  (one public bucket `call-artifacts`) for all artifacts.

Cross-origin implications: FastAPI sets CORS from `FRONTEND_ORIGIN`; the frontend
must prefix every fetch with `VITE_API_BASE` (an `apiUrl()` helper). The audio
route 307-redirects to the Supabase CDN.

Frontend fetch contract (served by the API from storage/DB):
`/calls_index.json`, `/calls/{id}.json`, `/sentence_segments/{id}.json`,
`/sentiment/{id}.json`, `/audio/{id}.mp3`.

IDs: the pipeline uses a **string public id** (`en_CA_Banking_1586889` for
seeds; `{domain}_{yyyymmdd}_{uuid8}` for uploads) which is the storage key and
public identifier. The DB `Call.call_id` is a UUID PK; the public id lives in
`Call.call_metadata.public_call_id` (JSONB). `SentimentSegment.call_id` and
`Transcript.source_call_id` are the string public id; `Evaluation.call_id` is
the UUID.

---

## 3. What is DONE and verified (Phases 1–3)

All commits are on `feat/transcription-eval-pipeline`.

**Phase 1 — portability** (`2ac72f2`): `ml-services/evaluation/paths.py`
centralizes every path, env-overridable via `CAPSTONE_DATA_ROOT` /
`CAPSTONE_EVAL_RESULTS` / `CAPSTONE_FRONTEND_PUBLIC`. `.env` loads at import
time. Killed the dual-root bug and the manifest hard-gate (`load_manifest()` now
merges `manifest_runtime.json`). `.env.example` documents every var.

**Phase 2 — single-call orchestrator** (`b4f317b`, `43538b5`, `eb59b77`):
- `ml-services/pipeline/orchestrator.py` — `process_call(spec, *, progress,
  enable_acoustic) -> artifacts`. Stages transcribe→register→segment→
  (acoustic)→evaluate→export, reusing existing modules. Idempotent transcript
  reuse (crash-retry skips re-transcription).
- `pipeline/acoustic.py` (wav2vec2 per-sentence, downloads weights from the
  PUBLIC HF repo `clarayoussef/wav2vec2-callcenter-emotion-v5`),
  `pipeline/audio_io.py` (stereo split / dual-mono normalize via ffmpeg),
  `pipeline/prompts_domain.py` (shared whisper prompts), `pipeline/run_one.py`
  (CLI).
- `43538b5` vendored the wav2vec2 inference package into `ml-services/src/`.
- `b4f317b` fixed a real bug: eval nodes now escalate PROVIDER on unparseable
  JSON (mistral→llama-70b→gpt-4o-mini). LiteLLM's fallback only fires on
  transport errors, not on a 200-OK response with invalid JSON, so a flaky
  provider used to kill whole calls.
- **Verified**: a fresh un-analyzed manifest call (`en_CA_Agriculture_1586885`)
  ran fully both text-only and with acoustic; rendered in the dashboard +
  detail page with seekable audio; acoustic fusion produced `weighted_mean`
  quality dims and a sentiment overlay.

**Phase 3 — backend + worker + Supabase** (`a6c0c63`, `e8fcf43`, `8f6ac58`):
- Adopted the backend service (7-table SQLAlchemy schema + sentiment router)
  from the backend branch, unchanged.
- `backend/app/worker.py` — single daemon thread + `queue.Queue`. Runs
  `process_call`, uploads artifacts to storage, writes transcripts/sentiment/
  evaluation rows, sets `Call.call_metadata.index_summary`. Startup recovery
  re-enqueues only THIS pipeline's interrupted jobs (guarded by
  `public_call_id`, so foreign jobs are left alone).
- `backend/app/pipeline_bridge.py` — the lone `sys.path` seam into
  `ml-services/{pipeline,evaluation,scripts}` + `ml-services` (for `src.*`),
  imported lazily in the worker thread only.
- `backend/app/storage.py` — Supabase Storage over REST. Public reads via CDN
  URL; writes send the key in the **`apikey` header** (the object endpoint
  rejects `sb_secret_` keys sent only as `Authorization: Bearer`).
- `backend/app/routers/artifacts.py` — serves the fetch contract:
  `calls_index` from Call rows; JSON proxied; mp3 307-redirect to CDN.
- `backend/app/routers/calls.py` — real `ingest` (domain field, stereo-split or
  two mono files, uploads originals for recovery, enqueues the worker);
  `get_call_status` returns `public_call_id/stage/error`.
- `backend/app/routing.py` — deterministic flag logic (shared by worker + router
  without the worker importing FastAPI).
- `backend/app/main.py` — lifespan inits DB, starts worker, pre-warms acoustic
  weights, sets CORS from `FRONTEND_ORIGIN`. `database.py` adds `pool_pre_ping`
  + `pool_recycle` for the Space's long-idle pooler connections.
- `backend/scripts/seed_calls.py` — idempotent seeder (artifacts → storage,
  Call rows with index_summary). Run from the dev machine.
- **Verified against the REAL Supabase project**: pooler `DATABASE_URL`
  connects (PG 17.6); live schema matches models; worker DB-write mapping
  (transcripts+sentiment+evaluation) validated non-destructively (insert then
  rollback); full uvicorn boot with all endpoints; storage write/read
  round-trip; `call-artifacts` public bucket created; the demo call
  `en_CA_Banking_1586889` seeded and served end-to-end
  (calls_index from DB + call JSON from storage + audio CDN redirect).

Current DB state: the backend project already had 2 legacy calls (untouched) +
1 job `queued` (a legacy job, correctly ignored by recovery). We seeded 1 call
(`en_CA_Banking_1586889`).

---

## 4. Environment & credentials

`.env` at the repo root (gitignored — NEVER commit it) holds all secrets:
LLM router keys (MISTRAL/SAMBANOVA/GITHUB + optional others), plus:

```
DATABASE_URL=<supabase session-pooler URI, ...pooler.supabase.com:5432/postgres?sslmode=require>
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_KEY=sb_publishable_...        # anon; reads only
SUPABASE_SERVICE_KEY=sb_secret_...     # REQUIRED for storage writes
SUPABASE_BUCKET=call-artifacts
ENABLE_ACOUSTIC=1
ACOUSTIC_MODEL_REPO=clarayoussef/wav2vec2-callcenter-emotion-v5
```

Gotchas already solved (do not rediscover):
- Storage object writes need the key in the **`apikey` header**, not just
  `Authorization: Bearer` (new `sb_secret_` keys aren't JWTs). `storage.py`
  already sends both.
- `DATABASE_URL` MUST be the **session-pooler (IPv4)** URI. The direct db host
  is IPv6-only and times out from HF.
- The acoustic weights repo is **public** — no `HF_TOKEN` needed.

Local deps installed this session: `sqlalchemy psycopg2-binary python-multipart
uvicorn librosa` (plus the Phase-2 ML stack). See `backend/requirements.txt`.

---

## 5. Remaining work

### 5a. Finish Phase 3 (backend, small additions for the demo flow)

The worker already processes any Call whose `call_metadata.public_call_id` is set
and whose channel wavs are in storage `uploads/{public_id}/{agent,customer}.wav`.
Two things are missing for the catalog→analyze flow:

1. **Stage demo calls** — new `backend/scripts/stage_demo_calls.py`: for N chosen
   un-analyzed manifest call_ids (short ones — see §7), upload their channel wavs
   (from `data/na_testset/{accent}/{id}_{agent,customer}.wav`) to storage
   `uploads/{id}/...` and insert a `Call` row with
   `call_metadata={public_call_id:id, domain, accent}` and **no** `index_summary`
   (→ shows as "available, not analyzed"). Use the manifest id as the public id.
2. **Analyze endpoint** — add `POST /calls/analyze` in `routers/calls.py`, body
   `{call_ids: [uuid,...]}`. For each, create a `Job(queued, stage="uploaded")`
   and `worker.enqueue(job_id)`; return the job ids. (The worker does the rest.)
3. **Catalog endpoint** — the dashboard must show analyzed AND un-analyzed calls.
   Either add `GET /calls/catalog` returning every Call with
   `{public_call_id, domain, accent, analyzed: bool, latest job status}`, or
   extend `calls_index` to include un-analyzed entries with a status field.
   (`calls_index.json` currently returns only calls that have `index_summary`.)

Then run the **full seed**: `python backend/scripts/seed_calls.py` (uploads all
~22–23 calls in `frontend/public/{calls,audio,sentence_segments,sentiment}` and
inserts their Call rows). ~200 MB of mp3 upload; run once from the dev machine.

Live E2E to confirm (local uvicorn + real Supabase): stage 1 short call → POST
`/calls/analyze` → poll `/calls/{uuid}/status` through the stages → call appears
in `calls_index` → detail served from storage.

### 5b. Phase 4 — frontend (catalog → select → analyze → progress)

- `frontend/src/api.js`: `const API_BASE = import.meta.env.VITE_API_BASE ?? "";
  export const apiUrl = p => API_BASE + p;`. Route EVERY existing fetch and the
  `<audio src>` in `Dashboard.jsx` + `CallDetail.jsx` through `apiUrl()`
  (they currently fetch same-origin paths).
- Extend the dashboard into a **catalog**: analyzed calls link to
  `/calls/{public_id}` (existing detail page, unchanged); un-analyzed calls get
  a checkbox + "not analyzed" badge.
- Selection + **"Analyze N calls"** button → `POST apiUrl('/calls/analyze')`.
- New `ProcessingStatus` UI (can be inline per-row or a dedicated page): poll
  `apiUrl('/calls/{uuid}/status')` every ~3–5 s; render a verbose stepper/
  progress bar over the stages `uploaded → transcribing → segmenting →
  acoustic → evaluating → exporting → done`; on `succeeded` refetch the catalog
  so the call flips to analyzed. Job status terminal values: worker sets
  `succeeded`/`failed` (note legacy rows may use `complete`).
- `vite.config.js`: dev proxy `/calls`, `/sentence_segments`, `/sentiment`,
  `/audio`, `/calls_index.json` to `http://localhost:8000` so local dev works
  with an empty `VITE_API_BASE`.
- Add `frontend/.env.example` documenting `VITE_API_BASE`.

### 5c. Phase 5 — deploy

- **HF Space (API):** root `Dockerfile`, single-stage `python:3.10-slim` +
  `ffmpeg` + `libpq5`; `pip install -r backend/requirements.txt
  --extra-index-url https://download.pytorch.org/whl/cpu` (CPU torch); copy
  `backend/`, `ml-services/`, and `data/na_testset/manifest.json`; non-root user
  1000; `HF_HOME` + `CAPSTONE_DATA_ROOT=/home/user/workdata`;
  `CMD uvicorn app.main:app --host 0.0.0.0 --port 7860` (run from `backend/`).
  `.dockerignore` excludes `ml-services/{outputs,notebooks,src/**/test_*}`,
  `data/na_testset/*.wav`, `frontend`, `.env`. Space `README.md` frontmatter:
  `sdk: docker`, `app_port: 7860`. Keep `backend/` and `ml-services/` as
  SIBLINGS in the image — `pipeline_bridge.py` resolves `../ml-services`.
- **Vercel (frontend):** import repo, root `frontend/`, framework Vite; set
  `VITE_API_BASE=https://<user>-<space>.hf.space`; add `vercel.json` SPA rewrite
  (`/(.*) -> /index.html`). After the Space URL is known, set `FRONTEND_ORIGIN`
  = the Vercel URL as a Space secret and redeploy.
- Set all Space secrets (the `.env` vars). Run `seed_calls.py` from dev once.
- Verify locally FIRST: `docker build && docker run --env-file .env -p 7860:7860`,
  then a Vite build with `VITE_API_BASE=http://localhost:7860`. Then live.
- **Deliberate persistence test**: reboot the Space mid-idle → all calls persist
  (Postgres + Storage survive the ephemeral disk); confirm the ~1–2 min cold
  start surfaces as a loading state in the always-warm Vercel shell, not a hard
  error.

---

## 6. Landmines discovered (do not re-hit)

- **LiteLLM fallback ≠ JSON validity.** Provider rotation on bad JSON lives in
  `nodes.py::_llm_eval_with_retry` (app layer), not the router.
- **`graph.py --results_dir` defaults to `results_channels`** (stale, 12 calls).
  Always pass/keep `results_dir="results"`.
- **`acoustic build_timeline` must stay default (True).** Fusion thresholds were
  fitted on that; changing it silently biases escalation tiers.
- **Worker must not clobber foreign jobs.** Recovery + `_run_job` are guarded by
  `public_call_id`. Keep that guard.
- **model_version is varchar(100).** The worker truncates it; keep that (a raw HF
  cache path overflows the column).
- **Storage writes need the `apikey` header** (see §4).

## 7. Demo timing reality (flag for the user)

Each call is ~5–15 min on the free CPU Space (whisper + wav2vec2 are CPU-bound).
3 calls sequential = too long to watch live. Options:
- Pick **short** demo calls (browse the manifest for low-duration ones) so each
  finishes in ~2–3 min.
- Or show the mechanism starting (progress bars moving) while the pre-seeded
  dashboard is already populated, and let them finish in the background.
- **Parallelization**: naive parallelism won't help the CPU-bound stages on 2
  vCPU and risks LLM rate limits. The realistic win is pipeline OVERLAP
  (transcribe call N+1 while the I/O-bound LLM eval of call N runs). Treat as a
  later optimization; the current single-thread worker is correct for the
  foundation.

---

## 8. How to run locally

```
# backend (needs .env at repo root)
cd backend && python -m uvicorn app.main:app --port 8000
# smoke: curl http://127.0.0.1:8000/health ; curl http://127.0.0.1:8000/calls_index.json

# one call through the pipeline directly (no backend)
cd ml-services/pipeline && python run_one.py --call_id <manifest_id> [--acoustic|--no-acoustic]

# seed / stage from the dev machine
python backend/scripts/seed_calls.py            # all calls in frontend/public
python backend/scripts/seed_calls.py --only <id>

# frontend
cd frontend && npm run dev    # http://localhost:5173
```

Find an un-analyzed manifest call (has wavs, no transcript yet): iterate
`data/na_testset/manifest.json`, skip ids that have
`data/na_testset/results/{accent}/{id}.json`.
