"""
Single-call orchestrator: fresh call in -> full dashboard artifacts out.

Runs the same chain the offline CLIs run, but for ONE call, in-process, with a
progress callback so a web worker can report stage transitions:

  transcribing -> registering -> segmenting -> (acoustic) -> evaluating -> exporting

Every stage reuses the existing modules (no logic forked): faster-whisper via
transcribe_channels, segmentation via batch_sentence_segments.segment_transcript,
the LangGraph evaluation via graph.evaluate_call, and the frontend export via
export_frontend.export_call_artifacts. The acoustic stage is gated (ENABLE_ACOUSTIC
or the enable_acoustic arg) and degrades to text-only fusion on any failure.

Artifacts land in the same places the batch flow uses, plus the flat
frontend/public/{sentence_segments,sentiment}/ files CallDetail fetches.
"""
import os
import sys
import json
import shutil
from pathlib import Path

# make the flat evaluation/scripts modules and the src package importable
# regardless of the caller's cwd (mirrors run_batch.py's sys.path shim)
_ML = Path(__file__).resolve().parents[1]                 # ml-services/
_HERE = Path(__file__).resolve().parent                   # ml-services/pipeline/
for _p in (_HERE, _ML / "evaluation", _ML / "scripts", _ML):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                               # noqa: E402
from assemble import load_manifest                         # noqa: E402
from batch_sentence_segments import segment_transcript     # noqa: E402
from evaluator_runtime import evaluate_call               # noqa: E402
from export_frontend import export_call_artifacts, summarize  # noqa: E402
from prompts_domain import prompt_for                      # noqa: E402

RESULTS_DIR_NAME = "results"       # graph reads na_testset/results/{accent}/{id}.json
EVAL_RESULTS = _ML / "evaluation" / "results"


def _noop(*_a, **_k):
    pass


_WHISPER = None


def _whisper_model():
    """Lazy singleton faster-whisper model (loaded once per process)."""
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        name = os.environ.get("WHISPER_MODEL", "small.en")
        _WHISPER = WhisperModel(name, device="cpu", compute_type="int8")
    return _WHISPER


def _acoustic_enabled(explicit):
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("ENABLE_ACOUSTIC", "0").strip().lower() not in ("", "0", "false", "no")


def _canonical_wavs(call_id, accent, agent_wav, customer_wav):
    """Ensure per-channel WAVs sit at the dataset convention path the export +
    manifest consumers expect: NA_TESTSET/{accent}/{call_id}_{role}.wav.
    Copies only if the source is elsewhere (fresh uploads); manifest calls are
    already in place."""
    dst_dir = paths.NA_TESTSET / accent
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for role, src in (("agent", agent_wav), ("customer", customer_wav)):
        dst = dst_dir / f"{call_id}_{role}.wav"
        if os.path.abspath(src) != os.path.abspath(str(dst)):
            shutil.copyfile(src, dst)
        out[role] = str(dst)
    return out["agent"], out["customer"]


def _register(call_id, accent, domain):
    """Add/replace this call in the runtime manifest so load_manifest() (and
    thus every eval/export consumer) finds it. Lean entry -- the hosted path
    reads only accent/domain/call_id; reference transcripts aren't needed."""
    rm = paths.RUNTIME_MANIFEST
    entries = json.loads(rm.read_text(encoding="utf-8")) if rm.exists() else []
    entries = [e for e in entries if e.get("call_id") != call_id]
    entries.append({
        "call_id": call_id, "accent": accent, "domain": domain,
        "agent_wav": f"{accent}/{call_id}_agent.wav",
        "customer_wav": f"{accent}/{call_id}_customer.wav",
    })
    rm.parent.mkdir(parents=True, exist_ok=True)
    rm.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _publish_index(call_id, summary):
    """Upsert one row into frontend/public/calls_index.json (dashboard list)."""
    idx = paths.FRONTEND_PUBLIC / "calls_index.json"
    index = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else []
    index = [e for e in index if e.get("call_id") != call_id]
    index.append(summary)
    _write(idx, index)


def process_call(spec, *, progress=_noop, enable_acoustic=None):
    """
    Run the full pipeline for one call.

    spec: {call_id, domain, accent, agent_wav, customer_wav} (wav paths absolute).
    progress: called with a stage name string at each transition.
    Returns an artifacts dict describing what was written (paths + index row).
    """
    call_id = spec["call_id"]
    accent = spec["accent"]
    domain = spec["domain"]
    do_acoustic = _acoustic_enabled(enable_acoustic)

    agent_wav, customer_wav = _canonical_wavs(
        call_id, accent, spec["agent_wav"], spec["customer_wav"])

    artifacts = {"call_id": call_id, "domain": domain, "accent": accent,
                 "acoustic": False, "paths": {}}

    # 1. transcribe -----------------------------------------------------------
    # idempotent: an existing non-empty transcript is reused, so a job retried
    # after a mid-pipeline crash skips the expensive re-transcription (fresh
    # uploads carry unique call_ids, so this never wrongly reuses another call)
    progress("transcribing")
    transcript_path = paths.NA_TESTSET / RESULTS_DIR_NAME / accent / f"{call_id}.json"
    if transcript_path.exists() and transcript_path.stat().st_size > 0:
        print(f"  [transcribe] reusing existing transcript {transcript_path.name}")
        result = json.loads(transcript_path.read_text(encoding="utf-8"))
    else:
        from transcribe_channels import transcribe_call
        model = _whisper_model()
        agent_words, customer_words, _segs, _dur = transcribe_call(
            model, agent_wav, customer_wav,
            decode={"initial_prompt": prompt_for(domain)})
        result = {
            "model": f"faster-whisper {os.environ.get('WHISPER_MODEL', 'small.en')}/int8 + per-channel",
            "call_id": call_id, "accent": accent, "domain": domain,
            "agent": agent_words, "customer": customer_words,
        }
        _write(transcript_path, result)
    artifacts["paths"]["transcript"] = str(transcript_path)

    # 2. register -------------------------------------------------------------
    progress("registering")
    _register(call_id, accent, domain)

    # 3. segment --------------------------------------------------------------
    progress("segmenting")
    seg = segment_transcript(result)
    _write(paths.SENTENCE_SEG_ROOT / domain.lower() / f"{call_id}.json", seg)
    # flat, frontend-shaped copy CallDetail fetches (it reads only .sentences)
    _write(paths.FRONTEND_PUBLIC / "sentence_segments" / f"{call_id}.json", {
        "call": call_id, "model": seg.get("model", ""),
        "total": seg["total"], "agent_sentences": seg["agent_sentences"],
        "customer_sentences": seg["customer_sentences"], "sentences": seg["sentences"],
    })
    artifacts["paths"]["sentence_segments"] = str(
        paths.FRONTEND_PUBLIC / "sentence_segments" / f"{call_id}.json")

    # 4. acoustic (gated) -----------------------------------------------------
    if do_acoustic:
        progress("acoustic")
        try:
            import acoustic
            sentiment = acoustic.analyze_call(call_id, seg, agent_wav, customer_wav)
            sdir = paths.SENTIMENT_ROOT / domain.lower()
            _write(sdir / f"{call_id}.json", sentiment)          # fusion reads here
            _write(sdir / f"{call_id}_segments.json", seg)       # fusion join
            _write(paths.FRONTEND_PUBLIC / "sentiment" / f"{call_id}.json", sentiment)
            artifacts["acoustic"] = True
            artifacts["paths"]["sentiment"] = str(
                paths.FRONTEND_PUBLIC / "sentiment" / f"{call_id}.json")
        except Exception as e:
            print(f"  [acoustic] unavailable -> text-only fusion: "
                  f"{type(e).__name__}: {e}")
    else:
        print("  [acoustic] disabled (ENABLE_ACOUSTIC=0) -> text-only fusion")

    # 5. evaluate -------------------------------------------------------------
    progress("evaluating")
    ev = evaluate_call(call_id, results_dir=RESULTS_DIR_NAME)
    _write(EVAL_RESULTS / f"{call_id}_graph.json", ev)
    artifacts["paths"]["evaluation"] = str(EVAL_RESULTS / f"{call_id}_graph.json")

    # 6. export ---------------------------------------------------------------
    progress("exporting")
    meta = load_manifest()[call_id]
    calls_out = str(paths.FRONTEND_PUBLIC / "calls")
    audio_out = str(paths.FRONTEND_PUBLIC / "audio")
    out = export_call_artifacts(call_id, meta, calls_out, audio_out)
    summary = summarize(call_id, out)
    _publish_index(call_id, summary)
    artifacts["paths"]["call_json"] = os.path.join(calls_out, f"{call_id}.json")
    artifacts["paths"]["audio"] = os.path.join(audio_out, f"{call_id}.mp3")
    artifacts["index_summary"] = summary

    progress("done")
    return artifacts
