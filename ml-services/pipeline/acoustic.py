"""
Acoustic sentiment stage (gated).

Runs Clara's wav2vec2 call-center emotion model over each sentence segment and
emits the SIMPLE per-seq_id schema that `fusion.load_acoustic()` joins against
the segmentation:

  {call_id, model_version, segments:[{seq_id, sentiment, dominant_emotion,
                                      escalation_score, processing_status}, ...]}

Fidelity note (correctness-critical): fusion's escalation tiers (late_mean >=
0.30 review, >= 0.50 escalate) were FITTED on the 22-call batch, which was
produced by `run_timestamped_sentiment.py` calling `analyze_audio(clip)` PER
SENTENCE with the default `build_timeline=True`. We reproduce that call exactly
(default build_timeline) so a fresh call's escalation_score lands on the same
distribution the thresholds assume. Changing build_timeline or the per-sentence
granularity would silently bias the tiers.

Weights come from the public HF repo (cached by huggingface_hub), overridable
with ACOUSTIC_MODEL_DIR. Any unavailability raises AcousticUnavailable; the
orchestrator catches it and lets the graph fuse text-only.
"""
import os
import tempfile
from pathlib import Path

DEFAULT_MODEL_REPO = "clarayoussef/wav2vec2-callcenter-emotion-v5"
MIN_SEG_S = 1.0   # matches run_timestamped_sentiment.py --min-segment-seconds


class AcousticUnavailable(RuntimeError):
    """Model weights, deps, or predictor could not be loaded."""


def _enum(v):
    """Enum -> its string value; pass through plain strings/None."""
    return v.value if hasattr(v, "value") else v


def resolve_model_dir():
    """Local dir override (ACOUSTIC_MODEL_DIR), else download the hub repo."""
    local = os.environ.get("ACOUSTIC_MODEL_DIR")
    if local:
        if not Path(local).exists():
            raise AcousticUnavailable(f"ACOUSTIC_MODEL_DIR not found: {local}")
        return local
    repo = os.environ.get("ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO)
    try:
        from huggingface_hub import snapshot_download
        # HF_TOKEN in the environment is honored automatically if the repo
        # is ever made private; the default repo is public.
        return snapshot_download(repo_id=repo)
    except Exception as e:
        raise AcousticUnavailable(f"could not fetch acoustic model {repo}: {e}")


_PREDICTOR = None


def _get_predictor():
    """Lazy singleton EmotionPredictor (loads weights once per process)."""
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            from src.inference.emotion_predictor import EmotionPredictor
        except Exception as e:                       # missing torch/transformers/src
            raise AcousticUnavailable(f"emotion predictor import failed: {e}")
        try:
            _PREDICTOR = EmotionPredictor(model_dir=resolve_model_dir())
        except AcousticUnavailable:
            raise
        except Exception as e:
            raise AcousticUnavailable(f"emotion predictor init failed: {e}")
    return _PREDICTOR


def analyze_call(call_id, segmentation, agent_wav, customer_wav):
    """
    Per-sentence acoustic sentiment for one call.

    `segmentation` is the `segment_transcript()` dict; each sentence carries
    seq_id, speaker (AGENT/CUSTOMER), start, end (seconds). Slices the matching
    speaker channel per sentence and runs the emotion model. Returns the simple
    fusion-ready dict described in the module docstring.
    """
    import soundfile as sf
    predictor = _get_predictor()

    # preload each channel once; slice in-memory per sentence
    channels = {}
    for speaker, wav in (("AGENT", agent_wav), ("CUSTOMER", customer_wav)):
        data, sr = sf.read(str(wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        channels[speaker] = (data, sr)

    rows = []
    for s in segmentation["sentences"]:
        seq_id = s["seq_id"]
        speaker = s.get("speaker")
        start, end = float(s["start"]), float(s["end"])

        if end - start < MIN_SEG_S or speaker not in channels:
            rows.append({"seq_id": seq_id, "processing_status": "skipped_too_short"})
            continue

        data, sr = channels[speaker]
        clip = data[int(start * sr): int(end * sr)]
        if clip.size == 0:
            rows.append({"seq_id": seq_id, "processing_status": "skipped_too_short"})
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, clip, sr)
            # default build_timeline=True -- see module fidelity note
            res = predictor.analyze_audio(Path(tmp.name), call_id=call_id)
            rows.append({
                "seq_id": seq_id,
                "sentiment": _enum(res.overall_audio_sentiment),
                "dominant_emotion": _enum(res.dominant_emotion),
                "escalation_score": round(float(res.audio_escalation_score), 4),
                "processing_status": res.processing_status or "success",
            })
        except Exception as e:
            rows.append({"seq_id": seq_id, "processing_status": f"error: {e}"[:120]})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # record the repo id (stable, machine-independent) -- NOT
    # config._name_or_path, which resolves to the local HF cache dir and would
    # leak an absolute user path into an uploaded artifact
    model_version = os.environ.get("ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO)

    return {"call_id": call_id, "model_version": model_version, "segments": rows}
