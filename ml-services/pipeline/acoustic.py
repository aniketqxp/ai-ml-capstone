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
import threading
import math
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL_REPO = "clarayoussef/wav2vec2-callcenter-emotion-v5"
MIN_SEG_S = 1.0   # matches run_timestamped_sentiment.py --min-segment-seconds
CACHE_VERSION = "acoustic-segment-v3"


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
        # Avoid a Hub metadata round-trip on every backend restart once all
        # weights are cached. Fall back to downloading only on a cache miss.
        try:
            return snapshot_download(repo_id=repo, local_files_only=True)
        except Exception:
            return snapshot_download(repo_id=repo)
    except Exception as e:
        raise AcousticUnavailable(f"could not fetch acoustic model {repo}: {e}")


_PREDICTOR = None
_PREDICTOR_INIT_LOCK = threading.Lock()


def _get_predictor():
    """Lazy singleton EmotionPredictor (loads weights once per process)."""
    global _PREDICTOR
    if _PREDICTOR is None:
        with _PREDICTOR_INIT_LOCK:
            if _PREDICTOR is None:
                try:
                    from src.inference.emotion_predictor import EmotionPredictor
                except Exception as e:               # missing torch/transformers/src
                    raise AcousticUnavailable(f"emotion predictor import failed: {e}")
                try:
                    _PREDICTOR = EmotionPredictor(model_dir=resolve_model_dir())
                except AcousticUnavailable:
                    raise
                except Exception as e:
                    raise AcousticUnavailable(f"emotion predictor init failed: {e}")
    return _PREDICTOR


def _cache_path(call_id, sentence, clip):
    """Stable checkpoint key for one exact audio/text segment and model."""
    from paths import DATA_ROOT
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode())
    digest.update(os.environ.get("ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO).encode())
    digest.update(str(call_id).encode())
    digest.update(json.dumps({
        "seq_id": sentence.get("seq_id"), "speaker": sentence.get("speaker"),
        "start": sentence.get("start"), "end": sentence.get("end"),
        "text": sentence.get("text"),
    }, sort_keys=True).encode())
    digest.update(clip.tobytes())
    directory = DATA_ROOT / "acoustic_segment_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest.hexdigest()}.json"


def _save_cache(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row), encoding="utf-8")
    os.replace(temporary, path)


def analyze_call(call_id, segmentation, agent_wav, customer_wav, progress=None):
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
    sentences = segmentation["sentences"]
    total = len(sentences)
    for index, s in enumerate(sentences):
        if progress:
            progress(index, total)
        seq_id = s["seq_id"]
        speaker = s.get("speaker")
        start, end = float(s["start"]), float(s["end"])

        if end - start < MIN_SEG_S or speaker not in channels:
            rows.append({"seq_id": seq_id, "segment_index": len(rows) + 1,
                         "segment_key": f"{call_id}_{len(rows) + 1:04d}",
                         "speaker": speaker, "start_time": start, "end_time": end,
                         "text": s.get("text"),
                         "processing_status": "skipped_too_short"})
            continue

        data, sr = channels[speaker]
        clip = data[int(start * sr): int(end * sr)]
        if clip.size == 0:
            rows.append({"seq_id": seq_id, "segment_index": len(rows) + 1,
                         "segment_key": f"{call_id}_{len(rows) + 1:04d}",
                         "speaker": speaker, "start_time": start, "end_time": end,
                         "text": s.get("text"),
                         "processing_status": "skipped_too_short"})
            continue

        cache_path = _cache_path(call_id, s, clip)
        if cache_path.exists():
            try:
                rows.append(json.loads(cache_path.read_text(encoding="utf-8")))
                continue
            except (OSError, ValueError):
                pass

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, clip, sr)
            from src.features.inference_audio_features import extract_raw_audio_features
            raw_features = extract_raw_audio_features(Path(tmp.name))
            # Reuse these raw features inside analyze_audio. Previously it ran
            # the same librosa pitch/pause extraction a second time.
            res = predictor.analyze_audio(
                Path(tmp.name), call_id=call_id,
                raw_audio_features=raw_features)
            detail = res.model_dump(mode="json")
            duration = max(raw_features.duration_seconds, 1e-6)
            word_count = len((s.get("text") or "").split())
            categorical = detail.get("audio_features") or {}
            audio_features = {
                **categorical,
                "duration_seconds": round(raw_features.duration_seconds, 4),
                "pitch_mean_hz": round(raw_features.pitch_mean, 4),
                "pitch_std_hz": round(raw_features.pitch_std, 4),
                "rms_energy_mean": round(raw_features.rms_mean, 6),
                "rms_energy_std": round(raw_features.rms_std, 6),
                "volume_db_mean": round(20 * math.log10(max(raw_features.rms_mean, 1e-9)), 4),
                "pause_count": raw_features.pause_count,
                "pause_ratio": round(raw_features.total_silence_duration_seconds / duration, 4),
                "speech_rate_words_per_minute": round(word_count / duration * 60, 2),
                "word_count": word_count,
                "volume_level": str(categorical.get("vocal_intensity", "unknown")).lower(),
                "speech_rate_level": (
                    "fast" if word_count / duration * 60 >= 180
                    else "slow" if word_count / duration * 60 < 90
                    else "normal"
                ),
                "pause_level": (
                    "high" if raw_features.total_silence_duration_seconds / duration >= 0.35
                    else "medium" if raw_features.total_silence_duration_seconds / duration >= 0.15
                    else "low"
                ),
            }
            row = {
                "seq_id": seq_id,
                "segment_index": len(rows) + 1,
                "segment_key": f"{call_id}_{len(rows) + 1:04d}",
                "speaker": speaker,
                "start_time": start,
                "end_time": end,
                "text": s.get("text"),
                "sentiment": _enum(res.overall_audio_sentiment),
                "dominant_emotion": _enum(res.dominant_emotion),
                "escalation_score": round(float(res.audio_escalation_score), 4),
                "confidence_level": _enum(res.confidence_level),
                "prediction_confidence": res.prediction_confidence,
                "emotion_confidence": res.prediction_confidence,
                "sentiment_confidence": res.prediction_confidence,
                "negative_emotion_probability": res.negative_emotion_probability,
                "audio_features": audio_features,
                "processing_status": res.processing_status or "success",
            }
            rows.append(row)
            _save_cache(cache_path, row)
        except Exception as e:
            rows.append({"seq_id": seq_id, "processing_status": f"error: {e}"[:120]})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    if progress:
        progress(total, total)

    # record the repo id (stable, machine-independent) -- NOT
    # config._name_or_path, which resolves to the local HF cache dir and would
    # leak an absolute user path into an uploaded artifact
    model_version = os.environ.get("ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO)

    successful = [row for row in rows if row.get("processing_status") == "success"]
    sentiment_counts = Counter(row.get("sentiment") for row in successful if row.get("sentiment"))
    emotion_counts = Counter(row.get("dominant_emotion") for row in successful if row.get("dominant_emotion"))
    scores = [row["escalation_score"] for row in successful
              if row.get("escalation_score") is not None]
    payload = {
        "call_id": call_id,
        "model_version": model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segments": rows,
        "call_summary": {
            "total_segments": len(rows),
            "successful_segments": len(successful),
            "skipped_segments": len(rows) - len(successful),
            "dominant_sentiment": sentiment_counts.most_common(1)[0][0] if sentiment_counts else None,
            "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else None,
            "average_escalation_score": round(sum(scores) / len(scores), 6) if scores else None,
            "max_escalation_score": round(max(scores), 6) if scores else None,
            "sentiment_distribution": dict(sentiment_counts),
            "emotion_distribution": dict(emotion_counts),
        },
    }
    try:
        from src.integration.merge_sentiment_audio_features import (
            build_agent_empathy_tone_alignment,
            build_audio_feature_series,
            build_audio_feature_summary,
            build_calibrated_call_sentiment,
            build_customer_escalation_trend,
            build_manager_review_recommendation,
            build_multi_signal_escalation_intelligence,
            build_segment_explainability,
            build_speaker_audio_feature_summary,
            build_temporal_emotion_trajectory,
            build_top_risky_segments,
        )
        for row in rows:
            explanation = build_segment_explainability(row)
            row.update(explanation)
        payload["has_audio_features"] = any(row.get("audio_features") for row in rows)
        payload["audio_feature_version"] = "runtime-v2"
        payload["audio_feature_match_summary"] = {
            "matched_segments": sum(bool(row.get("audio_features")) for row in rows),
            "total_sentiment_segments": len(rows),
        }
        payload["dashboard_audio_feature_series"] = build_audio_feature_series(payload)
        payload["audio_feature_summary"] = build_audio_feature_summary(payload)
        payload["speaker_audio_feature_summary"] = build_speaker_audio_feature_summary(payload)
        payload["customer_escalation_trend"] = build_customer_escalation_trend(payload)
        payload["calibrated_sentiment_summary"] = build_calibrated_call_sentiment(payload)
        payload["manager_review_recommendation"] = build_manager_review_recommendation(payload)
        payload["multi_signal_escalation_intelligence"] = build_multi_signal_escalation_intelligence(payload)
        payload["temporal_emotion_trajectory"] = build_temporal_emotion_trajectory(payload)
        payload["agent_empathy_tone_alignment"] = build_agent_empathy_tone_alignment(payload)
        payload["top_risky_segments"] = build_top_risky_segments(payload)
    except Exception as exc:
        print(f"  [acoustic] enrichment unavailable: {type(exc).__name__}: {exc}")
    return payload
