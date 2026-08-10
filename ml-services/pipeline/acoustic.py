"""Per-sentence sentiment and interpretable voice features."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL_REPO = "clarayoussef/wav2vec2-callcenter-emotion-v5"
MIN_SEG_S = 1.0
CACHE_VERSION = "acoustic-segment-v4"


class AcousticUnavailable(RuntimeError):
    """Model weights, dependencies, or the predictor could not be loaded."""


def _enum(value):
    return value.value if hasattr(value, "value") else value


def resolve_model_dir():
    local = os.environ.get("ACOUSTIC_MODEL_DIR")
    if local:
        if not Path(local).exists():
            raise AcousticUnavailable(f"ACOUSTIC_MODEL_DIR not found: {local}")
        return local
    repo = os.environ.get("ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO)
    try:
        from huggingface_hub import snapshot_download

        try:
            return snapshot_download(repo_id=repo, local_files_only=True)
        except Exception:
            return snapshot_download(repo_id=repo)
    except Exception as exc:
        raise AcousticUnavailable(
            f"could not fetch acoustic model {repo}: {exc}"
        ) from exc


_PREDICTOR = None
_PREDICTOR_INIT_LOCK = threading.Lock()


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        with _PREDICTOR_INIT_LOCK:
            if _PREDICTOR is None:
                try:
                    from src.inference.emotion_predictor import EmotionPredictor

                    _PREDICTOR = EmotionPredictor(model_dir=resolve_model_dir())
                except AcousticUnavailable:
                    raise
                except Exception as exc:
                    raise AcousticUnavailable(
                        f"emotion predictor init failed: {exc}"
                    ) from exc
    return _PREDICTOR


def _three_class_sentiment(result) -> str:
    sentiment = str(_enum(result.overall_audio_sentiment) or "").lower()
    if sentiment in {"positive", "negative", "neutral"}:
        return sentiment
    emotion = str(_enum(result.dominant_emotion) or "").lower()
    if emotion in {"happy", "happiness", "calm"}:
        return "positive"
    if emotion in {
        "anger",
        "angry",
        "fear",
        "sad",
        "sadness",
        "disgust",
        "anxiety",
        "stress",
    }:
        return "negative"
    return "neutral"


def _cache_path(call_id, sentence, clip):
    from paths import DATA_ROOT

    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode())
    digest.update(os.environ.get("ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO).encode())
    digest.update(str(call_id).encode())
    digest.update(
        json.dumps(
            {
                "seq_id": sentence.get("seq_id"),
                "speaker": sentence.get("speaker"),
                "start": sentence.get("start"),
                "end": sentence.get("end"),
                "text": sentence.get("text"),
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(clip.tobytes())
    directory = DATA_ROOT / "acoustic_segment_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest.hexdigest()}.json"


def _save_cache(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row), encoding="utf-8")
    os.replace(temporary, path)


def _audio_features(raw, categorical, text):
    duration = max(raw.duration_seconds, 1e-6)
    word_count = len((text or "").split())
    speech_rate = word_count / duration * 60
    pause_ratio = raw.total_silence_duration_seconds / duration
    pitch_mean = float(raw.pitch_mean)
    rms_mean = float(raw.rms_mean)
    return {
        **categorical,
        "duration_seconds": round(raw.duration_seconds, 4),
        "pitch_mean_hz": round(pitch_mean, 4),
        "pitch_std_hz": round(float(raw.pitch_std), 4),
        "rms_energy_mean": round(rms_mean, 6),
        "rms_energy_std": round(float(raw.rms_std), 6),
        "volume_db_mean": round(20 * math.log10(max(rms_mean, 1e-9)), 4),
        "total_pause_duration_seconds": round(
            raw.total_silence_duration_seconds, 4
        ),
        "longest_pause_seconds": round(raw.longest_silence_seconds, 4),
        "pause_count": raw.pause_count,
        "pause_ratio": round(pause_ratio, 4),
        "speech_rate_words_per_minute": round(speech_rate, 2),
        "word_count": word_count,
        "audio_quality_flags": {
            "very_short_segment": raw.duration_seconds < 1.5,
            "low_energy_segment": rms_mean < 0.005,
            "missing_pitch": pitch_mean <= 0,
            "unrealistic_speech_rate": speech_rate < 40 or speech_rate > 260,
        },
    }


def _average(rows, key):
    values = [
        row["audio_features"].get(key)
        for row in rows
        if isinstance(row.get("audio_features"), dict)
        and isinstance(row["audio_features"].get(key), (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


def _feature_summary(rows):
    return {
        "average_pitch_hz": _average(rows, "pitch_mean_hz"),
        "average_volume_db": _average(rows, "volume_db_mean"),
        "average_pause_ratio": _average(rows, "pause_ratio"),
        "average_speech_rate_wpm": _average(
            rows, "speech_rate_words_per_minute"
        ),
    }


def _speaker_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("speaker") or "UNKNOWN"].append(row)
    summaries = {}
    for speaker, speaker_rows in grouped.items():
        sentiments = Counter(
            row.get("sentiment_class")
            for row in speaker_rows
            if row.get("sentiment_class")
        )
        summaries[speaker.lower()] = {
            "total_segments": len(speaker_rows),
            "dominant_sentiment": (
                sentiments.most_common(1)[0][0] if sentiments else None
            ),
            **_feature_summary(speaker_rows),
        }
    return summaries


def analyze_call(
    call_id,
    segmentation,
    agent_wav,
    customer_wav,
    progress=None,
):
    """Analyze each sentence against the matching speaker channel."""
    import soundfile as sf
    from src.features.inference_audio_features import extract_raw_audio_features

    predictor = _get_predictor()
    channels = {}
    for speaker, wav in (("AGENT", agent_wav), ("CUSTOMER", customer_wav)):
        data, sample_rate = sf.read(str(wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        channels[speaker] = (data, sample_rate)

    rows = []
    sentences = segmentation["sentences"]
    total = len(sentences)
    for index, sentence in enumerate(sentences):
        if progress:
            progress(index, total)
        seq_id = sentence["seq_id"]
        speaker = sentence.get("speaker")
        start = float(sentence["start"])
        end = float(sentence["end"])
        base = {
            "seq_id": seq_id,
            "segment_index": index + 1,
            "segment_key": f"{call_id}_{index + 1:04d}",
            "speaker": speaker,
            "start_time": start,
            "end_time": end,
            "text": sentence.get("text"),
        }
        if end - start < MIN_SEG_S or speaker not in channels:
            rows.append({**base, "processing_status": "skipped_too_short"})
            continue

        data, sample_rate = channels[speaker]
        clip = data[int(start * sample_rate) : int(end * sample_rate)]
        if clip.size == 0:
            rows.append({**base, "processing_status": "skipped_too_short"})
            continue

        cache_path = _cache_path(call_id, sentence, clip)
        if cache_path.exists():
            try:
                rows.append(json.loads(cache_path.read_text(encoding="utf-8")))
                continue
            except (OSError, ValueError):
                pass

        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temporary.close()
        try:
            sf.write(temporary.name, clip, sample_rate)
            raw = extract_raw_audio_features(Path(temporary.name))
            result = predictor.analyze_audio(
                Path(temporary.name),
                call_id=call_id,
                raw_audio_features=raw,
            )
            detail = result.model_dump(mode="json")
            row = {
                **base,
                "sentiment": _enum(result.overall_audio_sentiment),
                "sentiment_class": _three_class_sentiment(result),
                "dominant_emotion": _enum(result.dominant_emotion),
                "escalation_score": round(float(result.audio_escalation_score), 4),
                "confidence_level": _enum(result.confidence_level),
                "prediction_confidence": result.prediction_confidence,
                "emotion_confidence": result.prediction_confidence,
                "sentiment_confidence": result.prediction_confidence,
                "negative_emotion_probability": result.negative_emotion_probability,
                "audio_features": _audio_features(
                    raw,
                    detail.get("audio_features") or {},
                    sentence.get("text"),
                ),
                "processing_status": result.processing_status or "success",
            }
            rows.append(row)
            _save_cache(cache_path, row)
        except Exception as exc:
            rows.append(
                {**base, "processing_status": f"error: {exc}"[:120]}
            )
        finally:
            try:
                os.unlink(temporary.name)
            except OSError:
                pass

    if progress:
        progress(total, total)

    successful = [
        row for row in rows if row.get("processing_status") == "success"
    ]
    sentiment_counts = Counter(
        row.get("sentiment_class")
        for row in successful
        if row.get("sentiment_class")
    )
    emotion_counts = Counter(
        row.get("dominant_emotion")
        for row in successful
        if row.get("dominant_emotion")
    )
    scores = [
        row["escalation_score"]
        for row in successful
        if row.get("escalation_score") is not None
    ]
    return {
        "call_id": call_id,
        "domain": segmentation.get("domain"),
        "model_version": os.environ.get(
            "ACOUSTIC_MODEL_REPO", DEFAULT_MODEL_REPO
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "has_audio_features": any(
            isinstance(row.get("audio_features"), dict) for row in successful
        ),
        "audio_feature_version": "runtime-v2",
        "segments": rows,
        "call_summary": {
            "total_segments": len(rows),
            "successful_segments": len(successful),
            "skipped_segments": len(rows) - len(successful),
            "dominant_sentiment": (
                sentiment_counts.most_common(1)[0][0]
                if sentiment_counts
                else None
            ),
            "dominant_emotion": (
                emotion_counts.most_common(1)[0][0] if emotion_counts else None
            ),
            "average_escalation_score": (
                round(sum(scores) / len(scores), 6) if scores else None
            ),
            "max_escalation_score": round(max(scores), 6) if scores else None,
            "sentiment_distribution": dict(sentiment_counts),
            "emotion_distribution": dict(emotion_counts),
        },
        "audio_feature_summary": _feature_summary(successful),
        "speaker_audio_feature_summary": _speaker_summaries(successful),
        "dashboard_audio_feature_series": [
            {
                "seq_id": row.get("seq_id"),
                "speaker": row.get("speaker"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "sentiment": row.get("sentiment_class"),
                "dominant_emotion": row.get("dominant_emotion"),
                **(row.get("audio_features") or {}),
            }
            for row in successful
        ],
    }
