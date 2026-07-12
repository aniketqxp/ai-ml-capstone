"""
Audio feature extraction for call-center sentiment analysis.

This script extracts explainable voice/tone features from audio segments:
- pitch
- volume
- energy
- pauses
- speech rate estimate

These features are separate from Wav2Vec2. Wav2Vec2 predicts emotion,
while this file extracts interpretable acoustic features for dashboard use.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import librosa
import numpy as np


def load_audio_segment(
    audio_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    target_sr: int = 16000,
):
    """Load full audio or a timestamped segment."""
    if start_time is not None and end_time is not None:
        duration = max(0.0, end_time - start_time)
        y, sr = librosa.load(
            audio_path,
            sr=target_sr,
            mono=True,
            offset=start_time,
            duration=duration,
        )
    else:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    return y, sr

def slice_audio_segment(
    full_audio: np.ndarray,
    sr: int,
    start_time: Optional[float],
    end_time: Optional[float],
) -> np.ndarray:
    """
    Slice a segment from audio that was already loaded once.

    This is much faster than calling librosa.load for every segment.
    """
    if start_time is None or end_time is None:
        return full_audio

    start_sample = max(0, int(float(start_time) * sr))
    end_sample = min(len(full_audio), int(float(end_time) * sr))

    if end_sample <= start_sample:
        return np.array([], dtype=full_audio.dtype)

    return full_audio[start_sample:end_sample]


def safe_float(value):
    """Convert numpy values to normal Python floats for JSON."""
    if value is None:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def extract_pitch(y: np.ndarray, sr: int, pitch_mode: str = "full") -> Dict[str, Any]:
    """
    Extract pitch features.

    full = more accurate but slower, uses librosa.pyin
    fast = faster, uses librosa.yin
    skip = fastest, skips pitch extraction
    """
    if y is None or len(y) == 0:
        return {
            "pitch_mean_hz": None,
            "pitch_min_hz": None,
            "pitch_max_hz": None,
            "pitch_std_hz": None,
            "pitch_level": "unknown",
            "pitch_mode": pitch_mode,
        }

    if pitch_mode == "skip":
        return {
            "pitch_mean_hz": None,
            "pitch_min_hz": None,
            "pitch_max_hz": None,
            "pitch_std_hz": None,
            "pitch_level": "skipped",
            "pitch_mode": pitch_mode,
        }

    try:
        if pitch_mode == "fast":
            pitch_values = librosa.yin(
                y,
                fmin=50,
                fmax=500,
                sr=sr,
            )
        else:
            pitch_values, _, _ = librosa.pyin(
                y,
                fmin=50,
                fmax=500,
                sr=sr,
            )

        valid_pitch = pitch_values[~np.isnan(pitch_values)]

        if len(valid_pitch) == 0:
            return {
                "pitch_mean_hz": None,
                "pitch_min_hz": None,
                "pitch_max_hz": None,
                "pitch_std_hz": None,
                "pitch_level": "unknown",
                "pitch_mode": pitch_mode,
            }

        pitch_mean = float(np.mean(valid_pitch))
        pitch_min = float(np.min(valid_pitch))
        pitch_max = float(np.max(valid_pitch))
        pitch_std = float(np.std(valid_pitch))

        if pitch_mean < 120:
            pitch_level = "low"
        elif pitch_mean <= 220:
            pitch_level = "medium"
        else:
            pitch_level = "high"

        return {
            "pitch_mean_hz": safe_float(pitch_mean),
            "pitch_min_hz": safe_float(pitch_min),
            "pitch_max_hz": safe_float(pitch_max),
            "pitch_std_hz": safe_float(pitch_std),
            "pitch_level": pitch_level,
            "pitch_mode": pitch_mode,
        }

    except Exception:
        return {
            "pitch_mean_hz": None,
            "pitch_min_hz": None,
            "pitch_max_hz": None,
            "pitch_std_hz": None,
            "pitch_level": "error",
            "pitch_mode": pitch_mode,
        }


def extract_volume_energy(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract RMS energy and approximate volume in dB.
    """
    if len(y) == 0:
        return {
            "rms_energy_mean": None,
            "rms_energy_max": None,
            "volume_db_mean": None,
            "volume_level": "empty_audio",
        }

    rms = librosa.feature.rms(y=y)[0]

    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))

    # Convert RMS to dB. Add small number to avoid log(0).
    volume_db = librosa.amplitude_to_db(rms, ref=np.max)
    volume_db_mean = float(np.mean(volume_db))

    if rms_mean < 0.01:
        volume_level = "low"
    elif rms_mean < 0.04:
        volume_level = "medium"
    else:
        volume_level = "high"

    return {
        "rms_energy_mean": safe_float(rms_mean),
        "rms_energy_max": safe_float(rms_max),
        "volume_db_mean": safe_float(volume_db_mean),
        "volume_level": volume_level,
    }


def extract_pauses(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Detect non-silent intervals and estimate pause duration.

    This is useful for detecting hesitation or silence.
    """
    if len(y) == 0:
        return {
            "total_pause_duration_seconds": None,
            "pause_count": None,
            "pause_ratio": None,
            "pause_level": "empty_audio",
        }

    duration_seconds = len(y) / sr

    intervals = librosa.effects.split(
        y,
        top_db=30,
    )

    speech_duration = 0.0
    for start, end in intervals:
        speech_duration += (end - start) / sr

    pause_duration = max(0.0, duration_seconds - speech_duration)
    pause_ratio = pause_duration / duration_seconds if duration_seconds > 0 else 0.0

    # Estimate pauses as gaps between speech intervals
    pause_count = max(0, len(intervals) - 1)

    if pause_ratio < 0.15:
        pause_level = "low"
    elif pause_ratio < 0.35:
        pause_level = "medium"
    else:
        pause_level = "high"

    return {
        "total_pause_duration_seconds": safe_float(pause_duration),
        "pause_count": int(pause_count),
        "pause_ratio": safe_float(pause_ratio),
        "pause_level": pause_level,
    }


def estimate_speech_rate(y: np.ndarray, sr: int, transcript_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Estimate speech rate.

    Best option: use transcript text word count / segment duration.
    If transcript text is not available, return an audio-based placeholder estimate.
    """
    duration_seconds = len(y) / sr if sr else 0

    if duration_seconds <= 0:
        return {
            "word_count": None,
            "speech_rate_words_per_second": None,
            "speech_rate_words_per_minute": None,
            "speech_rate_level": "empty_audio",
        }

    if transcript_text:
        words = [w for w in transcript_text.strip().split() if w]
        word_count = len(words)
        wps = word_count / duration_seconds
        wpm = wps * 60

        if wpm < 110:
            speech_rate_level = "slow"
        elif wpm < 180:
            speech_rate_level = "normal"
        else:
            speech_rate_level = "fast"

        return {
            "word_count": int(word_count),
            "speech_rate_words_per_second": safe_float(wps),
            "speech_rate_words_per_minute": safe_float(wpm),
            "speech_rate_level": speech_rate_level,
        }

    return {
        "word_count": None,
        "speech_rate_words_per_second": None,
        "speech_rate_words_per_minute": None,
        "speech_rate_level": "transcript_required",
    }


def extract_audio_features(
    audio_path: str,
    start_time: float,
    end_time: float,
    transcript_text: Optional[str] = None,
    pitch_mode: str = "full",
) -> Dict[str, Any]:
    """
    Extract all audio features for one file or one timestamped segment.
    """
    y, sr = load_audio_segment(audio_path, start_time, end_time, target_sr)

    duration_seconds = len(y) / sr if sr else 0

    features = {
        "duration_seconds": safe_float(duration_seconds),
        "sample_rate": sr,
    }

    features.update(extract_pitch(y, sr, pitch_mode=pitch_mode))
    features.update(extract_volume_energy(y, sr))
    features.update(extract_pauses(y, sr))
    features.update(estimate_speech_rate(y, sr, transcript_text))

    return features


def extract_audio_features_from_array(
    y: np.ndarray,
    sr: int,
    transcript_text: Optional[str] = None,
    pitch_mode: str = "full",
) -> Dict[str, Any]:
    """
    Extract all audio features from an already-loaded audio segment.

    This avoids reloading the audio file for every transcript segment.
    """
    duration_seconds = len(y) / sr if sr else 0

    features = {
        "duration_seconds": safe_float(duration_seconds),
        "sample_rate": sr,
    }

    features.update(extract_pitch(y, sr, pitch_mode=pitch_mode))
    features.update(extract_volume_energy(y, sr))
    features.update(extract_pauses(y, sr))
    features.update(estimate_speech_rate(y, sr, transcript_text))

    return features


def extract_features_from_transcript_segments(
    transcript_path: str,
    audio_path: str,
    output_path: str,
    pitch_mode: str = "full",
):
    """
    Extract features for every segment in a transcript JSON using one audio file.

    This assumes the transcript has sentence rows with:
    seq_id, start_time, end_time, text
    """
    transcript_path = Path(transcript_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with transcript_path.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    call_id = transcript.get("call_id", transcript_path.stem)
    domain = transcript.get("domain")

    sentences = transcript.get("sentences", [])

    print(f"Loading full audio once: {audio_path}")
    full_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio loaded. Duration: {round(len(full_audio) / sr, 2)} seconds")

    segment_features: List[Dict[str, Any]] = []

    for idx, sentence in enumerate(sentences):
        start_time = (
            sentence.get("start_time")
            or sentence.get("start")
            or sentence.get("start_timestamp")
            or sentence.get("startTime")
            or sentence.get("start_seconds")
        )

        end_time = (
            sentence.get("end_time")
            or sentence.get("end")
            or sentence.get("end_timestamp")
            or sentence.get("endTime")
            or sentence.get("end_seconds")
        )

        text = (
            sentence.get("text")
            or sentence.get("sentence")
            or sentence.get("transcript")
            or ""
        )
        speaker = sentence.get("speaker")

        # Some transcripts have the first segment with missing start time.
        # If it is the first segment and end_time exists, assume it starts at 0.0.
        if start_time is None and idx == 0 and end_time is not None:
            start_time = 0.0

        if start_time is None or end_time is None:
            segment_features.append({
                "segment_index": idx + 1,
                "segment_key": f"{call_id}_{idx + 1:04d}",
                "seq_id": sentence.get("seq_id"),
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "processing_status": "missing_timestamps",
                "audio_features": None,
            })
            continue

        try:
            y_segment = slice_audio_segment(
                full_audio=full_audio,
                sr=sr,
                start_time=float(start_time),
                end_time=float(end_time),
            )

            features = extract_audio_features_from_array(
                y=y_segment,
                sr=sr,
                transcript_text=text,
                pitch_mode=pitch_mode,
            )

            segment_features.append({
                "segment_index": idx + 1,
                "segment_key": f"{call_id}_{idx + 1:04d}",
                "seq_id": sentence.get("seq_id"),
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "processing_status": "success",
                "audio_features": features,
            })

        except Exception as e:
            segment_features.append({
                "segment_index": idx + 1,
                "segment_key": f"{call_id}_{idx + 1:04d}",
                "seq_id": sentence.get("seq_id"),
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "processing_status": "feature_extraction_error",
                "error": str(e),
                "audio_features": None,
            })

    output = {
        "call_id": call_id,
        "domain": domain,
        "audio_path": audio_path,
        "feature_extraction_version": "audio_features_v1_librosa",
        "total_segments": len(segment_features),
        "segments": segment_features,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved audio features to: {output_path}")
    print(f"Total segments: {len(segment_features)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--transcript-path", default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument(
    "--pitch-mode",
    choices=["full", "fast", "skip"],
    default="full",
    help="Pitch extraction mode: full is accurate but slow, fast is quicker, skip is fastest.",
)

    args = parser.parse_args()

    if args.transcript_path:
        extract_features_from_transcript_segments(
            transcript_path=args.transcript_path,
            audio_path=args.audio_path,
            output_path=args.output_path,
            pitch_mode=args.pitch_mode,
        )
    else:
        features = extract_audio_features(
            audio_path=args.audio_path,
            start_time=args.start_time,
            end_time=args.end_time,
            transcript_text=args.text,
        )

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)

        print(json.dumps(features, indent=2))
        print(f"Saved audio features to: {output_path}")


if __name__ == "__main__":
    main()