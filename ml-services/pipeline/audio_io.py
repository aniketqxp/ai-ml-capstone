"""
Audio ingest helpers built on ffmpeg/ffprobe.

A fresh call arrives either as one stereo file (agent = ch0, customer = ch1,
the apptek convention) or as two mono files already split by channel. The
pipeline downstream of here always works on two mono 16 kHz WAVs (one per
speaker), so this module normalizes both shapes to that.
"""
import json
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000   # wav2vec2 + faster-whisper canonical rate


class AudioError(RuntimeError):
    pass


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"{cmd[0]} failed:\n{proc.stderr[-800:]}")
    return proc


def probe_channels(path):
    """Number of audio channels in the first audio stream (via ffprobe)."""
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels", "-of", "json", str(path),
    ])
    info = json.loads(proc.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise AudioError(f"no audio stream in {path}")
    return int(streams[0]["channels"])


def to_mono_16k(src, out):
    """Downmix/resample any input to a single-channel 16 kHz WAV."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(src),
          "-ac", "1", "-ar", str(SAMPLE_RATE), str(out)])
    return str(out)


def split_stereo(src, out_agent, out_customer):
    """
    Split a 2-channel file into two mono 16 kHz WAVs: ch0 -> agent, ch1 ->
    customer (apptek stereo attribution). Returns (agent_path, customer_path).
    """
    out_agent, out_customer = Path(out_agent), Path(out_customer)
    for p in (out_agent, out_customer):
        p.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[l][r]",
        "-map", "[l]", "-ac", "1", "-ar", str(SAMPLE_RATE), str(out_agent),
        "-map", "[r]", "-ac", "1", "-ar", str(SAMPLE_RATE), str(out_customer),
    ])
    return str(out_agent), str(out_customer)


def prepare_channels(primary, out_agent, out_customer, secondary=None):
    """
    Normalize an upload to (agent_wav, customer_wav) mono 16 kHz.

      - two files given            -> primary=agent, secondary=customer (resampled)
      - one 2-channel file         -> channel-split
      - one 1-channel file, no 2nd -> ambiguous, caller must reject (raises)
    """
    if secondary is not None:
        return (to_mono_16k(primary, out_agent),
                to_mono_16k(secondary, out_customer))
    ch = probe_channels(primary)
    if ch >= 2:
        return split_stereo(primary, out_agent, out_customer)
    # Mixed mono calls need the same waveform available for both inferred
    # speaker roles. Runtime role reconstruction assigns timestamp ranges after
    # transcription; acoustic analysis then slices the matching ranges from
    # these two identical source tracks.
    to_mono_16k(primary, out_agent)
    to_mono_16k(primary, out_customer)
    return str(out_agent), str(out_customer)
