"""
Phase 2 audio preprocessing: high-pass filter + loudness normalization.

Applied PER CHANNEL, before transcription. Both operations are zero-phase or
gain-only, so word-level timestamps are unaffected (critical -- we worked hard
for accurate timestamps and won't shift them here).

High-pass filter
----------------
Zero-phase Butterworth high-pass (sosfiltfilt -> no group delay -> no timestamp
shift). Removes DC offset, sub-bass rumble, and mains hum below the cutoff.

NOTE: spectral analysis of this dataset showed ~0% energy below 80 Hz -- these
recordings were already high-passed upstream. So on THIS data the filter is
near-neutral. It is kept because (a) it is correct, robust practice, (b) it is
cheap insurance for any future recording that does contain rumble, and (c) it
removes any residual DC before the loudness measurement.

Loudness normalization
----------------------
The real lever. Active-RMS normalization: estimate the speech level from
speech frames only (frames within `rel_db` of the loudest frame, so the estimate
is immune to the channel's silence fraction), then apply a single gain to hit
`target_dbfs`. Quiet channels (one was at -37 dBFS active RMS) get boosted toward
the level Whisper's VAD and encoder expect; already-loud channels barely move.

Safety: gain is capped (`max_gain_db`) so a near-silent channel is not blown up,
and the output peak is limited to `peak_ceiling` to prevent clipping.

We deliberately do NOT use pyloudnorm/LUFS (extra dependency, and gated LUFS is
overkill for mono speech) nor spectral denoising (Whisper is trained on noisy
audio; aggressive denoising tends to hurt -- that is a separate, measured
experiment, not a default).
"""

import numpy as np
from scipy import signal

SR = 16000


def highpass(audio, sr=SR, cutoff=80.0, order=2):
    """Zero-phase Butterworth high-pass. No timestamp shift."""
    if len(audio) < 32:
        return audio
    sos = signal.butter(order, cutoff, btype="high", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def active_rms(audio, sr=SR, frame_ms=20, rel_db=25.0):
    """
    Speech-level estimate: RMS over frames within `rel_db` of the loudest frame.
    Relative threshold -> robust to how much of the channel is silence.
    """
    fl = int(frame_ms / 1000 * sr)
    if len(audio) < fl:
        return float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    n = len(audio) // fl
    fr = audio[: n * fl].reshape(n, fl)
    fe = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    thr = fe.max() * (10 ** (-rel_db / 20))
    speech = fe >= thr
    if not speech.any():
        return float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    return float(np.sqrt((fr[speech] ** 2).mean() + 1e-12))


def soft_limit(x, knee=0.85, ceiling=0.98):
    """
    Soft knee limiter. Samples below `knee` pass through linearly (the bulk of
    speech); samples above are tanh-compressed toward `ceiling`. This saturates
    only the rare transient peaks instead of scaling the whole signal down --
    so a quiet channel with one click/pop can still reach target loudness.
    """
    ax = np.abs(x)
    over = ax > knee
    if not over.any():
        return x
    out = x.astype(np.float32).copy()
    s = np.sign(x[over])
    out[over] = s * (knee + (ceiling - knee) * np.tanh((ax[over] - knee) / (ceiling - knee)))
    return out


def loudness_normalize(audio, sr=SR, target_dbfs=-20.0, max_gain_db=25.0):
    """Scale to target active-RMS level (gain-capped), then soft-limit peaks."""
    lvl = active_rms(audio, sr)
    if lvl <= 1e-9:
        return audio
    gain_db = min(target_dbfs - 20 * np.log10(lvl), max_gain_db)
    out = audio * (10 ** (gain_db / 20))
    return soft_limit(out).astype(np.float32)


def preprocess(audio, sr=SR, do_highpass=True, do_loudness=True,
               hp_cutoff=80.0, target_dbfs=-20.0, loudness_gate_db=-26.0):
    """
    High-pass then conditional loudness normalization.

    Loudness normalization is only applied when the channel's active RMS is
    below `loudness_gate_db` (default -26 dBFS). Channels already at reasonable
    levels are left untouched, avoiding transcription regressions caused by
    over-amplifying carefully-spoken digits or quiet call openings.

    Order matters: high-pass first to remove DC/rumble before measuring level.
    """
    if do_highpass:
        audio = highpass(audio, sr, hp_cutoff)
    if do_loudness:
        lvl = 20 * np.log10(active_rms(audio, sr) + 1e-12)
        if lvl < loudness_gate_db:
            audio = loudness_normalize(audio, sr, target_dbfs)
    return audio


if __name__ == "__main__":
    import os, json, soundfile as sf
    DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"
    with open(os.path.join(DATA, "manifest.json")) as f:
        man = {m["call_id"]: m for m in json.load(f)}
    print("Self-test (active RMS dBFS, before -> after preprocess):")
    for cid in ["en_US_General_Health_1587175", "en_CA_Banking_1588683", "en_CA_Banking_1586889"]:
        m = man[cid]
        for role, key in [("agent", "agent_wav"), ("customer", "customer_wav")]:
            a, sr = sf.read(os.path.join(DATA, m[key]), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            before = 20 * np.log10(active_rms(a, sr) + 1e-12)
            out = preprocess(a, sr)
            after = 20 * np.log10(active_rms(out, sr) + 1e-12)
            peak_after = 20 * np.log10(float(np.abs(out).max()) + 1e-12)
            print(f"  {cid:<30} {role:<9} {before:6.1f} -> {after:6.1f} dBFS  (peak {peak_after:5.1f})")
