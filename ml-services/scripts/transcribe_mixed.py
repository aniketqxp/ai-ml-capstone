"""
Transcribes call_1_mixed.wav (stereo: ch0=agent, ch1=customer).

Pipeline
--------
1. Load stereo with soundfile  — no ffmpeg, no memory thrashing
2. Mono mix → faster-whisper base/int8 with word_timestamps=True + Silero VAD
3. Per-segment speaker attribution via channel energy (RMS ch0 vs ch1)
4. Write frontend/transcript_data.json in the format the dashboard expects

Why this works
--------------
- Isolated-channel files are 84% silence → Whisper timestamp drift
- Mono mix is only 42.5% silence → stable chunk-level VAD → accurate timestamps
- word_timestamps=True uses DTW in a single pass — no second model, no OOM
- Channel energy attribution is near-perfect because the stereo channels
  are exact agent/customer isolations (correlation = 1.000 each)
"""

import os
import json
import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIXED  = os.path.join(ROOT, "data", "apptek", "call_1_mixed.wav")
OUT    = os.path.join(ROOT, "frontend", "transcript_data.json")
SR     = 16000

# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_stereo(path):
    data, sr = sf.read(path, dtype="float32")
    assert sr == SR,              f"Expected {SR} Hz, got {sr}"
    assert data.ndim == 2 and data.shape[1] == 2, f"Expected stereo, got {data.shape}"
    return data  # (n_samples, 2) — ch0 = agent, ch1 = customer


def speaker_for_segment(stereo, start_s, end_s, pad_s=0.05):
    """
    Returns 'agent' or 'customer' by comparing RMS energy of ch0 vs ch1
    over [start_s - pad_s, end_s + pad_s].  Minimum window: 100 ms.
    """
    s = max(0, int((start_s - pad_s) * SR))
    e = min(len(stereo), int((end_s   + pad_s) * SR))
    if e - s < int(0.1 * SR):              # guarantee at least 100 ms
        mid = (s + e) // 2
        half = int(0.05 * SR)
        s, e = max(0, mid - half), min(len(stereo), mid + half)
    chunk = stereo[s:e]
    rms0 = np.sqrt(np.mean(chunk[:, 0] ** 2))
    rms1 = np.sqrt(np.mean(chunk[:, 1] ** 2))
    return "agent" if rms0 >= rms1 else "customer"

# ── Main ──────────────────────────────────────────────────────────────────────

def run(mixed_path=MIXED, out_path=OUT, model_size="base", call_id="call_1"):
    # 1. Load audio
    print("Loading stereo audio (soundfile — no ffmpeg)...")
    stereo = load_stereo(mixed_path)
    mono   = stereo.mean(axis=1)          # shape: (n_samples,)
    duration = len(mono) / SR
    print(f"  {duration:.1f}s  |  ch0 = agent  |  ch1 = customer")

    # 2. Load model
    print(f"\nLoading faster-whisper {model_size}/int8...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # 3. Transcribe mono mix
    # initial_prompt primes Whisper's vocabulary with domain-specific terms,
    # improving accuracy on proper nouns, numbers and banking terminology.
    initial_prompt = (
        "Banking customer service call. "
        "Agent Emily at North Star Bank. Customer James Carter. "
        "Topics: checking account, savings account, automatic transfer, "
        "Canadian dollars, account number, date of birth, mobile app, "
        "online banking, scheduled transfer, four hundred dollars."
    )

    print("Transcribing (word_timestamps=True, vad_filter=True, initial_prompt=True)...")
    t0 = time.time()
    seg_gen, info = model.transcribe(
        mono,
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=initial_prompt,
    )
    segments = list(seg_gen)
    elapsed  = time.time() - t0
    print(f"  {len(segments)} segments | {elapsed:.1f}s "
          f"| lang={info.language} p={info.language_probability:.2f}")

    # 4. Build per-speaker word lists
    agent_words    = []
    customer_words = []
    no_ts          = 0

    for seg in segments:
        if not seg.words:
            continue

        speaker = speaker_for_segment(stereo, seg.start, seg.end)
        bucket  = agent_words if speaker == "agent" else customer_words

        for w in seg.words:
            if w.start is None or w.end is None:
                no_ts += 1
                continue
            bucket.append({
                "word":  w.word.strip(),
                "start": round(w.start, 3),
                "end":   round(w.end,   3),
            })

    print(f"\n  Agent words    : {len(agent_words)}")
    print(f"  Customer words : {len(customer_words)}")
    if no_ts:
        print(f"  Skipped (no timestamp): {no_ts}")

    # 5. Spot-check a few lines
    def fmt(words, n=10):
        return "  " + "  ".join(w["word"] for w in words[:n])

    print("\n--- Agent start ---")
    print(fmt(agent_words))
    print("\n--- Customer start ---")
    print(fmt(customer_words))

    # 6. Sanity check near the 4:48 mark (288 s)
    target = 288.0
    agent_set = set(id(w) for w in agent_words)
    window = [w for w in (agent_words + customer_words)
              if abs(w["start"] - target) < 15]
    window.sort(key=lambda w: w["start"])
    if window:
        print("\n--- Words near 4:48 ---")
        for w in window:
            src = "AGT" if id(w) in agent_set else "CST"
            print(f"  [{src}] {w['start']:7.3f}s  {w['word']}")

    # 7. Save
    payload = {
        "model":    f"faster-whisper {model_size}/int8 + stereo channel attribution",
        "call":     call_id,
        "agent":    agent_words,
        "customer": customer_words,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="faster-whisper model size (tiny/base/small/medium)")
    parser.add_argument("--call",  default="call_1")
    args = parser.parse_args()
    run(model_size=args.model, call_id=args.call)
