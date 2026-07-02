"""
Batch transcription over the NA test set.

For each call in manifest.json:
  - load agent (ch1) + customer (ch2) WAVs
  - stack into stereo IN MEMORY (no mixed file written to disk)
  - transcribe the mono mix with faster-whisper + word timestamps
  - attribute each segment to agent/customer by channel energy
  - write results/<accent>/<call_id>.json

Resumable: skips calls whose result JSON already exists.
The model is loaded ONCE and reused across all calls.

Usage:
  python run_batch.py --model small.en
  python run_batch.py --model small.en --accent en-CA      # subset
"""

import os
import json
import time
import argparse
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

DATA_DIR = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
RESULTS  = os.path.join(DATA_DIR, "results")
SR       = 16000

INITIAL_PROMPT = (
    "Banking and customer service call between an agent and a customer. "
    "Topics include accounts, transfers, payments, balances, and account numbers."
)


def load_mono_and_stereo(agent_path, customer_path):
    a, sr_a = sf.read(agent_path,    dtype="float32")
    c, sr_c = sf.read(customer_path, dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    if c.ndim > 1: c = c.mean(axis=1)
    n = min(len(a), len(c))
    a, c = a[:n], c[:n]
    stereo = np.stack([a, c], axis=1)   # ch0=agent, ch1=customer
    mono   = stereo.mean(axis=1)
    return mono, stereo, (sr_a == SR and sr_c == SR)


def speaker_for(stereo, start_s, end_s, pad=0.05):
    s = max(0, int((start_s - pad) * SR))
    e = min(len(stereo), int((end_s + pad) * SR))
    if e - s < int(0.1 * SR):
        mid = (s + e) // 2
        h = int(0.05 * SR)
        s, e = max(0, mid - h), min(len(stereo), mid + h)
    chunk = stereo[s:e]
    return "agent" if np.sqrt(np.mean(chunk[:, 0]**2)) >= np.sqrt(np.mean(chunk[:, 1]**2)) else "customer"


def transcribe_call(model, agent_path, customer_path):
    mono, stereo, ok_sr = load_mono_and_stereo(agent_path, customer_path)
    seg_gen, info = model.transcribe(
        mono, language="en", beam_size=5,
        word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=INITIAL_PROMPT,
    )
    agent_words, customer_words = [], []
    for seg in seg_gen:
        if not seg.words:
            continue
        bucket = agent_words if speaker_for(stereo, seg.start, seg.end) == "agent" else customer_words
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            bucket.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
    return agent_words, customer_words, len(mono) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--accent", default=None, help="only run this accent")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.accent:
        manifest = [m for m in manifest if m["accent"] == args.accent]

    todo = []
    for m in manifest:
        out = os.path.join(RESULTS, m["accent"], m["call_id"] + ".json")
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            todo.append(m)

    print(f"Total calls: {len(manifest)} | already done: {len(manifest)-len(todo)} | to do: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading faster-whisper {args.model}/int8 (once)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    t_start = time.time()
    for i, m in enumerate(todo, 1):
        a_wav = os.path.join(DATA_DIR, m["agent_wav"])
        c_wav = os.path.join(DATA_DIR, m["customer_wav"])
        out_dir = os.path.join(RESULTS, m["accent"])
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, m["call_id"] + ".json")

        t0 = time.time()
        try:
            aw, cw, dur = transcribe_call(model, a_wav, c_wav)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {m['call_id']}  ERROR: {e}")
            continue

        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "model": f"faster-whisper {args.model}/int8 + stereo channel attribution",
                "call_id": m["call_id"], "accent": m["accent"], "domain": m["domain"],
                "agent": aw, "customer": cw,
            }, f, indent=2)

        elapsed = time.time() - t0
        avg = (time.time() - t_start) / i
        eta = avg * (len(todo) - i)
        print(f"  [{i}/{len(todo)}] {m['accent']:<14} {m['call_id']:<32} "
              f"{dur:5.0f}s audio | {elapsed:5.1f}s | A:{len(aw)} C:{len(cw)} | "
              f"ETA {eta/60:5.1f}m")

    print(f"\nDone. {len(todo)} calls in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
