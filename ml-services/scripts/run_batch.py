"""
Batch transcription over the NA test set using per-channel isolation.

Each call has separate agent and customer WAV files — we transcribe each
channel independently so channel identity IS the speaker label.
No RMS attribution guessing, no cross-speaker leakage.

Usage:
  python run_batch.py --model small.en
  python run_batch.py --model small.en --accent en-CA
  python run_batch.py --model small.en --domains health telecom
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
from transcribe_channels import transcribe_call
import paths

DATA_DIR = str(paths.NA_TESTSET)
MANIFEST = str(paths.MANIFEST)
RESULTS  = os.path.join(DATA_DIR, "results")

DOMAIN_PROMPTS = {
    "banking": (
        "Banking and customer service call between an agent and a customer. "
        "Topics include accounts, transfers, payments, balances, credit cards, "
        "loans, account numbers, and online banking."
    ),
    "health": (
        "Healthcare customer service call between an agent and a patient or caller. "
        "Topics include appointments, medical records, prescriptions, insurance "
        "coverage, billing, referrals, and health plan benefits."
    ),
    "telecom": (
        "Telecommunications customer service call between an agent and a customer. "
        "Topics include mobile plans, internet service, data usage, billing, "
        "account management, and technical support."
    ),
}
FALLBACK_PROMPT = "Customer service call between an agent and a customer."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="small.en")
    ap.add_argument("--accent",  default=None)
    ap.add_argument("--domains", nargs="+", default=None)
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    if args.accent:
        manifest = [m for m in manifest if m["accent"] == args.accent]
    if args.domains:
        manifest = [m for m in manifest if m.get("domain", "").lower() in args.domains]

    todo = []
    for m in manifest:
        out = os.path.join(RESULTS, m["accent"], m["call_id"] + ".json")
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            todo.append(m)

    print(f"Total calls: {len(manifest)} | done: {len(manifest)-len(todo)} | to do: {len(todo)}")
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

        domain = m.get("domain", "").lower()
        prompt = DOMAIN_PROMPTS.get(domain, FALLBACK_PROMPT)

        t0 = time.time()
        try:
            agent_words, customer_words, _, duration = transcribe_call(
                model, a_wav, c_wav,
                decode={"initial_prompt": prompt},
            )
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {m['call_id']}  ERROR: {e}")
            continue

        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "model":    f"faster-whisper {args.model}/int8 + per-channel",
                "call_id":  m["call_id"],
                "accent":   m["accent"],
                "domain":   m["domain"],
                "agent":    agent_words,
                "customer": customer_words,
            }, f, indent=2)

        elapsed = time.time() - t0
        avg = (time.time() - t_start) / i
        eta = avg * (len(todo) - i)
        print(f"  [{i}/{len(todo)}] {m['accent']:<14} {m['call_id']:<36} "
              f"{duration:5.0f}s | {elapsed:5.1f}s | A:{len(agent_words)} C:{len(customer_words)} | "
              f"ETA {eta/60:5.1f}m")

    print(f"\nDone. {len(todo)} calls in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
