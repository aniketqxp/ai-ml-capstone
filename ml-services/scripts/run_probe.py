"""
Run a transcription method over the FROZEN PROBE SET only (12 calls, ~30 min),
not the full 126-call set. This is the fast-iteration loop.

Writes per-call JSON to <out_dir>/<accent>/<call_id>.json. Resumable.

Usage:
  python run_probe.py --method channels --out results_channels --model small.en
"""

import os
import json
import time
import argparse
from faster_whisper import WhisperModel

import transcribe_channels

DATA_DIR  = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA_DIR, "manifest.json")
PROBE_SET = os.path.join(DATA_DIR, "probe_set.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="channels", choices=["channels"])
    ap.add_argument("--out",    default=None,
                    help="output dir (default: results_channels, or results_channels_pp if --preprocess)")
    ap.add_argument("--model",  default="small.en")
    ap.add_argument("--preprocess",  action="store_true", help="Phase 2: high-pass + loudness norm")
    ap.add_argument("--no-highpass", action="store_true", help="ablation: skip high-pass")
    ap.add_argument("--no-loudness", action="store_true", help="ablation: skip loudness norm")
    ap.add_argument("--hp-cutoff",   type=float, default=80.0)
    ap.add_argument("--target-dbfs", type=float, default=-20.0)
    ap.add_argument("--full", action="store_true",
                    help="run the whole 126-call manifest, not just the 12-call probe (resumable)")
    args = ap.parse_args()

    preprocess_fn = None
    if args.preprocess:
        import functools, audio_preprocess
        preprocess_fn = functools.partial(
            audio_preprocess.preprocess,
            do_highpass=not args.no_highpass,
            do_loudness=not args.no_loudness,
            hp_cutoff=args.hp_cutoff,
            target_dbfs=args.target_dbfs,
        )
    if args.out is None:
        args.out = "results_channels_pp" if args.preprocess else "results_channels"

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}

    if args.full:
        # whole 126-call set (resumable; skips calls already written)
        probe = [{"call_id": m["call_id"], "accent": m["accent"],
                  "domain": m["domain"], "tier": "full"} for m in manifest.values()]
    else:
        with open(PROBE_SET, encoding="utf-8") as f:
            probe = json.load(f)["calls"]

    out_root = os.path.join(DATA_DIR, args.out)

    todo = []
    for p in probe:
        out = os.path.join(out_root, p["accent"], p["call_id"] + ".json")
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            todo.append(p)

    print(f"Probe calls: {len(probe)} | done: {len(probe)-len(todo)} | to do: {len(todo)}")
    print(f"Method: {args.method} | model: {args.model} | out: {args.out}/")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading faster-whisper {args.model}/int8 (once)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    t_start = time.time()
    for i, p in enumerate(todo, 1):
        cid = p["call_id"]
        m = manifest[cid]
        a_wav = os.path.join(DATA_DIR, m["agent_wav"])
        c_wav = os.path.join(DATA_DIR, m["customer_wav"])
        out_dir = os.path.join(out_root, p["accent"])
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, cid + ".json")

        t0 = time.time()
        try:
            aw, cw, segs, dur = transcribe_channels.transcribe_call(
                model, a_wav, c_wav, preprocess_fn=preprocess_fn)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {cid}  ERROR: {e}")
            continue

        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "model":  f"faster-whisper {args.model}/int8",
                "method": "per_channel",
                "call_id": cid, "accent": p["accent"], "domain": p["domain"],
                "agent": aw, "customer": cw,
                "segments": segs,
            }, f, indent=2)

        elapsed = time.time() - t0
        print(f"  [{i}/{len(todo)}] {p['tier']:<13} {cid:<32} "
              f"{dur:5.0f}s audio | {elapsed:5.1f}s | A:{len(aw)} C:{len(cw)}")

    print(f"\nDone. {len(todo)} calls in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
