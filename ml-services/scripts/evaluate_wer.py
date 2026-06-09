"""
WER evaluation for a single call's transcription.

Reports TWO metrics for honesty:
  - Raw WER       : lowercase + strip punctuation only (conservative)
  - Normalised WER: Whisper EnglishTextNormalizer (semantic; numbers/contractions
                    canonicalised). This is the standard ASR-benchmark metric.

Usage:
  python evaluate_wer.py [--call call_1] [--hypo path/to/transcript_data.json]
"""

import json
import argparse
import jiwer
from eval_common import normalise, normalise_raw

METADATA = r"d:\Desktop\ai-ml-capstone\data\apptek\metadata.json"


def wer_for(ref, hyp):
    out = jiwer.process_words(ref, hyp)
    return out.wer, out.substitutions, out.deletions, out.insertions, out.hits


def report(ref_text, hyp_text, label, norm_fn):
    ref, hyp = norm_fn(ref_text), norm_fn(hyp_text)
    wer, S, D, I, H = wer_for(ref, hyp)
    n = len(ref.split())
    print(f"    {label:<10} WER {wer*100:5.1f}%  (acc {(1-wer)*100:4.1f}%)  "
          f"[S={S} D={D} I={I} ref_words={n}]")
    return wer, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", default="call_1")
    ap.add_argument("--hypo", default=r"d:\Desktop\ai-ml-capstone\frontend\transcript_data.json")
    args = ap.parse_args()

    with open(METADATA, encoding="utf-8") as f:
        meta = {c["call_id"]: c for c in json.load(f)}
    with open(args.hypo, encoding="utf-8") as f:
        hypo = json.load(f)

    call = meta[args.call]
    hyp_agent    = " ".join(w["word"] for w in hypo["agent"])
    hyp_customer = " ".join(w["word"] for w in hypo["customer"])

    print(f"Model : {hypo.get('model','?')}")
    print(f"Call  : {args.call}")
    print("=" * 64)

    results = {}
    for norm_name, norm_fn in [("RAW", normalise_raw), ("NORMALISED", normalise)]:
        print(f"\n  --- {norm_name} ---")
        wa, na = report(call["agent_transcript"],    hyp_agent,    "Agent",    norm_fn)
        wc, nc = report(call["customer_transcript"], hyp_customer, "Customer", norm_fn)
        overall = (wa * na + wc * nc) / (na + nc)
        print(f"    {'Overall':<10} WER {overall*100:5.1f}%  (acc {(1-overall)*100:4.1f}%)")
        results[norm_name] = 1 - overall

    print("\n" + "=" * 64)
    print(f"  Raw accuracy        : {results['RAW']*100:.1f}%")
    print(f"  Normalised accuracy : {results['NORMALISED']*100:.1f}%   (target >= 90%)")
    status = "PASS" if results["NORMALISED"] >= 0.90 else "NEEDS WORK"
    print(f"  Status              : {status}")
    print("=" * 64)


if __name__ == "__main__":
    main()
