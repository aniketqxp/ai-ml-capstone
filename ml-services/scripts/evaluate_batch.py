"""
Aggregate WER evaluation over the NA test set.

Reads manifest.json (references) + results/*/*.json (hypotheses), computes
per-call agent/customer WER (raw + Whisper-normalised), then aggregates:
  - per-accent mean WER and accuracy
  - overall (word-weighted) WER and accuracy
  - distribution: median, worst calls

Usage:
  python evaluate_batch.py
  python evaluate_batch.py --csv results_summary.csv
"""

import os
import json
import argparse
import statistics
import jiwer
from collections import defaultdict
from eval_common import normalise, normalise_raw

DATA_DIR = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
RESULTS  = os.path.join(DATA_DIR, "results")


def weighted_wer(ref, hyp, norm_fn):
    r, h = norm_fn(ref), norm_fn(hyp)
    n = len(r.split())
    if n == 0:
        return None, 0
    return jiwer.wer(r, h), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}

    rows = []                                   # per-call records
    for accent_dir in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        adir = os.path.join(RESULTS, accent_dir)
        if not os.path.isdir(adir):
            continue
        for fn in sorted(os.listdir(adir)):
            if not fn.endswith(".json"):
                continue
            cid = fn[:-5]
            if cid not in manifest:
                continue
            with open(os.path.join(adir, fn), encoding="utf-8") as f:
                hyp = json.load(f)
            m = manifest[cid]
            ha = " ".join(w["word"] for w in hyp["agent"])
            hc = " ".join(w["word"] for w in hyp["customer"])

            rec = {"call_id": cid, "accent": m["accent"], "domain": m["domain"]}
            for tag, norm_fn in [("raw", normalise_raw), ("norm", normalise)]:
                wa, na = weighted_wer(m["agent_transcript"],    ha, norm_fn)
                wc, nc = weighted_wer(m["customer_transcript"], hc, norm_fn)
                tot_n = na + nc
                tot_w = ((wa or 0) * na + (wc or 0) * nc) / tot_n if tot_n else None
                rec[f"{tag}_wer"]   = tot_w
                rec[f"{tag}_words"] = tot_n
            rows.append(rec)

    if not rows:
        print("No results found. Run run_batch.py first.")
        return

    # ── Per-accent aggregation (word-weighted) ────────────────────────────────
    print(f"Evaluated {len(rows)} calls\n")
    print(f"{'Accent':<16} {'Calls':>5} {'RawAcc':>8} {'NormAcc':>8} {'NormWER':>8} {'MedWER':>8}")
    print("-" * 60)

    def agg(subset, tag):
        num = sum(r[f"{tag}_wer"] * r[f"{tag}_words"] for r in subset)
        den = sum(r[f"{tag}_words"] for r in subset)
        return num / den if den else None

    by_accent = defaultdict(list)
    for r in rows:
        by_accent[r["accent"]].append(r)

    for accent in sorted(by_accent):
        s = by_accent[accent]
        raw_w  = agg(s, "raw")
        norm_w = agg(s, "norm")
        med    = statistics.median([r["norm_wer"] for r in s])
        print(f"{accent:<16} {len(s):>5} {(1-raw_w)*100:>7.1f}% {(1-norm_w)*100:>7.1f}% "
              f"{norm_w*100:>7.1f}% {med*100:>7.1f}%")

    raw_all  = agg(rows, "raw")
    norm_all = agg(rows, "norm")
    print("-" * 60)
    print(f"{'OVERALL':<16} {len(rows):>5} {(1-raw_all)*100:>7.1f}% {(1-norm_all)*100:>7.1f}% "
          f"{norm_all*100:>7.1f}%")
    print(f"\nTarget: >= 90% normalised accuracy  ->  "
          f"{'PASS' if (1-norm_all) >= 0.90 else 'NEEDS WORK'}")

    # ── Worst 5 calls (normalised) ────────────────────────────────────────────
    worst = sorted(rows, key=lambda r: r["norm_wer"], reverse=True)[:5]
    print("\nWorst 5 calls (normalised WER):")
    for r in worst:
        print(f"  {r['accent']:<14} {r['call_id']:<32} {r['norm_wer']*100:5.1f}%  ({r['domain']})")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-call CSV written: {args.csv}")


if __name__ == "__main__":
    main()
