"""
A/B comparison on the frozen probe set. Compares two result dirs without
re-transcribing. Default: per-channel (A) vs per-channel+preprocess (B).
The 'base' column is the static mono-mix baseline_acc recorded in probe_set.json.

Also reports:
  - no_speech_prob diagnostics per dir (mean, #high-ns segments, total words) to
    show whether preprocessing reduced Whisper's silence-hallucination signal
  - a confidence-filter experiment on B (drop words with segment ns > threshold)

Usage:
  python compare_probe.py                               # chan vs chan+pp
  python compare_probe.py --a-dir results --a-label mix --b-dir results_channels --b-label chan
"""

import os
import json
import argparse
import jiwer
from eval_common import normalise

DATA_DIR  = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA_DIR, "manifest.json")
PROBE_SET = os.path.join(DATA_DIR, "probe_set.json")


def load_result(root, accent, cid):
    p = os.path.join(DATA_DIR, root, accent, cid + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def text_of(words, ns_max=None):
    if ns_max is None:
        return " ".join(w["word"] for w in words)
    return " ".join(w["word"] for w in words if w.get("ns", 0.0) <= ns_max)


def score_call(ref_a, ref_c, hyp_a, hyp_c):
    ra, rc = normalise(ref_a), normalise(ref_c)
    ha, hc = normalise(hyp_a), normalise(hyp_c)
    na, nc = len(ra.split()), len(rc.split())
    if na + nc == 0:
        return None, 0
    wa = jiwer.wer(ra, ha) if na else 0.0
    wc = jiwer.wer(rc, hc) if nc else 0.0
    return (wa * na + wc * nc) / (na + nc), na + nc


def acc(wer):
    return None if wer is None else (1 - wer) * 100


def ns_diag(result):
    """(#high-ns segments, total segments, total hyp words) from captured segments."""
    if not result or "segments" not in result:
        return None
    segs = result["segments"].get("agent", []) + result["segments"].get("customer", [])
    hi = sum(1 for s in segs if s.get("no_speech_prob", 0) > 0.6)
    words = len(result.get("agent", [])) + len(result.get("customer", []))
    return hi, len(segs), words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", default="results_channels"); ap.add_argument("--a-label", default="chan")
    ap.add_argument("--b-dir", default="results_channels_pp"); ap.add_argument("--b-label", default="chan+pp")
    ap.add_argument("--ns-thresholds", nargs="*", type=float, default=[0.6, 0.8])
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    with open(PROBE_SET, encoding="utf-8") as f:
        probe = json.load(f)["calls"]

    print(f"{'tier':<13} {'call_id':<32} {'base':>6} {args.a_label:>7} {args.b_label:>8} {'delta':>7}")
    print("-" * 80)

    agg = {"a": [0.0, 0], "b": [0.0, 0]}
    agg_ns = {t: [0.0, 0] for t in args.ns_thresholds}
    nsA = [0, 0, 0]; nsB = [0, 0, 0]   # hi, segs, words
    missing = []

    for p in probe:
        cid, accent = p["call_id"], p["accent"]
        m = manifest[cid]
        ra_, rc_ = m["agent_transcript"], m["customer_transcript"]
        A = load_result(args.a_dir, accent, cid)
        B = load_result(args.b_dir, accent, cid)
        if A is None or B is None:
            missing.append((cid, args.a_dir if A is None else args.b_dir))
            continue

        a_wer, n = score_call(ra_, rc_, text_of(A["agent"]), text_of(A["customer"]))
        b_wer, _ = score_call(ra_, rc_, text_of(B["agent"]), text_of(B["customer"]))
        agg["a"][0] += a_wer * n; agg["a"][1] += n
        agg["b"][0] += b_wer * n; agg["b"][1] += n

        for t in args.ns_thresholds:
            w, _ = score_call(ra_, rc_, text_of(B["agent"], t), text_of(B["customer"], t))
            agg_ns[t][0] += w * n; agg_ns[t][1] += n

        for store, res in [(nsA, A), (nsB, B)]:
            d = ns_diag(res)
            if d:
                store[0] += d[0]; store[1] += d[1]; store[2] += d[2]

        delta = acc(b_wer) - acc(a_wer)
        flag = "  <<" if delta >= 3 else ("  !!" if delta <= -3 else "")
        print(f"{p['tier']:<13} {cid:<32} {p['baseline_acc']*100:>5.1f}% "
              f"{acc(a_wer):>6.1f}% {acc(b_wer):>7.1f}% {delta:>+6.1f}{flag}")

    print("-" * 80)
    a_acc = acc(agg["a"][0] / agg["a"][1])
    b_acc = acc(agg["b"][0] / agg["b"][1])
    print(f"{'OVERALL (word-weighted)':<46} {a_acc:>6.1f}% {b_acc:>7.1f}% {b_acc - a_acc:>+6.1f}")

    print(f"\nno_speech_prob diagnostics (hallucination signal):")
    print(f"  {args.a_label:<10} high-ns segs: {nsA[0]:>4} / {nsA[1]:<5} segments | total hyp words: {nsA[2]}")
    print(f"  {args.b_label:<10} high-ns segs: {nsB[0]:>4} / {nsB[1]:<5} segments | total hyp words: {nsB[2]}")

    print(f"\nConfidence filter on {args.b_label} (drop words where segment no_speech_prob > t):")
    print(f"  no filter        : {b_acc:5.1f}%")
    for t in args.ns_thresholds:
        a = acc(agg_ns[t][0] / agg_ns[t][1])
        print(f"  ns <= {t:<4}       : {a:5.1f}%  ({a - b_acc:+.1f})")

    if missing:
        print("\nMissing results:")
        for cid, which in missing:
            print(f"  {cid}  ({which})")


if __name__ == "__main__":
    main()
