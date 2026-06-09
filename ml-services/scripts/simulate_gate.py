"""
Find the optimal loudness-gate threshold WITHOUT any new transcription run.

Key idea: the gated pipeline's output for each channel is already known:
  - channel active-RMS >= gate  -> identical to Phase 1  (results_channels/, no PP)
  - channel active-RMS <  gate  -> identical to Phase 2  (results_channels_pp/, full PP)
(high-pass is near-neutral on this data, so "above gate" ~ Phase 1 is exact enough.)

So we assemble each call per-channel from the two existing result sets according
to a candidate gate, score it, and sweep the gate to find the threshold that
keeps the quiet-channel gains while removing the over-amplification regressions.

This is pure analysis over cached results -> runs in seconds, no model, no CPU run.

Usage:
  python simulate_gate.py
  python simulate_gate.py --gate -26       # per-call detail at one gate
"""

import os
import json
import argparse
import numpy as np
import soundfile as sf
import jiwer
from eval_common import normalise
from audio_preprocess import highpass, active_rms

DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"
P1   = "results_channels"      # no preprocessing  (Phase 1)
P2   = "results_channels_pp"   # full preprocessing (Phase 2, target -20 dBFS)
SR   = 16000


def load_result(root, accent, cid):
    with open(os.path.join(DATA, root, accent, cid + ".json"), encoding="utf-8") as f:
        return json.load(f)


def channel_rms_db(wav_rel):
    a, sr = sf.read(os.path.join(DATA, wav_rel), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = highpass(a, sr)                       # match preprocess(): measure post-highpass
    return 20 * np.log10(active_rms(a, sr) + 1e-12)


def wer(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    if n == 0:
        return 0.0, 0
    return jiwer.wer(r, h), n


def text(words):
    return " ".join(w["word"] for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=None, help="show per-call detail at this gate")
    ap.add_argument("--sweep", nargs="*", type=float,
                    default=[-99, -34, -32, -30, -28, -26, -24, -22, -20, 0])
    args = ap.parse_args()

    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    with open(os.path.join(DATA, "probe_set.json"), encoding="utf-8") as f:
        probe = json.load(f)["calls"]

    # Pre-load everything once: refs, both result sets, per-channel RMS.
    calls = []
    for p in probe:
        cid, accent = p["call_id"], p["accent"]
        m = manifest[cid]
        calls.append({
            "cid": cid, "tier": p["tier"],
            "ref_a": m["agent_transcript"], "ref_c": m["customer_transcript"],
            "rms_a": channel_rms_db(m["agent_wav"]),
            "rms_c": channel_rms_db(m["customer_wav"]),
            "p1_a": text(load_result(P1, accent, cid)["agent"]),
            "p1_c": text(load_result(P1, accent, cid)["customer"]),
            "p2_a": text(load_result(P2, accent, cid)["agent"]),
            "p2_c": text(load_result(P2, accent, cid)["customer"]),
        })

    def score_at(gate):
        """Word-weighted accuracy across probe at this gate, + per-call accs + #boosted."""
        num = den = 0
        per_call = {}
        boosted = 0
        for c in calls:
            ha = c["p2_a"] if c["rms_a"] < gate else c["p1_a"]
            hc = c["p2_c"] if c["rms_c"] < gate else c["p1_c"]
            boosted += (c["rms_a"] < gate) + (c["rms_c"] < gate)
            wa, na = wer(c["ref_a"], ha)
            wc, nc = wer(c["ref_c"], hc)
            cw = (wa * na + wc * nc) / (na + nc)
            per_call[c["cid"]] = (1 - cw) * 100
            num += wa * na + wc * nc
            den += na + nc
        return (1 - num / den) * 100, per_call, boosted

    # ── Gate sweep ────────────────────────────────────────────────────────────
    print("Gate sweep (accuracy assembled from existing P1/P2 results, no re-run):\n")
    print(f"  {'gate':>6}  {'boosted':>7}  {'overall':>8}   {'Bank_1586157':>12}  {'Health_1587175':>14}")
    print("  " + "-" * 56)
    reg_cid, quiet_cid = "en_US_General_Banking_1586157", "en_US_General_Health_1587175"
    for g in args.sweep:
        acc, per_call, boosted = score_at(g)
        label = {-99: "none", 0: "all"}.get(g, f"{g:.0f}")
        print(f"  {label:>6}  {boosted:>7}  {acc:>7.1f}%   {per_call[reg_cid]:>11.1f}%  {per_call[quiet_cid]:>13.1f}%")

    print("\n  ('none' = pure Phase 1 / no PP;  'all' = pure Phase 2 / full PP)")
    print("  Bank_1586157 = the regression call;  Health_1587175 = a quiet-channel call")

    # ── Per-call detail at one gate ───────────────────────────────────────────
    if args.gate is not None:
        acc, per_call, boosted = score_at(args.gate)
        print(f"\nPer-call at gate = {args.gate} dBFS  (overall {acc:.1f}%, {boosted} channels boosted):\n")
        print(f"  {'tier':<13} {'call_id':<32} {'rmsA':>6} {'rmsC':>6}  {'boost':>9}  {'acc':>6}")
        print("  " + "-" * 74)
        for c in calls:
            ba = "A" if c["rms_a"] < args.gate else "-"
            bc = "C" if c["rms_c"] < args.gate else "-"
            print(f"  {c['tier']:<13} {c['cid']:<32} {c['rms_a']:>6.1f} {c['rms_c']:>6.1f}  "
                  f"{ba+bc:>9}  {per_call[c['cid']]:>5.1f}%")


if __name__ == "__main__":
    main()
