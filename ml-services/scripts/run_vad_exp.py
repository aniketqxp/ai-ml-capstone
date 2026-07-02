"""
VAD-recall experiment: does a more sensitive VAD (or no VAD) recover the dropped
speech segments we saw in error analysis, WITHOUT adding hallucination?

Runs a named decode config on a small mini-set with small.en (fast), writes to
results_vad_<config>/, and prints per-call accuracy vs the frozen small.en baseline
(results_channels). Includes a good-banking call as a REGRESSION GUARD -- a config
that helps hard calls but hurts easy ones is no good.

Usage:
  python run_vad_exp.py --config sensitive
  python run_vad_exp.py --config novad
  python run_vad_exp.py --config midvad --full   # all 12 probe calls
"""
import os, json, time, argparse, jiwer
from faster_whisper import WhisperModel
import transcribe_channels
from eval_common import normalise

DATA      = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA, "manifest.json")
PROBE_SET = os.path.join(DATA, "probe_set.json")
BASELINE  = "results_channels"   # small.en Phase-1 baseline

# decode configs (merged onto the Phase-1 defaults inside transcribe_channels)
CONFIGS = {
    "baseline":  {},  # sanity: should reproduce results_channels
    "sensitive": {"vad_parameters": {"threshold": 0.3, "min_silence_duration_ms": 200,
                                     "speech_pad_ms": 600}},
    "midvad":    {"vad_parameters": {"threshold": 0.4, "min_silence_duration_ms": 300,
                                     "speech_pad_ms": 500}},
    "novad":     {"vad_filter": False},
    # no-VAD but lean on Whisper's own silence suppression to limit hallucination
    "novad_strict": {"vad_filter": False, "no_speech_threshold": 0.5,
                     "condition_on_previous_text": False},
}

# mini-set: 2 filler-heavy catastrophic + 2 bad-banking + 1 good-banking (guard)
MINISET = [
    "en_US_General_Health_1587175",   # catastrophic, lots of dropped filler/segments
    "en_CA_Aviation_1588678",         # catastrophic, API regressed here
    "en_CA_Banking_1588683",          # bad_banking
    "en_US_General_Banking_1584540",  # bad_banking, big API gain
    "en_CA_Banking_1592237",          # good_banking -- REGRESSION GUARD (94.8%)
]


def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n


def overall(m, ag, cu):
    aa, na = acc(m["agent_transcript"], ag)
    ac, nc = acc(m["customer_transcript"], cu)
    return (aa*na + ac*nc) / (na+nc), na+nc


def baseline_acc(m, accent, cid):
    with open(os.path.join(DATA, BASELINE, accent, cid + ".json"), encoding="utf-8") as f:
        b = json.load(f)
    return overall(m, " ".join(w["word"] for w in b["agent"]),
                      " ".join(w["word"] for w in b["customer"]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--full", action="store_true", help="all 12 probe calls, not mini-set")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    if args.full:
        with open(PROBE_SET, encoding="utf-8") as f:
            ids = [c["call_id"] for c in json.load(f)["calls"]]
    else:
        ids = MINISET

    decode = CONFIGS[args.config]
    model_tag = args.model.replace(".", "")
    # keep small.en config-only dirs (backward compat with the first mini-set runs);
    # tag non-small models so medium/large results never collide with small ones.
    out_dir = (f"results_vad_{args.config}" if args.model == "small.en"
               else f"results_vad_{args.config}_{model_tag}")
    print(f"VAD experiment | config={args.config} | model={args.model} | {len(ids)} calls")
    print(f"  decode overrides: {decode}\n")

    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"  {'call_id':<32} {'base':>6} {'new':>6} {'delta':>6} {'A_wd':>5} {'C_wd':>5}")
    print("  " + "-" * 68)

    deltas, ws, news = [], [], []
    t_start = time.time()
    for cid in ids:
        m = manifest[cid]
        accent = m["accent"]
        a_wav = os.path.join(DATA, m["agent_wav"])
        c_wav = os.path.join(DATA, m["customer_wav"])
        out = os.path.join(DATA, out_dir, accent, cid + ".json")
        os.makedirs(os.path.dirname(out), exist_ok=True)

        if os.path.exists(out) and os.path.getsize(out) > 0:
            with open(out, encoding="utf-8") as f:
                r = json.load(f)
            aw, cw = r["agent"], r["customer"]
        else:
            aw, cw, segs, dur = transcribe_channels.transcribe_call(
                model, a_wav, c_wav, decode=decode)
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"call_id": cid, "accent": accent, "domain": m["domain"],
                           "config": args.config, "agent": aw, "customer": cw}, f, indent=2)

        new_o, w = overall(m, " ".join(x["word"] for x in aw),
                              " ".join(x["word"] for x in cw))
        base_o = baseline_acc(m, accent, cid)
        d = new_o - base_o
        deltas.append(d * w); ws.append(w); news.append(new_o * w)
        print(f"  {cid:<32} {base_o:>5.1f}% {new_o:>5.1f}% {d:>+5.1f} {len(aw):>5} {len(cw):>5}")

    print("  " + "-" * 68)
    wavg     = sum(deltas) / sum(ws)
    overall_abs = sum(news) / sum(ws)
    print(f"  OVERALL absolute accuracy: {overall_abs:.2f}%   "
          f"(word-weighted mean delta vs small.en baseline: {wavg:+.2f})")
    print(f"  {'>> HELPS' if wavg > 0.1 else '>> NEUTRAL' if abs(wavg) <= 0.1 else '>> HURTS'}")
    if args.full:
        tag = ("CLEARS +cushion" if overall_abs >= 93.5 else
               "clears (thin)" if overall_abs >= 93.0 else "short of 93%")
        print(f"  GOAL (>=93% over 12 calls): {overall_abs:.2f}%  ->  {tag}")
    print(f"  ({(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
