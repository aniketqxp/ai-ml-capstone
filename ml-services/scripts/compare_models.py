"""
Standalone model comparison over the 12-call probe. NO API calls, NO transcription
-- pure evaluation of whatever result dirs exist on disk. Run anytime.

Reads (when present):
  results_channels         small.en  (words: agent/customer lists)
  results_channels_medium  medium.en (words)
  results_channels_large   large-v3  (words)
  results_api_vad          whisper-1 API + VAD trim (agent_text/customer_text)
  results_channels_llm     small.en + LLM cleanup   (agent_text/customer_text)

Prints: per-call table, OVERALL, per-tier averages, and the banking-only cut
(banking is the project's target domain). Flags whether each model clears the
93% goal with cushion.
"""
import os, json, jiwer
from collections import defaultdict
from eval_common import normalise

DATA      = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA, "manifest.json")
PROBE_SET = os.path.join(DATA, "probe_set.json")

MODELS = [
    ("small",  "results_channels",        "words"),
    ("medium", "results_channels_medium", "words"),
    ("large",  "results_channels_large",  "words"),
    ("API",    "results_api_vad",         "text"),
    ("LLM",    "results_channels_llm",    "text"),
]


def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n


def load_call(model_dir, fmt, accent, cid):
    """Return (agent_str, customer_str) or None if not present."""
    path = os.path.join(DATA, model_dir, accent, cid + ".json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if fmt == "words":
        return (" ".join(w["word"] for w in d["agent"]),
                " ".join(w["word"] for w in d["customer"]))
    return d["agent_text"], d["customer_text"]


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    with open(PROBE_SET, encoding="utf-8") as f:
        probe = json.load(f)["calls"]

    # which models actually have data
    present = []
    for name, d, fmt in MODELS:
        if os.path.isdir(os.path.join(DATA, d)):
            present.append((name, d, fmt))

    # per-call overall accuracy per model
    rows = []                     # (cid, tier, domain, {model: (acc, words, n_done)})
    agg  = {n: [0.0, 0, 0] for n, _, _ in present}   # [acc*words, words, n_calls]
    tier_agg = defaultdict(lambda: {n: [0.0, 0] for n, _, _ in present})
    bank_agg = {n: [0.0, 0] for n, _, _ in present}

    for p in probe:
        cid, accent, tier = p["call_id"], p["accent"], p["tier"]
        m = manifest[cid]
        domain = m["domain"]
        cell = {}
        for name, d, fmt in present:
            loaded = load_call(d, fmt, accent, cid)
            if loaded is None:
                cell[name] = None
                continue
            ag, cu = loaded
            aa, na = acc(m["agent_transcript"], ag)
            ac, nc = acc(m["customer_transcript"], cu)
            o = (aa*na + ac*nc) / (na+nc)
            cell[name] = (o, na+nc)
            agg[name][0] += o*(na+nc); agg[name][1] += na+nc; agg[name][2] += 1
            tier_agg[tier][name][0] += o; tier_agg[tier][name][1] += 1
            if domain.lower() == "banking":
                bank_agg[name][0] += o; bank_agg[name][1] += 1
        rows.append((cid, tier, domain, cell))

    names = [n for n, _, _ in present]
    hdr = "  {:<32} {:<13}".format("call_id", "tier") + "".join(f"{n:>8}" for n in names)
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    for cid, tier, domain, cell in rows:
        line = "  {:<32} {:<13}".format(cid, tier)
        for n in names:
            v = cell[n]
            line += f"{v[0]:>7.1f}%" if v else f"{'--':>8}"
        print(line)
    print("  " + "-" * (len(hdr)-2))

    # overall
    line = "  {:<32} {:<13}".format("OVERALL", "")
    overall = {}
    for n in names:
        if agg[n][1]:
            o = agg[n][0]/agg[n][1]; overall[n] = (o, agg[n][2])
            line += f"{o:>7.1f}%"
        else:
            line += f"{'--':>8}"
    print(line)
    # n calls done per model
    line = "  {:<32} {:<13}".format("(calls scored)", "")
    for n in names:
        line += f"{agg[n][2]:>7}/12" if agg[n][2] else f"{'--':>8}"
    print(line.replace("/12  ", "/12"))

    # per-tier
    print("\n  Per-tier average accuracy:")
    tiers = ["catastrophic", "bad_banking", "mid_banking", "good_banking", "good_other"]
    print("  {:<14}".format("tier") + "".join(f"{n:>8}" for n in names))
    for t in tiers:
        if t not in tier_agg: continue
        line = "  {:<14}".format(t)
        for n in names:
            s, c = tier_agg[t][n]
            line += f"{s/c:>7.1f}%" if c else f"{'--':>8}"
        print(line)

    # banking cut
    print("\n  BANKING-ONLY (target domain):")
    line = "  {:<14}".format("banking avg")
    for n in names:
        s, c = bank_agg[n]
        line += f"{s/c:>7.1f}%" if c else f"{'--':>8}"
    print(line)

    # goal check
    print("\n  Goal = >=93.0% overall with cushion:")
    for n in names:
        if n in overall and overall[n][1] == 12:
            o = overall[n][0]
            tag = "CLEARS +cushion" if o >= 93.5 else "clears (thin)" if o >= 93.0 else "short"
            print(f"    {n:<8} {o:5.1f}%  ({12} calls)  -> {tag}")
        elif n in overall:
            print(f"    {n:<8} {overall[n][0]:5.1f}%  ({overall[n][1]}/12 calls -- partial)")


if __name__ == "__main__":
    main()
