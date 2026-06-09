"""Quick eval: WER before vs after LLM cleanup pass."""
import os, json, jiwer
from eval_common import normalise

DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"

with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
    manifest = {m["call_id"]: m for m in json.load(f)}
with open(os.path.join(DATA, "probe_set.json"), encoding="utf-8") as f:
    probe = json.load(f)["calls"]


def wer_acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n


print(f"  {'call_id':<32} {'tier':<13} {'before':>7} {'after':>7} {'delta':>7} {'words_chg':>10}")
print("  " + "-" * 80)

agg_b_w = agg_a_w = agg_n = 0.0

for p in probe:
    cid, accent = p["call_id"], p["accent"]
    m = manifest[cid]
    llm_path = os.path.join(DATA, "results_channels_llm", accent, cid + ".json")
    p1_path  = os.path.join(DATA, "results_channels",     accent, cid + ".json")
    if not os.path.exists(llm_path):
        print(f"  {cid:<32} MISSING")
        continue

    with open(llm_path, encoding="utf-8") as f:
        llm = json.load(f)
    with open(p1_path, encoding="utf-8") as f:
        p1 = json.load(f)

    before_a = " ".join(w["word"] for w in p1["agent"])
    before_c = " ".join(w["word"] for w in p1["customer"])
    after_a  = llm["agent_text"]
    after_c  = llm["customer_text"]

    ba, na = wer_acc(m["agent_transcript"], before_a)
    bc, nc = wer_acc(m["customer_transcript"], before_c)
    aa, _  = wer_acc(m["agent_transcript"],   after_a)
    ac, _  = wer_acc(m["customer_transcript"], after_c)

    before = (ba * na + bc * nc) / (na + nc)
    after  = (aa * na + ac * nc) / (na + nc)
    agg_b_w += before * (na + nc)
    agg_a_w += after  * (na + nc)
    agg_n   += na + nc

    # rough word-level change count
    n_chg = sum(1 for a, b in zip(after_a.split(), before_a.split()) if a != b) + \
            sum(1 for a, b in zip(after_c.split(), before_c.split()) if a != b)

    print(f"  {cid:<32} {p['tier']:<13} {before:>6.1f}%  {after:>6.1f}%  {after-before:>+6.1f}  {n_chg:>10}")

overall_b = agg_b_w / agg_n
overall_a = agg_a_w / agg_n
print("  " + "-" * 80)
print(f"  {'OVERALL':<32} {'':13} {overall_b:>6.1f}%  {overall_a:>6.1f}%  {overall_a-overall_b:>+6.1f}")
print()
if overall_a > overall_b + 0.05:
    verdict = "IMPROVED"
elif abs(overall_a - overall_b) < 0.05:
    verdict = "NEUTRAL — within noise"
else:
    verdict = "REGRESSED — LLM is over-correcting"
print(f"  Verdict: {verdict}")
