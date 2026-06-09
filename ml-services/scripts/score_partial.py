"""Score whatever is in results_vad_sensitive_mediumen so far vs small baseline."""
import os, json, jiwer
from eval_common import normalise

DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"
NEWDIR = "results_vad_sensitive_mediumen"

with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
    manifest = {m["call_id"]: m for m in json.load(f)}
with open(os.path.join(DATA, "probe_set.json"), encoding="utf-8") as f:
    probe = {p["call_id"]: p for p in json.load(f)["calls"]}

def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n

def overall(m, ag, cu):
    aa, na = acc(m["agent_transcript"], ag)
    ac, nc = acc(m["customer_transcript"], cu)
    return (aa*na + ac*nc) / (na+nc), na+nc

print(f"  {'call_id':<32} {'tier':<13} {'small':>6} {'med+svad':>9} {'delta':>6}")
print("  " + "-"*70)

total_w_s = total_w_m = total_n = 0
for root, dirs, files in os.walk(os.path.join(DATA, NEWDIR)):
    for fn in sorted(files):
        if not fn.endswith(".json"): continue
        cid = fn[:-5]
        m = manifest.get(cid)
        if not m: continue
        accent = m["accent"]

        # new result
        with open(os.path.join(root, fn), encoding="utf-8") as f:
            r = json.load(f)
        new_ag = " ".join(x["word"] for x in r["agent"])
        new_cu = " ".join(x["word"] for x in r["customer"])
        new_o, w = overall(m, new_ag, new_cu)

        # small baseline
        bp = os.path.join(DATA, "results_channels", accent, cid + ".json")
        with open(bp, encoding="utf-8") as f:
            b = json.load(f)
        base_o, _ = overall(m, " ".join(x["word"] for x in b["agent"]),
                               " ".join(x["word"] for x in b["customer"]))

        tier = probe.get(cid, {}).get("tier", "?")
        print(f"  {cid:<32} {tier:<13} {base_o:>5.1f}%  {new_o:>7.1f}%  {new_o-base_o:>+5.1f}")
        total_w_s += base_o * w; total_w_m += new_o * w; total_n += w

if total_n:
    s_avg = total_w_s / total_n
    m_avg = total_w_m / total_n
    n_calls = sum(1 for _ in os.walk(os.path.join(DATA, NEWDIR)) for f in _[2] if f.endswith(".json"))
    print("  " + "-"*70)
    print(f"  {'PARTIAL AVERAGE':<32} {'':13} {s_avg:>5.1f}%  {m_avg:>7.1f}%  {m_avg-s_avg:>+5.1f}  ({n_calls} calls so far)")
    print(f"\n  Catastrophic tier avg (these 4): {m_avg:.1f}%")
    print(f"  If remaining 8 calls (banking+ag) hold ~93.5% as expected,")
    print(f"  overall would be approximately: {(m_avg*4 + 93.5*8)/12:.1f}%")
