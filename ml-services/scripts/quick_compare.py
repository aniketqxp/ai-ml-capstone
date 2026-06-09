"""Quick partial comparison: sensitive vs novad_strict on available calls."""
import os, json, jiwer
from eval_common import normalise

DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"
with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
    manifest = {m["call_id"]: m for m in json.load(f)}

def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n

def overall(m, ag, cu):
    aa, na = acc(m["agent_transcript"], ag)
    ac, nc = acc(m["customer_transcript"], cu)
    return (aa*na + ac*nc) / (na+nc), na+nc

def words(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    return " ".join(x["word"] for x in r["agent"]), " ".join(x["word"] for x in r["customer"])

configs = [
    ("baseline",    "results_channels"),
    ("sensitive",   "results_vad_sensitive"),
    ("novad_str",   "results_vad_novad_strict"),
]

calls = [
    "en_US_General_Health_1587175",
    "en_CA_Aviation_1588678",
    "en_CA_Banking_1588683",
    "en_US_General_Banking_1584540",
    "en_CA_Banking_1592237",
]

hdr = f"  {'call_id':<32}"
for name, _ in configs: hdr += f"  {name:>10}"
print(hdr)
print("  " + "-" * 70)

for cid in calls:
    m = manifest[cid]; accent = m["accent"]
    row = f"  {cid:<32}"
    base_o = None
    for name, d in configs:
        p = os.path.join(DATA, d, accent, cid + ".json")
        if not os.path.exists(p):
            row += f"  {'--':>10}"; continue
        if name == "baseline":
            with open(p, encoding="utf-8") as f: j = json.load(f)
            ag = " ".join(x["word"] for x in j["agent"])
            cu = " ".join(x["word"] for x in j["customer"])
        else:
            ag, cu = words(p)
        o, _ = overall(m, ag, cu)
        if name == "baseline":
            base_o = o; row += f"  {o:>9.1f}%"
        else:
            delta = f"({o-base_o:+.1f})" if base_o else ""
            row += f"  {o:>6.1f}%{delta:>4}"
    print(row)
