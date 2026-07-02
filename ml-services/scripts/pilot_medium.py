"""
Single-call pilot for medium.en — times model load + one transcription.
Picks the first probe call (Health catastrophic) as the stress test.
"""
import os, json, time
from faster_whisper import WhisperModel
import transcribe_channels

DATA      = r"d:\Desktop\ai-ml-capstone\data\na_testset"
PROBE_SET = os.path.join(DATA, "probe_set.json")
MANIFEST  = os.path.join(DATA, "manifest.json")

with open(MANIFEST, encoding="utf-8") as f:
    manifest = {m["call_id"]: m for m in json.load(f)}
with open(PROBE_SET, encoding="utf-8") as f:
    probe = json.load(f)["calls"]

# pick the first call (Health catastrophic — longest/hardest)
p = probe[0]
cid, accent = p["call_id"], p["accent"]
m = manifest[cid]
a_wav = os.path.join(DATA, m["agent_wav"])
c_wav = os.path.join(DATA, m["customer_wav"])

print(f"Pilot call: {cid}  ({accent}  {m['domain']})")
print("Loading faster-whisper medium.en/int8 (will download if not cached)...")
t0 = time.time()
model = WhisperModel("medium.en", device="cpu", compute_type="int8")
load_s = time.time() - t0
print(f"  Model loaded in {load_s:.1f}s")

print("Transcribing...")
t1 = time.time()
aw, cw, segs, dur = transcribe_channels.transcribe_call(model, a_wav, c_wav)
trans_s = time.time() - t1
print(f"  Audio duration : {dur:.0f}s  ({dur/60:.1f} min)")
print(f"  Transcribe time: {trans_s:.1f}s  ({trans_s/60:.1f} min)")
print(f"  Real-time factor: {trans_s/dur:.2f}x")
print(f"  Projected time for 12-call probe: {12 * trans_s / 60:.0f} min  (~{12 * trans_s / 3600:.1f}h)")

# quick WER check vs small.en baseline
import jiwer
from eval_common import normalise

p1_path = os.path.join(DATA, "results_channels", accent, cid + ".json")
with open(p1_path, encoding="utf-8") as f:
    p1 = json.load(f)

small_a = " ".join(w["word"] for w in p1["agent"])
small_c = " ".join(w["word"] for w in p1["customer"])
med_a   = " ".join(w["word"] for w in aw)
med_c   = " ".join(w["word"] for w in cw)

def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    return (1 - jiwer.wer(r, h)) * 100

sa = acc(m["agent_transcript"],    small_a)
sc = acc(m["customer_transcript"], small_c)
ma = acc(m["agent_transcript"],    med_a)
mc = acc(m["customer_transcript"], med_c)

na = len(normalise(m["agent_transcript"]).split())
nc = len(normalise(m["customer_transcript"]).split())

small_overall = (sa*na + sc*nc) / (na+nc)
med_overall   = (ma*na + mc*nc) / (na+nc)

print(f"\n  small.en accuracy: {small_overall:.1f}%  (A:{sa:.1f}% C:{sc:.1f}%)")
print(f"  medium.en accuracy:{med_overall:.1f}%  (A:{ma:.1f}% C:{mc:.1f}%)")
print(f"  Delta: {med_overall - small_overall:+.1f}%")
