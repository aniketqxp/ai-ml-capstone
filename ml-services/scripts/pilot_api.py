"""
Single-call API pilot for whisper-1 (OpenAI) — same call as pilot_medium.py
so we get a direct 3-way comparison: small.en / medium.en / whisper-1 API.

Uses per-channel approach: agent WAV + customer WAV sent separately,
same structural speaker attribution as the local pipeline.

Usage:
  set OPENAI_API_KEY=sk-...
  python pilot_api.py
"""
import os, json, time
import jiwer
from openai import OpenAI
from eval_common import normalise

DATA      = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA, "manifest.json")
PROBE_SET = os.path.join(DATA, "probe_set.json")

# same initial prompt as local pipeline for fair comparison
INITIAL_PROMPT = (
    "Banking call center transcript. "
    "Speakers discuss account numbers, balances, transfers, loans, credit cards, "
    "PINs, dates, dollar amounts, authentication, and customer service."
)

with open(MANIFEST, encoding="utf-8") as f:
    manifest = {m["call_id"]: m for m in json.load(f)}
with open(PROBE_SET, encoding="utf-8") as f:
    probe = json.load(f)["calls"]

# same call as medium.en pilot
p = probe[0]
cid, accent = p["call_id"], p["accent"]
m = manifest[cid]
a_wav = os.path.join(DATA, m["agent_wav"])
c_wav = os.path.join(DATA, m["customer_wav"])

print(f"Pilot call : {cid}  ({accent}  {m['domain']})")
print(f"Agent WAV  : {os.path.getsize(a_wav)/1e6:.1f} MB")
print(f"Customer WAV: {os.path.getsize(c_wav)/1e6:.1f} MB")
print()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Run: set OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=api_key)


def transcribe_via_api(wav_path, label):
    t0 = time.time()
    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            prompt=INITIAL_PROMPT,
            response_format="text",
        )
    elapsed = time.time() - t0
    text = resp.strip() if isinstance(resp, str) else resp
    words = text.split()
    print(f"  {label:<10} {elapsed:5.1f}s  |  {len(words)} words")
    return text, elapsed


print("Calling whisper-1 API (per channel)...")
agent_text,    t_agent    = transcribe_via_api(a_wav, "agent")
customer_text, t_customer = transcribe_via_api(c_wav, "customer")
t_total = t_agent + t_customer

# audio duration from manifest or estimate from file size (16-bit 8kHz mono)
# WAV file = header(44) + samples. 20.1MB @ 16-bit 8kHz = ~1256s...
# Actually let's compute from the WAV header
import wave
with wave.open(a_wav) as wf:
    dur = wf.getnframes() / wf.getframerate()

print(f"\n  Audio duration : {dur:.0f}s  ({dur/60:.1f} min)")
print(f"  API wall time  : {t_total:.1f}s  ({t_total/60:.1f} min)")
print(f"  Real-time factor: {t_total/dur:.2f}x")
print(f"  Projected for 12-call probe: {12*t_total/60:.0f} min")

# ── WER comparison ────────────────────────────────────────────────────────────
def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n

# load small.en and medium.en results for the same call
p1_path  = os.path.join(DATA, "results_channels",        accent, cid + ".json")
med_path = os.path.join(DATA, "results_channels_medium", accent, cid + ".json")

with open(p1_path, encoding="utf-8") as f:
    p1 = json.load(f)

small_a = " ".join(w["word"] for w in p1["agent"])
small_c = " ".join(w["word"] for w in p1["customer"])

aa_s, na = acc(m["agent_transcript"],    small_a)
ac_s, nc = acc(m["customer_transcript"], small_c)
aa_api,_ = acc(m["agent_transcript"],    agent_text)
ac_api,_ = acc(m["customer_transcript"], customer_text)

small_overall = (aa_s*na + ac_s*nc) / (na+nc)
api_overall   = (aa_api*na + ac_api*nc) / (na+nc)

# medium.en — load from disk if saved, else use known pilot numbers
aa_m = ac_m = med_overall = None
if os.path.exists(med_path):
    with open(med_path, encoding="utf-8") as f:
        med = json.load(f)
    med_a = " ".join(w["word"] for w in med["agent"])
    med_c = " ".join(w["word"] for w in med["customer"])
    aa_m, _ = acc(m["agent_transcript"],    med_a)
    ac_m, _ = acc(m["customer_transcript"], med_c)
    med_overall = (aa_m*na + ac_m*nc) / (na+nc)
else:
    # known numbers from pilot_medium.py run (same call, same model)
    aa_m, ac_m, med_overall = 90.7, 87.5, 89.6

print(f"\n{'='*60}")
print(f"  Head-to-head on {cid}")
print(f"  {'Model':<22} {'Agent':>7} {'Customer':>9} {'Overall':>8} {'Time':>8}")
print(f"  {'-'*56}")
print(f"  {'small.en (local)':<22} {aa_s:>6.1f}%  {ac_s:>8.1f}%  {small_overall:>7.1f}%  {'~3.5min':>8}")
print(f"  {'medium.en (local)':<22} {aa_m:>6.1f}%  {ac_m:>8.1f}%  {med_overall:>7.1f}%  {'~10min':>8}")
print(f"  {'whisper-1 (API)':<22} {aa_api:>6.1f}%  {ac_api:>8.1f}%  {api_overall:>7.1f}%  {t_total/60:>7.1f}min")
print(f"{'='*60}")
