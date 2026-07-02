"""
API-conducive pipeline: VAD-trim each channel LOCALLY (remove the 80% silence
that makes whisper-1 hallucinate), then send only the dense speech to the API.

This is the fair test: the local pipeline removes silence inside the model via
vad_filter=True; the API has no such knob, so we replicate it as a preprocessing
step using the SAME silero VAD (same min_silence_duration_ms=500 as our local run).

Saves the compact WAVs and the API transcripts so we can inspect what changed.

Usage:
  set OPENAI_API_KEY=sk-...
  python pilot_api_vad.py
"""
import os, json, time, tempfile
import numpy as np
import soundfile as sf
import jiwer
from openai import OpenAI
from faster_whisper.vad import get_speech_timestamps, collect_chunks, VadOptions
from eval_common import normalise

DATA      = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA, "manifest.json")
PROBE_SET = os.path.join(DATA, "probe_set.json")
OUT_DIR   = os.path.join(DATA, "results_api_vad")

INITIAL_PROMPT = (
    "Banking call center transcript. "
    "Speakers discuss account numbers, balances, transfers, loans, credit cards, "
    "PINs, dates, dollar amounts, authentication, and customer service."
)

# same VAD config as the local per-channel pipeline (transcribe_channels.py)
VAD_OPTS = VadOptions(min_silence_duration_ms=500)

with open(MANIFEST, encoding="utf-8") as f:
    manifest = {m["call_id"]: m for m in json.load(f)}
with open(PROBE_SET, encoding="utf-8") as f:
    probe = json.load(f)["calls"]

p = probe[0]
cid, accent = p["call_id"], p["accent"]
m = manifest[cid]
a_wav = os.path.join(DATA, m["agent_wav"])
c_wav = os.path.join(DATA, m["customer_wav"])

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: OPENAI_API_KEY not set.")
client = OpenAI(api_key=api_key)

os.makedirs(os.path.join(OUT_DIR, accent), exist_ok=True)


def vad_trim(wav_path, label):
    """Load 16k mono, VAD-trim silence, return compact float32 + stats."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur_in = len(audio) / sr
    ts = get_speech_timestamps(audio, VAD_OPTS, sampling_rate=sr)
    if not ts:
        print(f"  {label:<10} no speech detected; sending original")
        return audio, sr, dur_in, dur_in
    chunks, _ = collect_chunks(audio, ts, sampling_rate=sr)
    # join speech chunks with 0.15s of silence so words don't mash at seams
    gap = np.zeros(int(0.15 * sr), dtype="float32")
    pieces = []
    for i, ch in enumerate(chunks):
        if i: pieces.append(gap)
        pieces.append(ch)
    compact = np.concatenate(pieces)
    dur_out = len(compact) / sr
    print(f"  {label:<10} {dur_in:5.0f}s -> {dur_out:5.0f}s  "
          f"({(1-dur_out/dur_in)*100:.0f}% silence removed, {len(ts)} speech regions)")
    return compact, sr, dur_in, dur_out


def transcribe_api(audio, sr, label):
    # write compact wav straight to the output dir (no temp -> no cross-drive move)
    keep = os.path.join(OUT_DIR, accent, f"{cid}_{label}_compact.wav")
    sf.write(keep, audio, sr, subtype="PCM_16")
    size_mb = os.path.getsize(keep) / 1e6
    t0 = time.time()
    with open(keep, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en",
            prompt=INITIAL_PROMPT, response_format="text")
    elapsed = time.time() - t0
    text = (resp if isinstance(resp, str) else str(resp)).strip()
    print(f"  {label:<10} {elapsed:5.1f}s  {size_mb:4.1f}MB  |  {len(text.split())} words")
    return text, elapsed


def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n


print(f"Pilot call : {cid}  ({accent}  {m['domain']})\n")
print("Step 1 - local VAD trim:")
a_audio, sr, a_din, a_dout = vad_trim(a_wav, "agent")
c_audio, sr, c_din, c_dout = vad_trim(c_wav, "customer")

print("\nStep 2 - whisper-1 API on trimmed speech:")
agent_text,    t_a = transcribe_api(a_audio, sr, "agent")
customer_text, t_c = transcribe_api(c_audio, sr, "customer")
t_total = t_a + t_c

# save transcripts
with open(os.path.join(OUT_DIR, accent, cid + ".json"), "w", encoding="utf-8") as f:
    json.dump({"call_id": cid, "accent": accent, "domain": m["domain"],
               "model": "whisper-1 + local VAD trim",
               "agent_text": agent_text, "customer_text": customer_text}, f, indent=2)

# ── comparison ────────────────────────────────────────────────────────────────
p1_path = os.path.join(DATA, "results_channels", accent, cid + ".json")
with open(p1_path, encoding="utf-8") as f:
    p1 = json.load(f)
small_a = " ".join(w["word"] for w in p1["agent"])
small_c = " ".join(w["word"] for w in p1["customer"])

aa_s, na = acc(m["agent_transcript"],    small_a)
ac_s, nc = acc(m["customer_transcript"], small_c)
aa_v, _  = acc(m["agent_transcript"],    agent_text)
ac_v, _  = acc(m["customer_transcript"], customer_text)
small_overall = (aa_s*na + ac_s*nc) / (na+nc)
vad_overall   = (aa_v*na + ac_v*nc) / (na+nc)

dur = a_din  # ~equal channels
print(f"\n  API wall time: {t_total:.1f}s  | projected 12-probe: {12*t_total/60:.0f} min")
print(f"\n{'='*66}")
print(f"  Head-to-head on {cid}")
print(f"  {'Model':<28} {'Agent':>7} {'Customer':>9} {'Overall':>8}")
print(f"  {'-'*60}")
print(f"  {'small.en (local)':<28} {aa_s:>6.1f}%  {ac_s:>8.1f}%  {small_overall:>7.1f}%")
print(f"  {'medium.en (local)':<28} {90.7:>6.1f}%  {87.5:>8.1f}%  {89.6:>7.1f}%")
print(f"  {'whisper-1 API (naive)':<28} {88.6:>6.1f}%  {71.4:>8.1f}%  {82.5:>7.1f}%")
print(f"  {'whisper-1 API (VAD-trim)':<28} {aa_v:>6.1f}%  {ac_v:>8.1f}%  {vad_overall:>7.1f}%")
print(f"{'='*66}")
print(f"\n  Customer channel: naive 71.4% -> VAD-trim {ac_v:.1f}%  ({ac_v-71.4:+.1f})")
