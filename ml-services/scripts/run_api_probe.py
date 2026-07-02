"""
API-conducive pipeline over the full 12-call probe set.

Per channel: local silero VAD trim (remove silence) -> whisper-1 API on dense
speech. Same VAD config (min_silence_duration_ms=500) as the local pipeline, so
the only variable is the acoustic model (small.en/int8 local vs whisper-1 API).

Resumable: skips calls already written to results_api_vad/.
Prints a per-call + overall comparison vs small.en (and medium.en where present).

Usage:
  set OPENAI_API_KEY=sk-...
  python run_api_probe.py
"""
import os, json, time
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
SMALL_DIR = os.path.join(DATA, "results_channels")
MED_DIR   = os.path.join(DATA, "results_channels_medium")

INITIAL_PROMPT = (
    "Banking call center transcript. "
    "Speakers discuss account numbers, balances, transfers, loans, credit cards, "
    "PINs, dates, dollar amounts, authentication, and customer service."
)
VAD_OPTS = VadOptions(min_silence_duration_ms=500)


def vad_trim(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ts = get_speech_timestamps(audio, VAD_OPTS, sampling_rate=sr)
    if not ts:
        return audio, sr
    chunks, _ = collect_chunks(audio, ts, sampling_rate=sr)
    gap = np.zeros(int(0.15 * sr), dtype="float32")
    pieces = []
    for i, ch in enumerate(chunks):
        if i: pieces.append(gap)
        pieces.append(ch)
    return np.concatenate(pieces), sr


def transcribe_api(client, audio, sr, keep_path):
    sf.write(keep_path, audio, sr, subtype="PCM_16")
    with open(keep_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en",
            prompt=INITIAL_PROMPT, response_format="text")
    return (resp if isinstance(resp, str) else str(resp)).strip()


def acc(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (1 - jiwer.wer(r, h)) * 100 if n else 100.0, n


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    with open(PROBE_SET, encoding="utf-8") as f:
        probe = json.load(f)["calls"]

    print(f"API VAD-trim probe | {len(probe)} calls\n")
    t_start = time.time()

    for i, p in enumerate(probe, 1):
        cid, accent = p["call_id"], p["accent"]
        m = manifest[cid]
        out = os.path.join(OUT_DIR, accent, cid + ".json")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"  [{i:>2}/12] {cid:<32} cached")
            continue
        os.makedirs(os.path.join(OUT_DIR, accent), exist_ok=True)

        t0 = time.time()
        try:
            a_audio, sr = vad_trim(os.path.join(DATA, m["agent_wav"]))
            c_audio, sr = vad_trim(os.path.join(DATA, m["customer_wav"]))
            a_keep = os.path.join(OUT_DIR, accent, f"{cid}_agent_compact.wav")
            c_keep = os.path.join(OUT_DIR, accent, f"{cid}_customer_compact.wav")
            agent_text    = transcribe_api(client, a_audio, sr, a_keep)
            customer_text = transcribe_api(client, c_audio, sr, c_keep)
        except Exception as e:
            print(f"  [{i:>2}/12] {cid:<32} ERROR: {e}")
            continue

        with open(out, "w", encoding="utf-8") as f:
            json.dump({"call_id": cid, "accent": accent, "domain": m["domain"],
                       "model": "whisper-1 + local VAD trim",
                       "agent_text": agent_text, "customer_text": customer_text}, f, indent=2)
        # remove the compact wavs to save space (keep transcripts)
        for w in (a_keep, c_keep):
            try: os.remove(w)
            except OSError: pass
        print(f"  [{i:>2}/12] {p['tier']:<13} {cid:<32} {time.time()-t0:5.1f}s")

    print(f"\nAll API transcripts ready in {(time.time()-t_start)/60:.1f} min.\n")

    # ── comparison table ──────────────────────────────────────────────────────
    print(f"  {'call_id':<32} {'tier':<13} {'small':>6} {'medium':>7} {'API':>6}")
    print("  " + "-" * 74)
    agg = {"small": [0.0, 0], "med": [0.0, 0], "api": [0.0, 0]}

    for p in probe:
        cid, accent = p["call_id"], p["accent"]
        m = manifest[cid]

        def overall_from_channels(agent_words, customer_words):
            aa, na = acc(m["agent_transcript"], agent_words)
            ac, nc = acc(m["customer_transcript"], customer_words)
            return (aa*na + ac*nc) / (na+nc), na + nc

        # small.en
        with open(os.path.join(SMALL_DIR, accent, cid + ".json"), encoding="utf-8") as f:
            s = json.load(f)
        small_o, w = overall_from_channels(
            " ".join(x["word"] for x in s["agent"]),
            " ".join(x["word"] for x in s["customer"]))
        agg["small"][0] += small_o * w; agg["small"][1] += w

        # medium.en (optional)
        med_str = "  --  "
        med_path = os.path.join(MED_DIR, accent, cid + ".json")
        if os.path.exists(med_path):
            with open(med_path, encoding="utf-8") as f:
                md = json.load(f)
            med_o, _ = overall_from_channels(
                " ".join(x["word"] for x in md["agent"]),
                " ".join(x["word"] for x in md["customer"]))
            agg["med"][0] += med_o * w; agg["med"][1] += w
            med_str = f"{med_o:5.1f}%"

        # api
        api_str = "  --  "
        api_path = os.path.join(OUT_DIR, accent, cid + ".json")
        if os.path.exists(api_path):
            with open(api_path, encoding="utf-8") as f:
                ad = json.load(f)
            api_o, _ = overall_from_channels(ad["agent_text"], ad["customer_text"])
            agg["api"][0] += api_o * w; agg["api"][1] += w
            api_str = f"{api_o:5.1f}%"

        print(f"  {cid:<32} {p['tier']:<13} {small_o:5.1f}% {med_str:>7} {api_str:>6}")

    print("  " + "-" * 74)
    so = agg["small"][0]/agg["small"][1]
    mo = agg["med"][0]/agg["med"][1] if agg["med"][1] else None
    ao = agg["api"][0]/agg["api"][1] if agg["api"][1] else None
    mo_str = f"{mo:5.1f}%" if mo else "  --  "
    ao_str = f"{ao:5.1f}%" if ao else "  --  "
    print(f"  {'OVERALL':<32} {'':13} {so:5.1f}% {mo_str:>7} {ao_str:>6}")
    if ao:
        print(f"\n  API vs small.en: {ao-so:+.1f}   "
              f"(medium.en partial: {agg['med'][1]}/{agg['small'][1]} words)")


if __name__ == "__main__":
    main()
