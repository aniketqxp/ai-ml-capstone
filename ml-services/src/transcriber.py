import os
import json
import time
import requests
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

os.environ["HF_HUB_OFFLINE"] = "1"
import whisperx

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

TARGET_SR = 16000


def load_audio_sf(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        n = int(len(data) * TARGET_SR / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return data


def get_api_segments(api_key, file_path, label):
    print(f"[{label}] Transcribing via OpenAI API...")
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as f:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files={"file": (os.path.basename(file_path), f, "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json"},
        )
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API failed ({response.status_code}): {response.text}")
    segments = [
        {"text": s["text"].strip(), "start": s["start"], "end": s["end"]}
        for s in response.json().get("segments", [])
    ]
    print(f"[{label}] Obtained {len(segments)} segments from API")
    return segments


def align_locally(segments, audio_path, label, align_model, metadata):
    print(f"[{label}] Loading audio...")
    audio = load_audio_sf(audio_path)
    print(f"[{label}] Aligning with wav2vec2 (forced-alignment)...")
    t0 = time.time()
    result = whisperx.align(
        segments, align_model, metadata, audio, device="cpu", return_char_alignments=False
    )
    print(f"[{label}] Aligned in {time.time() - t0:.2f}s")
    words = [
        {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for seg in result.get("segments", [])
        for w in seg.get("words", [])
        if "start" in w and "end" in w
    ]
    print(f"[{label}] Extracted {len(words)} aligned words")
    return words


def run(agent_audio, customer_audio, output_path, call_id="call_1"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    agent_segs = get_api_segments(api_key, agent_audio, "AGENT")
    customer_segs = get_api_segments(api_key, customer_audio, "CUSTOMER")

    print("\nLoading WhisperX wav2vec2 alignment model from cache...")
    align_model, metadata = whisperx.load_align_model(language_code="en", device="cpu")

    agent_words = align_locally(agent_segs, agent_audio, "AGENT", align_model, metadata)
    customer_words = align_locally(customer_segs, customer_audio, "CUSTOMER", align_model, metadata)

    out = {
        "model": "Hybrid (OpenAI API + Local WhisperX Forced-Alignment)",
        "call": call_id,
        "agent": agent_words,
        "customer": customer_words,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Agent: {len(agent_words)} words | Customer: {len(customer_words)} words")
    return out


if __name__ == "__main__":
    APPTEK = r"d:\Desktop\ai-ml-capstone\data\apptek"
    OUT = r"d:\Desktop\ai-ml-capstone\frontend\transcript_data.json"
    run(
        agent_audio=os.path.join(APPTEK, "call_1_agent.wav"),
        customer_audio=os.path.join(APPTEK, "call_1_customer.wav"),
        output_path=OUT,
    )
