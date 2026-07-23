"""Recover AGENT/CUSTOMER roles for mixed mono call recordings.

The original dataset has physically separated channels. A normal mono upload
does not, so channel identity cannot supply speaker roles. We transcribe it
once, sentence-segment the timestamped words, and ask the existing routed LLM
to label those sentences using dialogue context. The labels are then projected
back onto word timestamps, producing the same agent/customer transcript shape
consumed by the rest of the pipeline.
"""
import json
import hashlib
import threading
from pathlib import Path

import paths

_CACHE_LOCK = threading.Lock()


SYSTEM = """You label speakers in a customer-service call transcript.
Return AGENT for the company representative and CUSTOMER for the caller.
Use context, questions/answers, company procedures, greetings, and requests.
Do not rewrite text. Return one label for every supplied seq_id as JSON."""


def _heuristic(text, previous):
    value = (text or "").lower()
    agent_markers = (
        "how can i help", "welcome to", "could you please", "can you please",
        "i can see", "i have submitted", "anything else", "thank you for verifying",
        "please confirm", "would you like me", "our system", "your request",
    )
    customer_markers = (
        "i lost", "my card", "i need", "i want", "yes", "no", "thank you",
        "my address", "my number", "what is", "how long", "can i",
    )
    if any(marker in value for marker in agent_markers):
        return "AGENT"
    if any(marker in value for marker in customer_markers):
        return "CUSTOMER"
    return previous or "AGENT"


def label_sentences(sentences, chunk_size=45):
    from router import chat_json_routed

    identity = hashlib.sha256(json.dumps([
        {"start": row.get("start"), "end": row.get("end"), "text": row.get("text")}
        for row in sentences
    ], sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = Path(paths.DATA_ROOT) / "mono_role_cache" / f"{identity}.json"
    with _CACHE_LOCK:
        if cache_path.exists():
            return {int(key): value for key, value in
                    json.loads(cache_path.read_text(encoding="utf-8")).items()}

        labels = {}
        previous = "AGENT"
        for offset in range(0, len(sentences), chunk_size):
            chunk = sentences[offset:offset + chunk_size]
            context = sentences[max(0, offset - 3):offset]
            payload = {
            "previous_context": [
                {"seq_id": row.get("seq_id"), "text": row.get("text", "")}
                for row in context
            ],
            "sentences_to_label": [
                {"seq_id": row.get("seq_id"), "text": row.get("text", "")}
                for row in chunk
            ],
            "required_shape": {
                "labels": [{"seq_id": 1, "speaker": "AGENT"}]
            },
        }
            parsed = {}
            try:
                raw = chat_json_routed(SYSTEM, json.dumps(payload), max_tokens=2500)
                data = json.loads(raw)
                for row in data.get("labels", []):
                    speaker = str(row.get("speaker", "")).upper()
                    if speaker in {"AGENT", "CUSTOMER"}:
                        parsed[int(row["seq_id"])] = speaker
            except Exception as exc:
                print(f"  [mono roles] LLM chunk fallback: {type(exc).__name__}: {exc}")

            for row in chunk:
                seq_id = int(row["seq_id"])
                speaker = parsed.get(seq_id) or _heuristic(row.get("text"), previous)
                labels[seq_id] = speaker
                previous = speaker
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        return labels


def rebuild_word_channels(result, segmentation):
    """Return a transcript with mixed words assigned to inferred roles."""
    sentences = segmentation.get("sentences") or []
    if not sentences:
        return result
    labels = label_sentences(sentences)
    spans = [
        (float(row["start"]), float(row["end"]), labels[int(row["seq_id"])])
        for row in sentences
    ]
    mixed_words = sorted(
        list(result.get("agent") or []) + list(result.get("customer") or []),
        key=lambda row: float(row.get("start", 0)),
    )
    channels = {"AGENT": [], "CUSTOMER": []}
    for word in mixed_words:
        midpoint = (float(word.get("start", 0)) + float(word.get("end", 0))) / 2
        match = min(
            spans,
            key=lambda span: 0 if span[0] <= midpoint <= span[1]
            else min(abs(midpoint - span[0]), abs(midpoint - span[1])),
        )
        channels[match[2]].append(word)
    updated = dict(result)
    updated["agent"] = channels["AGENT"]
    updated["customer"] = channels["CUSTOMER"]
    updated["speaker_attribution"] = "llm_role_reconstruction_from_mixed_mono"
    return updated
