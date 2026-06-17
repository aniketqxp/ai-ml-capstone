"""
Build sentence-level segments from word-level faster-whisper output.

Sentence boundaries are defined by:
  1. Sentence-ending punctuation (. ? !) on a word — excluding known abbreviations
  2. Silence gaps >= GAP_SPLIT_S between consecutive words (speaker turn break)

Fragments shorter than MIN_DURATION_S are merged into the preceding sentence
(only when the gap between them is small) so wav2vec2 always receives a clip
long enough to infer emotion.

Input:  frontend/transcript_data.json  (word-level agent + customer arrays)
Output: frontend/src/sentence_segments.json

Schema per sentence:
  {
    "id":       "AGENT_001",      # <SPEAKER>_<zero-padded index>
    "seq_id":   1,                # global chronological index across both speakers
    "speaker":  "AGENT",
    "start":    13.49,
    "end":      20.43,
    "duration": 6.94,
    "text":     "Oh, good afternoon. ..."
  }
"""
import json
import os
import re

MIN_DURATION_S = 1.5   # merge fragments shorter than this into the previous sentence
GAP_SPLIT_S    = 2.0   # silence gap that always forces a new sentence regardless of punctuation
MAX_MERGE_GAP  = 1.0   # only merge a short fragment if the gap from previous sentence is <= this

TRANSCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "frontend", "transcript_data.json"
)
OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "frontend", "src", "sentence_segments.json"
)

# abbreviations whose trailing period must NOT trigger a sentence split
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "st",
    "ave", "blvd", "dept", "est", "approx", "etc", "e.g", "i.e",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

SENTENCE_END = re.compile(r"[.?!]$")


def is_sentence_end(word: str) -> bool:
    """True if this word ends a sentence (not an abbreviation period)."""
    if not SENTENCE_END.search(word):
        return False
    # strip trailing punctuation to get the base token
    base = re.sub(r"[.?!,;:]+$", "", word).lower()
    if base in ABBREVIATIONS:
        return False
    # single uppercase letter followed by period → initial, not sentence end
    if re.match(r"^[a-z]$", base):
        return False
    return True


def words_to_sentences(words: list[dict], speaker: str) -> list[dict]:
    """
    Split words into sentences using punctuation + gap heuristics,
    then merge short trailing fragments into their predecessor.
    """
    if not words:
        return []

    # ── phase 1: split into raw sentence buckets ─────────────────────────────
    raw_sentences = []
    bucket = [words[0]]

    for prev_w, curr_w in zip(words, words[1:]):
        gap = curr_w["start"] - prev_w["end"]
        if is_sentence_end(prev_w["word"]) or gap >= GAP_SPLIT_S:
            raw_sentences.append(bucket)
            bucket = [curr_w]
        else:
            bucket.append(curr_w)

    if bucket:
        raw_sentences.append(bucket)

    # ── phase 2: merge short fragments into predecessor ───────────────────────
    merged = []
    for sent_words in raw_sentences:
        start = sent_words[0]["start"]
        end   = sent_words[-1]["end"]
        dur   = end - start
        text  = " ".join(w["word"] for w in sent_words)

        # only merge fragments that lack terminal punctuation (trailing word scraps,
        # not real sentences that just happen to be short)
        last_word_has_terminal = SENTENCE_END.search(sent_words[-1]["word"])
        if merged and dur < MIN_DURATION_S and not last_word_has_terminal:
            gap_from_prev = start - merged[-1]["end"]
            if gap_from_prev <= MAX_MERGE_GAP:
                prev = merged[-1]
                prev["text"]     = prev["text"] + " " + text
                prev["end"]      = end
                prev["duration"] = round(prev["end"] - prev["start"], 3)
                continue

        merged.append({
            "speaker":  speaker,
            "start":    round(start, 3),
            "end":      round(end, 3),
            "duration": round(dur, 3),
            "text":     text,
        })

    for idx, s in enumerate(merged):
        s["id"] = f"{speaker}_{idx + 1:03d}"

    return merged


def interleave(agent_sents: list[dict], customer_sents: list[dict]) -> list[dict]:
    """Merge both speaker lists sorted by start time, add global seq_id."""
    combined = sorted(agent_sents + customer_sents, key=lambda s: s["start"])
    for i, s in enumerate(combined):
        s["seq_id"] = i + 1
    return combined


def main():
    with open(TRANSCRIPT, encoding="utf-8") as f:
        data = json.load(f)

    agent_sents    = words_to_sentences(data["agent"],    "AGENT")
    customer_sents = words_to_sentences(data["customer"], "CUSTOMER")
    all_sents      = interleave(agent_sents, customer_sents)

    output = {
        "call":               data.get("call", "call_1"),
        "model":              data.get("model"),
        "total":              len(all_sents),
        "agent_sentences":    len(agent_sents),
        "customer_sentences": len(customer_sents),
        "sentences":          all_sents,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"call_1  |  {len(agent_sents)} agent + {len(customer_sents)} customer"
          f" = {len(all_sents)} sentences total")
    print(f"saved -> {os.path.abspath(OUTPUT)}")

    print("\nfirst 8 sentences:")
    for s in all_sents[:8]:
        print(f"  [{s['id']:>12}] seq={s['seq_id']:>3}  "
              f"{s['start']:>7.2f}s – {s['end']:>7.2f}s  "
              f"({s['duration']:.2f}s)  \"{s['text'][:65]}\"")


if __name__ == "__main__":
    main()
