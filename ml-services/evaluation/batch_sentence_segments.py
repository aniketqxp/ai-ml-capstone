"""
Batch sentence segmentation for banking, health, and telecom calls.

Reads every word-level transcription JSON from data/na_testset/results/
for the target domains and produces sentence_segments.json per call under:

  data/sentence_segments/
    banking/
      en_CA_Banking_1586889.json
      ...
    health/
      en_CA_Health_1587315.json
      ...
    telecom/
      en_CA_Telecom_1590675.json
      ...

Each output file has the same schema as frontend/src/sentence_segments.json
so the frontend and sentiment pipeline can consume any call directly.

Usage:  python batch_sentence_segments.py [--domains banking health telecom]
"""
import os
import json
import re
import argparse
from pathlib import Path

import paths

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT = paths.NA_TESTSET / "results"
OUTPUT_ROOT  = paths.SENTENCE_SEG_ROOT

DEFAULT_DOMAINS = {"banking", "health", "telecom"}

MIN_DURATION_S = 1.5   # merge non-terminated fragments shorter than this
GAP_SPLIT_S    = 2.0   # silence gap that always starts a new sentence
MAX_MERGE_GAP  = 1.0   # only merge a short fragment if gap from prev ≤ this

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "st",
    "ave", "blvd", "dept", "est", "approx", "etc", "e.g", "i.e",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}
SENTENCE_END = re.compile(r"[.?!]$")


def is_sentence_end(word: str) -> bool:
    if not SENTENCE_END.search(word):
        return False
    base = re.sub(r"[.?!,;:]+$", "", word).lower()
    if base in ABBREVIATIONS:
        return False
    if re.match(r"^[a-z]$", base):
        return False
    return True


def words_to_sentences(words: list, speaker: str) -> list:
    if not words:
        return []

    raw = []
    bucket = [words[0]]
    for prev_w, curr_w in zip(words, words[1:]):
        gap = curr_w["start"] - prev_w["end"]
        if is_sentence_end(prev_w["word"]) or gap >= GAP_SPLIT_S:
            raw.append(bucket)
            bucket = [curr_w]
        else:
            bucket.append(curr_w)
    if bucket:
        raw.append(bucket)

    merged = []
    for sent_words in raw:
        start = sent_words[0]["start"]
        end   = sent_words[-1]["end"]
        dur   = end - start
        text  = " ".join(w["word"] for w in sent_words)
        last_word_terminal = SENTENCE_END.search(sent_words[-1]["word"])

        if merged and dur < MIN_DURATION_S and not last_word_terminal:
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


def interleave(agent_sents: list, customer_sents: list) -> list:
    combined = sorted(agent_sents + customer_sents, key=lambda s: s["start"])
    for i, s in enumerate(combined):
        s["seq_id"] = i + 1
    return combined


def segment_call(src_path: Path) -> dict:
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    agent_sents    = words_to_sentences(data.get("agent", []),    "AGENT")
    customer_sents = words_to_sentences(data.get("customer", []), "CUSTOMER")
    all_sents      = interleave(agent_sents, customer_sents)

    return {
        "call_id":            data["call_id"],
        "accent":             data.get("accent"),
        "domain":             data.get("domain"),
        "model":              data.get("model"),
        "total":              len(all_sents),
        "agent_sentences":    len(agent_sents),
        "customer_sentences": len(customer_sents),
        "sentences":          all_sents,
    }


def collect_sources(domains: set) -> list[tuple[str, Path]]:
    """Return (domain_label, src_path) for every matching call."""
    sources = []
    for accent_dir in sorted(RESULTS_ROOT.iterdir()):
        if not accent_dir.is_dir():
            continue
        for json_file in sorted(accent_dir.glob("*.json")):
            stem = json_file.stem  # e.g. en_CA_Banking_1586889
            # domain is the part between accent and numeric id
            parts = stem.split("_")
            # find the domain token — the first purely alpha segment after the accent
            # accent = en_CA or en_US_General (variable length)
            # try matching known domains
            stem_lower = stem.lower()
            matched = None
            for d in domains:
                if f"_{d}_" in stem_lower or stem_lower.endswith(f"_{d}"):
                    matched = d
                    break
            if matched:
                sources.append((matched, json_file))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+",
                    default=sorted(DEFAULT_DOMAINS),
                    help="domains to segment (any manifest domain; "
                         "default: banking health telecom)")
    args = ap.parse_args()
    domains = set(args.domains)

    sources = collect_sources(domains)
    if not sources:
        print("No matching calls found. Check RESULTS_ROOT path.")
        return

    counts = {d: 0 for d in domains}
    total_sentences = 0

    for domain, src in sources:
        out_dir = OUTPUT_ROOT / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src.name

        result = segment_call(src)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        counts[domain] += 1
        total_sentences += result["total"]
        print(f"  [{domain:8}]  {src.stem:<40}  "
              f"{result['agent_sentences']:>3}A + {result['customer_sentences']:>3}C "
              f"= {result['total']:>4} sentences  ->  {out_path.relative_to(OUTPUT_ROOT.parent.parent)}")

    print()
    print("-" * 72)
    for d in sorted(counts):
        print(f"  {d:10}  {counts[d]} calls")
    print(f"  {'TOTAL':10}  {sum(counts.values())} calls  ·  {total_sentences} sentences")
    print(f"\nOutput root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
