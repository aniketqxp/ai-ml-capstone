"""
Phase 5 (prototype): conservative LLM cleanup pass over Phase 1 transcripts.

Reframed by the error analysis: the remaining errors are dominated by
function-word/filler deletions an LLM cannot restore, so this pass is NOT a
WER-to-99 play. Its job is to fix the small, genuinely addressable slice --
misheard homophones, garbled proper nouns/names, obviously wrong words -- to
improve DOWNSTREAM transcript quality for compliance. The main risk is
OVER-correction (the LLM paraphrasing or "fixing" correct words), so:

  - the prompt is strict / minimal-edit, temperature low
  - text is corrected in small sentence-aware chunks (edits stay local)
  - a guardrail rejects a chunk's correction if its length deviates too much
  - WER is measured BEFORE vs AFTER as the guardrail metric: if WER regresses,
    the pass is doing more harm than good.

Model-agnostic (--model). Default qwen2.5:7b-instruct (best quality, needs
~5 GB free RAM); fall back to llama3.2:3b on low-RAM machines.

Usage:
  python llm_cleanup.py                          # all 12 probe calls
  python llm_cleanup.py --model llama3.2:3b --limit 3
"""

import os
import re
import json
import time
import argparse
import urllib.request
import difflib
import jiwer
from eval_common import normalise

DATA   = r"d:\Desktop\ai-ml-capstone\data\na_testset"
RES    = "results_channels"            # Phase 1 input
OLLAMA = "http://localhost:11434/api/generate"

PROMPT = """You are correcting errors in an automatic speech-recognition transcript of a {domain} phone call (the {speaker} is speaking). Fix ONLY clear recognition errors:
- misheard words and wrong homophones (e.g. "segway" -> "segue")
- misspelled or garbled proper nouns, names, and places
- obviously wrong words that don't fit the sentence

STRICT RULES:
- Do NOT paraphrase, rephrase, reorder, summarize, or improve style.
- Do NOT add or remove words. Keep ALL filler and disfluencies (um, uh, okay, yeah, repeated words).
- Keep all numbers exactly as written.
- Preserve punctuation and capitalization as-is unless clearly wrong.
- If the text is already correct, return it EXACTLY unchanged.
- Output ONLY the corrected transcript text. No preamble, no quotes, no commentary.

Transcript:
{text}"""


def ollama(prompt, model, temperature=0.1, timeout=180):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


def sentence_chunks(text, max_words=130):
    """Sentence-aware chunks of <= max_words, so corrections stay local and the
    model keeps sentence context."""
    sents = re.split(r"(?<=[.?!])\s+", text)
    chunk, n, out = [], 0, []
    for s in sents:
        w = len(s.split())
        if n + w > max_words and chunk:
            out.append(" ".join(chunk)); chunk, n = [], 0
        chunk.append(s); n += w
    if chunk:
        out.append(" ".join(chunk))
    return out


def strip_wrapper(s):
    s = s.strip()
    s = re.sub(r'^(here is|here\'s|corrected( transcript)?:?)\s*', '', s, flags=re.IGNORECASE).strip()
    if len(s) >= 2 and s[0] in '"“' and s[-1] in '"”':
        s = s[1:-1].strip()
    return s


def clean_text(text, domain, speaker, model, temperature, max_words):
    out, n_changed = [], 0
    for chunk in sentence_chunks(text, max_words):
        try:
            corrected = strip_wrapper(ollama(
                PROMPT.format(domain=domain, speaker=speaker, text=chunk), model, temperature))
        except Exception:
            corrected = chunk
        # guardrail: reject implausible rewrites (length blow-up/collapse or empty)
        if (not corrected) or abs(len(corrected.split()) - len(chunk.split())) > 0.25 * len(chunk.split()) + 3:
            corrected = chunk
        if corrected != chunk:
            n_changed += 1
        out.append(corrected)
    return " ".join(out), n_changed


def wer_norm(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    n = len(r.split())
    return (jiwer.wer(r, h) if n else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--out", default="results_channels_llm")
    ap.add_argument("--chunk-words", type=int, default=130)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    with open(os.path.join(DATA, "probe_set.json"), encoding="utf-8") as f:
        probe = json.load(f)["calls"]
    if args.limit:
        probe = probe[: args.limit]

    out_root = os.path.join(DATA, args.out)
    print(f"LLM cleanup | model={args.model} | {len(probe)} calls\n")
    print(f"  {'call_id':<32} {'before':>7} {'after':>7} {'delta':>6} {'chunks_chg':>11}")
    print("  " + "-" * 70)

    agg = {"before": [0.0, 0], "after": [0.0, 0]}
    t_start = time.time()
    for p in probe:
        cid, accent = p["call_id"], p["accent"]
        m = manifest[cid]
        with open(os.path.join(DATA, RES, accent, cid + ".json"), encoding="utf-8") as f:
            hyp = json.load(f)

        before_a = " ".join(w["word"] for w in hyp["agent"])
        before_c = " ".join(w["word"] for w in hyp["customer"])
        after_a, chg_a = clean_text(before_a, m["domain"], "agent",    args.model, args.temperature, args.chunk_words)
        after_c, chg_c = clean_text(before_c, m["domain"], "customer", args.model, args.temperature, args.chunk_words)

        # word-weighted WER before/after
        wba, na = wer_norm(m["agent_transcript"], before_a)
        wbc, nc = wer_norm(m["customer_transcript"], before_c)
        waa, _  = wer_norm(m["agent_transcript"], after_a)
        wac, _  = wer_norm(m["customer_transcript"], after_c)
        before = (wba * na + wbc * nc) / (na + nc)
        after  = (waa * na + wac * nc) / (na + nc)
        agg["before"][0] += before * (na + nc); agg["before"][1] += na + nc
        agg["after"][0]  += after  * (na + nc); agg["after"][1]  += na + nc

        os.makedirs(os.path.join(out_root, accent), exist_ok=True)
        with open(os.path.join(out_root, accent, cid + ".json"), "w", encoding="utf-8") as f:
            json.dump({"call_id": cid, "accent": accent, "domain": m["domain"], "model": args.model,
                       "agent_text": after_a, "customer_text": after_c,
                       "agent_text_before": before_a, "customer_text_before": before_c}, f, indent=2)

        d = (after - before) * 100
        print(f"  {cid:<32} {(1-before)*100:>6.1f}% {(1-after)*100:>6.1f}% {-d:>+5.1f} {chg_a+chg_c:>11}")

    ba = (1 - agg["before"][0] / agg["before"][1]) * 100
    aa = (1 - agg["after"][0] / agg["after"][1]) * 100
    print("  " + "-" * 70)
    print(f"  {'OVERALL':<32} {ba:>6.1f}% {aa:>6.1f}% {aa-ba:>+5.1f}")
    print(f"\n  {'IMPROVED' if aa > ba + 0.1 else 'NO GAIN / REGRESSED'} "
          f"| {len(probe)} calls in {(time.time()-t_start)/60:.1f} min")
    print("  (positive delta = WER reduced = good; negative = over-correction, pass is harmful)")


if __name__ == "__main__":
    main()
