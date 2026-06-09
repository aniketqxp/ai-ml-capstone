"""
Error analysis over the Phase 1 per-channel probe results. No new transcription.

Answers the two questions that decide the next phase:

  Q1 (Phase 5 / LLM): WHAT are the remaining errors? Categorize normalized
     substitutions/deletions/insertions (numbers / morphology / function words /
     lexical-content) and surface the most frequent lexical ref->hyp pairs --
     i.e. exactly what an LLM cleanup pass would need to fix.

  Q2 (Phase 3 / routing): does word CONFIDENCE predict errors? Bucket hypothesis
     words by their decoding probability (and segment no_speech_prob) and show the
     error rate per bucket. If errors concentrate in low-confidence words, we can
     gate any correction pass to those spans (cheaper, less over-correction).

Pass 1 (categorization) runs at the normalized level so number/contraction
formatting is collapsed -> only genuine errors are counted.
Pass 2 (confidence) runs at the raw level so hypothesis tokens map 1:1 onto the
original word objects that carry prob/ns.
"""

import os
import json
import re
import jiwer
from collections import Counter, defaultdict
from eval_common import normalise, normalise_raw

DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"
RES  = "results_channels"   # Phase 1

FUNCTION_WORDS = set("a an the is are was were be been being am to of in on at for "
                     "and or but so it its i you he she we they me him her us them my "
                     "your his our their this that these those as if then than do does "
                     "did has have had will would can could may might shall should must "
                     "not no yes ok okay oh uh um mm yeah".split())
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve "
                   "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
                   "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
                   "first second third fourth fifth".split())


def is_number(tok):
    return bool(re.search(r"\d", tok)) or tok in NUMBER_WORDS


def same_stem(a, b):
    """Crude morphology check: same word, different inflection (plural/tense)."""
    if a == b:
        return False
    for suf in ("s", "es", "d", "ed", "ing", "'s"):
        if a == b + suf or b == a + suf:
            return True
    # shared stem of length >=4
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i >= 4 and abs(len(a) - len(b)) <= 3


def categorize(ref_w, hyp_w):
    if is_number(ref_w) or is_number(hyp_w):
        return "number"
    if same_stem(ref_w, hyp_w):
        return "morphology"
    if ref_w in FUNCTION_WORDS or hyp_w in FUNCTION_WORDS:
        return "function"
    return "lexical"


def align_chunks(ref_str, hyp_str):
    out = jiwer.process_words(ref_str, hyp_str)
    refs, hyps = out.references[0], out.hypotheses[0]
    return refs, hyps, out.alignments[0]


def main():
    # ── Pass 1: normalized categorization ─────────────────────────────────────
    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    with open(os.path.join(DATA, "probe_set.json"), encoding="utf-8") as f:
        probe = json.load(f)["calls"]

    cat_counts = Counter()
    lexical_pairs = Counter()
    del_words = Counter()
    ins_words = Counter()
    per_call_err = {}
    tot_ref = tot_S = tot_D = tot_I = 0

    # ── Pass 2 accumulators (raw level) ───────────────────────────────────────
    prob_buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    ns_buckets   = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]
    prob_stat = {b: [0, 0] for b in prob_buckets}   # [n_words, n_errors]
    ns_stat   = {b: [0, 0] for b in ns_buckets}
    err_lowprob = err_total = words_lowprob = words_total = 0

    for p in probe:
        cid, accent = p["call_id"], p["accent"]
        m = manifest[cid]
        with open(os.path.join(DATA, RES, accent, cid + ".json"), encoding="utf-8") as f:
            hyp = json.load(f)

        call_err = 0
        for ref_text, words in [(m["agent_transcript"], hyp["agent"]),
                                (m["customer_transcript"], hyp["customer"])]:
            # Pass 1: normalized
            refs, hyps, chunks = align_chunks(normalise(ref_text),
                                              normalise(" ".join(w["word"] for w in words)))
            for ch in chunks:
                if ch.type == "equal":
                    continue
                rspan = refs[ch.ref_start_idx:ch.ref_end_idx]
                hspan = hyps[ch.hyp_start_idx:ch.hyp_end_idx]
                if ch.type == "substitute":
                    tot_S += len(rspan); call_err += len(rspan)
                    for rw, hw in zip(rspan, hspan):
                        c = categorize(rw, hw)
                        cat_counts[c] += 1
                        if c == "lexical":
                            lexical_pairs[f"{rw} -> {hw}"] += 1
                elif ch.type == "delete":
                    tot_D += len(rspan); call_err += len(rspan)
                    for rw in rspan:
                        del_words[rw] += 1
                        cat_counts["number" if is_number(rw) else
                                   "function" if rw in FUNCTION_WORDS else "del_other"] += 1
                elif ch.type == "insert":
                    tot_I += len(hspan); call_err += len(hspan)
                    for hw in hspan:
                        ins_words[hw] += 1
            tot_ref += len(refs)

            # Pass 2: raw-level confidence mapping (1:1 token<->word meta)
            toks, meta = [], []
            for w in words:
                for t in normalise_raw(w["word"]).split():
                    toks.append(t); meta.append((w.get("prob", 1.0), w.get("ns", 0.0)))
            _, _, rchunks = align_chunks(normalise_raw(ref_text), " ".join(toks))
            err_flag = [False] * len(toks)
            for ch in rchunks:
                if ch.type in ("substitute", "insert"):
                    for hi in range(ch.hyp_start_idx, ch.hyp_end_idx):
                        if hi < len(err_flag):
                            err_flag[hi] = True
            for (prob, ns), is_err in zip(meta, err_flag):
                words_total += 1; err_total += is_err
                if prob < 0.5:
                    words_lowprob += 1; err_lowprob += is_err
                for b in prob_buckets:
                    if b[0] <= prob < b[1]:
                        prob_stat[b][0] += 1; prob_stat[b][1] += is_err; break
                for b in ns_buckets:
                    if b[0] <= ns < b[1]:
                        ns_stat[b][0] += 1; ns_stat[b][1] += is_err; break

        per_call_err[cid] = call_err

    # ── Report ────────────────────────────────────────────────────────────────
    print("=== ERROR ANALYSIS (Phase 1 per-channel, 12 probe calls) ===\n")
    acc = (1 - (tot_S + tot_D + tot_I) / tot_ref) * 100
    print(f"Normalized: {acc:.1f}% accuracy | {tot_ref} ref words | "
          f"S={tot_S} D={tot_D} I={tot_I}\n")

    print("Q1 -- Error type breakdown (normalized S+D+I):")
    total_cat = sum(cat_counts.values())
    for c, n in cat_counts.most_common():
        print(f"  {c:<12} {n:>4}  ({n/total_cat*100:4.1f}%)")

    print("\n  Top lexical substitutions (ref -> hyp) -- the LLM-fixable content errors:")
    for pair, n in lexical_pairs.most_common(20):
        print(f"    {pair:<32} x{n}")

    print("\n  Top deletions (ref words missing):", ", ".join(f"{w}({n})" for w, n in del_words.most_common(10)))
    print("  Top insertions (extra hyp words):  ", ", ".join(f"{w}({n})" for w, n in ins_words.most_common(10)))

    print("\nQ2 -- Does confidence predict errors? (raw-level hyp words)")
    print(f"  {'prob bucket':<14} {'#words':>7} {'#err':>6} {'err-rate':>9}")
    for b in prob_buckets:
        n, e = prob_stat[b]
        print(f"  [{b[0]:.2f},{b[1]-0.01:.2f}]  {n:>7} {e:>6} {(e/n*100 if n else 0):>8.1f}%")
    print(f"  {'no_speech':<14} {'#words':>7} {'#err':>6} {'err-rate':>9}")
    for b in ns_buckets:
        n, e = ns_stat[b]
        print(f"  ns[{b[0]:.1f},{b[1]-0.01:.2f}] {n:>7} {e:>6} {(e/n*100 if n else 0):>8.1f}%")

    if err_total:
        print(f"\n  Words with prob<0.5 hold {err_lowprob/err_total*100:.0f}% of all errors "
              f"while being only {words_lowprob/words_total*100:.0f}% of all words.")

    print("\nPer-call error count (normalized S+D+I), worst first:")
    for cid, n in sorted(per_call_err.items(), key=lambda x: -x[1]):
        tier = next(p["tier"] for p in probe if p["call_id"] == cid)
        print(f"  {tier:<13} {cid:<32} {n}")


if __name__ == "__main__":
    main()
