"""
Phase 0 context-engineering: turn our per-channel transcript into the
evaluation INPUT PACKET the LLM will consume.

Steps (deterministic, no LLM):
  1. Role mapping  -- channel identity IS the speaker (AGENT=ch1, CUSTOMER=ch2).
                      No diarization, no "who spoke first" heuristic -> 100% correct.
  2. Turn ordering -- merge both channels' segments by start time into a single
                      chronological [AGENT]/[CUSTOMER] turn list with mm:ss stamps.
  3. Metadata injection -- prepend call metadata (domain as "Call Type" lens,
                      duration, etc.) so the LLM judges with the right context.

Output: a single text packet (+ saved .txt) ready to drop into the Phase 1 prompt.

Usage:
  python assemble.py --result ../../data/na_testset/results_channels/en-CA/en_CA_Banking_1592237.json
  python assemble.py --call_id en_CA_Banking_1592237   # auto-locates result + manifest
"""
import os, json, argparse

DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST = os.path.join(DATA, "manifest.json")


def mmss(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return {m["call_id"]: m for m in json.load(f)}


def assemble_turns(result):
    """Merge agent+customer segments into one time-ordered turn list."""
    turns = []
    segs = result.get("segments", {})
    if segs and segs.get("agent") is not None:
        for speaker, key in (("AGENT", "agent"), ("CUSTOMER", "customer")):
            for s in segs.get(key, []):
                text = s.get("text", "").strip()
                if text:
                    turns.append({"start": s["start"], "speaker": speaker, "text": text})
    else:
        # fallback: no segments -> group words by speaker flips
        merged = []
        for speaker, key in (("AGENT", "agent"), ("CUSTOMER", "customer")):
            for w in result.get(key, []):
                merged.append((w["start"], speaker, w["word"]))
        merged.sort()
        cur_sp, cur_words, cur_start = None, [], None
        for start, sp, word in merged:
            if sp != cur_sp and cur_words:
                turns.append({"start": cur_start, "speaker": cur_sp, "text": " ".join(cur_words)})
                cur_words = []
            if sp != cur_sp:
                cur_sp, cur_start = sp, start
            cur_words.append(word)
        if cur_words:
            turns.append({"start": cur_start, "speaker": cur_sp, "text": " ".join(cur_words)})

    turns.sort(key=lambda t: t["start"])
    return turns


def estimate_duration(result):
    last = 0.0
    for key in ("agent", "customer"):
        for w in result.get(key, []):
            last = max(last, w.get("end", w.get("start", 0)))
    return round(last, 1)


def build_packet(result, meta):
    turns = assemble_turns(result)
    dur = estimate_duration(result)

    # ── metadata header (the "lens") ──────────────────────────────────────────
    header = [
        "=" * 70,
        "CALL METADATA",
        "=" * 70,
        f"Call ID      : {result.get('call_id', meta.get('call_id', '?'))}",
        f"Call Type    : {result.get('domain', meta.get('domain', 'unknown'))}",
        f"Duration     : {mmss(dur)} ({dur:.0f}s)",
        f"Transcript   : {result.get('model', 'unknown')} (per-channel; AGENT=ch1, CUSTOMER=ch2)",
        f"Turns        : {len(turns)}",
        "",
        "=" * 70,
        "ROLE-MAPPED TRANSCRIPT  (chronological; [SPEAKER mm:ss])",
        "=" * 70,
    ]

    body = [f"[{t['speaker']} {mmss(t['start'])}] {t['text']}" for t in turns]
    return "\n".join(header) + "\n" + "\n".join(body), {"turns": len(turns), "duration_s": dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", help="path to a per-channel result JSON")
    ap.add_argument("--call_id", help="call id (auto-locates result via manifest)")
    ap.add_argument("--results_dir", default="results_channels",
                    help="result dir under na_testset (default: results_channels)")
    ap.add_argument("--out", default=None, help="output .txt path (default: evaluation/packets/<id>.txt)")
    args = ap.parse_args()

    manifest = load_manifest()

    if args.result:
        path = args.result
    elif args.call_id:
        meta = manifest[args.call_id]
        path = os.path.join(DATA, args.results_dir, meta["accent"], args.call_id + ".json")
    else:
        raise SystemExit("Provide --result or --call_id")

    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    cid = result.get("call_id")
    meta = manifest.get(cid, {})
    packet, stats = build_packet(result, meta)

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "packets", f"{cid}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(packet)

    print(packet)
    print("\n" + "-" * 70)
    print(f"Saved packet -> {out}  ({stats['turns']} turns, {mmss(stats['duration_s'])})")


if __name__ == "__main__":
    main()
