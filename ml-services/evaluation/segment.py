"""
Call chaptering / segmentation layer.

Turns the role-mapped transcript into coherent CHAPTERS (phases of the call),
like automatic video chapters: greeting, identity verification, problem
statement, investigation, resolution, closing, etc.

DESIGN -- why this is robust
----------------------------
A naive approach asks the model for mm:ss boundaries, which it can hallucinate
(timestamps past the call end, overlapping spans, gaps). Instead we:

  1. NUMBER every turn and show the model the numbered transcript.
  2. Ask only for the START TURN INDEX of each chapter (+ label + summary).
  3. DERIVE all timestamps from the real turns, and REPAIR the result in code:
     - clamp indices into range, dedupe, sort
     - force chapter 1 to start at turn 0
     - each chapter ends exactly where the next begins (contiguous, no gaps)
     - the last chapter ends at the call duration

So the output is always valid by construction -- the model only chooses *where*
topics shift, never the raw numbers.

Uses the LiteLLM router by default (Mistral primary). Override with --provider.

Usage:
  python segment.py --call_id en_CA_Banking_1592237
  python segment.py --call_id en_CA_Banking_1592237 --provider github
"""
import os, json, time, argparse
from typing import List
from pydantic import BaseModel, Field, ValidationError

from env_util import load_env
from assemble import load_manifest, assemble_turns, estimate_duration, mmss
from llm_client import chat_json
from extract import strip_fences

load_env()
DATA = r"d:\Desktop\Main\Projects\ai-ml-capstone\data\na_testset"


# ── Schema ────────────────────────────────────────────────────────────────────
class Chapter(BaseModel):
    index: int
    label: str
    start_turn: int
    start_time: str          # mm:ss (derived)
    end_time: str            # mm:ss (derived)
    summary: str


class CallChapters(BaseModel):
    call_id: str
    domain: str
    duration: str
    n_turns: int
    n_chapters: int
    served_by: str
    chapters: List[Chapter]


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM = """You segment a customer-service phone call into coherent CHAPTERS -- the natural phases of the call, like automatic video chapters.

Typical phases (use only those that actually occur, in the order they occur):
greeting/introduction, identity verification, problem statement, investigation/discussion, options/explanation, resolution & next steps, upsell/offer, closing.

RULES:
- Identify between 3 and 8 chapters. Fewer for short calls, more for long ones.
- Each chapter is a run of CONSECUTIVE turns covering ONE phase. No overlaps, no gaps.
- Base boundaries on REAL topic shifts in the transcript, not fixed sizes.
- The first chapter MUST start at turn 0.
- For each chapter output: start_turn (the turn index where the phase BEGINS),
  a short label (2-4 words), and a one-sentence summary of what happens in it.
- Output ONLY JSON. No markdown, no commentary."""

SKELETON = """Return EXACTLY this shape:
{"chapters":[
  {"start_turn":0,"label":"Greeting & Introduction","summary":"Agent greets the caller and identifies themselves and the company."},
  {"start_turn":4,"label":"Identity Verification","summary":"Agent verifies the customer's identity before discussing the account."}
]}"""


def numbered_transcript(turns) -> str:
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"[{i}] [{t['speaker']} {mmss(t['start'])}] {t['text']}")
    return "\n".join(lines)


def _call(provider, system, user):
    """Return (raw_json_str, served_model)."""
    if provider == "router":
        from router import chat_json_routed
        return chat_json_routed(system, user, max_tokens=1500, return_meta=True)
    raw = chat_json(provider, system, user, max_tokens=1500)
    return raw, provider


def repair_chapters(seeds, turns, duration_str) -> List[Chapter]:
    """Turn raw model seeds into valid, contiguous, gap-free chapters."""
    n = len(turns)
    cleaned, seen = [], set()
    for s in seeds:
        try:
            st = int(s["start_turn"])
        except (KeyError, ValueError, TypeError):
            continue
        st = max(0, min(st, n - 1))
        if st in seen:
            continue
        seen.add(st)
        cleaned.append({"start_turn": st,
                        "label": str(s.get("label", "Untitled")).strip(),
                        "summary": str(s.get("summary", "")).strip()})

    cleaned.sort(key=lambda x: x["start_turn"])
    if not cleaned:
        cleaned = [{"start_turn": 0, "label": "Full Call",
                    "summary": "Entire call."}]
    # force coverage from the start
    if cleaned[0]["start_turn"] != 0:
        cleaned[0]["start_turn"] = 0

    chapters = []
    for i, c in enumerate(cleaned):
        st = c["start_turn"]
        start_time = mmss(turns[st]["start"])
        if i + 1 < len(cleaned):
            end_time = mmss(turns[cleaned[i + 1]["start_turn"]]["start"])
        else:
            end_time = duration_str
        chapters.append(Chapter(index=i + 1, label=c["label"], start_turn=st,
                                start_time=start_time, end_time=end_time,
                                summary=c["summary"]))
    return chapters


def segment(call_id, provider="router", results_dir="results_channels"):
    manifest = load_manifest()
    meta = manifest[call_id]
    result_path = os.path.join(DATA, results_dir, meta["accent"], call_id + ".json")
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    turns = assemble_turns(result)
    duration_str = mmss(estimate_duration(result))
    user = f"{numbered_transcript(turns)}\n\n{SKELETON}"

    t0 = time.time()
    raw, served = _call(provider, SYSTEM, user)
    dt = time.time() - t0

    data = json.loads(strip_fences(raw))
    seeds = data.get("chapters", [])
    chapters = repair_chapters(seeds, turns, duration_str)

    cc = CallChapters(
        call_id=call_id, domain=result.get("domain", meta.get("domain", "?")),
        duration=duration_str, n_turns=len(turns), n_chapters=len(chapters),
        served_by=served, chapters=chapters)
    issues = verify_invariants(cc, turns)
    return cc, dt, issues


def verify_invariants(cc: CallChapters, turns) -> List[str]:
    """Return a list of invariant violations (empty = correct)."""
    issues = []
    chs = cc.chapters
    if not chs:
        return ["no chapters produced"]
    if chs[0].start_turn != 0:
        issues.append("chapter 1 does not start at turn 0")
    if chs[0].start_time != mmss(turns[0]["start"]):
        issues.append("first chapter start != call start")
    if chs[-1].end_time != cc.duration:
        issues.append("last chapter end != call duration (coverage gap)")
    for a, b in zip(chs, chs[1:]):
        if a.end_time != b.start_time:
            issues.append(f"gap/overlap between ch{a.index} and ch{b.index}")
        if b.start_turn <= a.start_turn:
            issues.append(f"non-increasing start_turn at ch{b.index}")
    if not (1 <= cc.n_chapters <= 12):
        issues.append(f"chapter count {cc.n_chapters} outside sane range")
    return issues


def print_timeline(cc: CallChapters, dt, issues=None):
    print("\n" + "=" * 70)
    print(f"  CALL TIMELINE  {cc.call_id}  ({cc.domain})")
    print(f"  {cc.duration} | {cc.n_turns} turns | {cc.n_chapters} chapters "
          f"| {dt:.1f}s | {cc.served_by}")
    print("=" * 70)
    for ch in cc.chapters:
        print(f"\n  {ch.start_time}-{ch.end_time}   [{ch.index}] {ch.label}")
        print(f"                  {ch.summary}")
    print("\n  " + "-" * 66)
    if issues is None:
        print("  invariants: (not checked)")
    elif issues:
        print(f"  invariants: {len(issues)} VIOLATION(S):")
        for x in issues:
            print(f"     - {x}")
    else:
        print("  invariants: OK (contiguous, gap-free, full coverage)")
    print("=" * 70)


def main():
    from llm_client import list_providers
    ap = argparse.ArgumentParser()
    ap.add_argument("--call_id", required=True)
    ap.add_argument("--provider", default="router",
                    choices=["router"] + list_providers())
    ap.add_argument("--results_dir", default="results_channels")
    args = ap.parse_args()

    cc, dt, issues = segment(args.call_id, args.provider, args.results_dir)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "chapters")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.call_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cc.model_dump(), f, indent=2)

    print_timeline(cc, dt, issues)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
