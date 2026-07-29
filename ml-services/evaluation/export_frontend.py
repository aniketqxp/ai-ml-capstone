"""
Export every evaluated call into the frontend's per-call format + a merged
playable audio file, plus a summary index for the dashboard.

For each call_id with a results/{call_id}_graph.json:
  - turns     rebuilt from the per-channel word-level transcript, using the
              same word-grouping logic as gen_call_data.py::build_turns
              (assemble.py's assemble_turns() drops the per-turn `end` field
              the frontend needs, so it isn't reused here)
  - chapters  from segment.segment() (one LLM call per call -- chapters were
              never persisted anywhere), converted from its start_turn/mm:ss
              shape into the numeric start/end-seconds shape the frontend
              player expects
  - evaluation  the existing results/{call_id}_graph.json content, verbatim
  - audio     agent+customer wav merged into one 2-channel mp3 via ffmpeg
              (ch0=agent, ch1=customer -- same convention as the original
              call_1 demo audio)

Writes (served statically, fetched at runtime -- not bundled into the JS build):
  frontend/public/calls/{call_id}.json   {call, model, duration, turns, chapters, evaluation}
  frontend/public/word-timings/{call_id}.json
  frontend/public/calls_index.json       [{call_id, domain, ...summary}, ...]
  frontend/public/audio/{call_id}.mp3

The demo call (en_CA_Banking_1586889) is copied verbatim from the existing
call_data.json instead of being regenerated -- it already carries the
recheck-node bugfix's hand-verified evidence patch.

Usage:  python export_frontend.py
"""
import json
import os
import shutil
import subprocess

import paths
from assemble import DATA, load_manifest
from segment import segment

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_NAME = "results"

PUBLIC = str(paths.FRONTEND_PUBLIC)
CALLS_OUT = os.path.join(PUBLIC, "calls")
AUDIO_OUT = os.path.join(PUBLIC, "audio")
WORD_TIMINGS_OUT = os.path.join(PUBLIC, "word-timings")
INDEX_OUT = os.path.join(PUBLIC, "calls_index.json")

DEMO_CALL = "en_CA_Banking_1586889"
DEMO_CALL_DATA = str(paths.FRONTEND_ROOT / "src" / "call_data.json")
DEMO_AUDIO_SRC = os.path.join(PUBLIC, "call_1.mp3")


def build_turns(result):
    """Merge agent+customer words into time-ordered speaker turns."""
    merged = []
    for speaker, key in (("AGENT", "agent"), ("CUSTOMER", "customer")):
        for w in result.get(key, []):
            if w.get("start") is None:
                continue
            merged.append(
                (
                    w["start"],
                    w.get("end", w["start"]),
                    speaker,
                    w["word"],
                )
            )
    merged.sort(key=lambda x: x[0])

    turns = []
    cur = None
    for start, end, sp, word in merged:
        timed_word = {
            "word": word,
            "start": round(start, 2),
            "end": round(end, 2),
        }
        if cur is None or sp != cur["speaker"]:
            if cur:
                turns.append(cur)
            cur = {
                "speaker": sp,
                "text": word,
                "start": start,
                "end": end,
                "words": [timed_word],
            }
        else:
            cur["text"] += " " + word
            cur["end"] = end
            cur["words"].append(timed_word)
    if cur:
        turns.append(cur)

    for i, t in enumerate(turns):
        t["i"] = i
        t["start"] = round(t["start"], 2)
        t["end"] = round(t["end"], 2)
    return turns


def numeric_chapters(cc_chapters, turns, duration):
    """Convert segment.py's start_turn/mm:ss chapters into numeric seconds."""
    chapters = []
    for i, ch in enumerate(cc_chapters):
        start = turns[ch.start_turn]["start"]
        end = (turns[cc_chapters[i + 1].start_turn]["start"]
               if i + 1 < len(cc_chapters) else duration)
        chapters.append({"index": ch.index, "label": ch.label,
                          "summary": ch.summary, "start": round(start, 2),
                          "end": round(end, 2), "start_turn": ch.start_turn})
    return chapters


def merge_audio(agent_wav, customer_wav, out_mp3):
    os.makedirs(os.path.dirname(out_mp3), exist_ok=True)
    proc = subprocess.run([
        "ffmpeg", "-y", "-i", agent_wav, "-i", customer_wav,
        "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
        "-map", "[a]", "-ac", "2", "-b:a", "128k", out_mp3,
    ], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"    ffmpeg FAILED for {out_mp3}:\n{proc.stderr[-800:]}")
        return False
    return True


def export_call_artifacts(call_id, meta, out_calls_dir, out_audio_dir):
    """Build one call's frontend artifacts into the given output dirs.

    Writes {call_id}.json (turns + chapters + evaluation) and the merged
    2-channel mp3. Reads the graph evaluation and the per-channel transcript
    already on disk; runs one chapter LLM call. Channel WAVs are expected at
    the dataset convention DATA/{accent}/{call_id}_{agent,customer}.wav (the
    orchestrator places fresh-call channels there before calling this).
    Returns the call dict, which summarize() turns into a dashboard index row.
    """
    eval_path = os.path.join(EVAL_DIR, "results", f"{call_id}_graph.json")
    with open(eval_path, encoding="utf-8") as f:
        evaluation = json.load(f)
    duration = evaluation["metadata"]["duration_seconds"]

    result_path = os.path.join(DATA, RESULTS_DIR_NAME, meta["accent"], call_id + ".json")
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    turns = build_turns(result)

    cc, _duration, issues = segment(
        call_id,
        provider="router",
        results_dir=RESULTS_DIR_NAME,
    )
    if issues:
        print(f"    chapter issues: {issues}")
    chapters = numeric_chapters(cc.chapters, turns, duration)

    out = {
        "call": call_id,
        "model": result.get("model", ""),
        "duration": round(duration, 2),
        "turns": turns,
        "chapters": chapters,
        "evaluation": evaluation,
    }
    os.makedirs(out_calls_dir, exist_ok=True)
    with open(os.path.join(out_calls_dir, f"{call_id}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.makedirs(WORD_TIMINGS_OUT, exist_ok=True)
    with open(
        os.path.join(WORD_TIMINGS_OUT, f"{call_id}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({"call_id": call_id, "turns": turns}, f, indent=2)

    ok = merge_audio(
        os.path.join(DATA, meta["accent"], f"{call_id}_agent.wav"),
        os.path.join(DATA, meta["accent"], f"{call_id}_customer.wav"),
        os.path.join(out_audio_dir, f"{call_id}.mp3"),
    )
    print(f"  [{call_id}] {len(turns)} turns, {len(chapters)} chapters, "
          f"audio {'ok' if ok else 'FAILED'}")
    return out


def export_call(call_id, manifest):
    return export_call_artifacts(call_id, manifest[call_id], CALLS_OUT, AUDIO_OUT)


def summarize(call_id, out):
    ev = out["evaluation"]
    c = ev.get("compliance", {})
    passed = sum(1 for v in c.values() if isinstance(v, dict) and v.get("passed") is True)
    failed = sum(1 for v in c.values() if isinstance(v, dict) and v.get("passed") is False)
    wf = ev.get("workflow") or {}
    steps = wf.get("expected_steps", [])
    esc = ev.get("escalation", {})
    meta = ev.get("metadata", {})
    return {
        "call_id": call_id,
        "domain": meta.get("domain"),
        "accent": meta.get("accent"),
        "duration": out["duration"],
        "compliance_passed": passed,
        "compliance_applicable": passed + failed,
        "workflow_met": sum(1 for s in steps if s.get("met") is True),
        "workflow_total": len(steps),
        "risk_level": esc.get("risk_level"),
        "subject": wf.get("subject"),
    }


def main():
    manifest = load_manifest()
    call_ids = sorted(
        f[:-len("_graph.json")]
        for f in os.listdir(os.path.join(EVAL_DIR, "results"))
        if f.endswith("_graph.json")
    )

    index = []

    with open(DEMO_CALL_DATA, encoding="utf-8") as f:
        demo = json.load(f)
    demo_result_path = os.path.join(
        DATA,
        RESULTS_DIR_NAME,
        manifest[DEMO_CALL]["accent"],
        DEMO_CALL + ".json",
    )
    with open(demo_result_path, encoding="utf-8") as f:
        demo["turns"] = build_turns(json.load(f))
    os.makedirs(CALLS_OUT, exist_ok=True)
    with open(os.path.join(CALLS_OUT, f"{DEMO_CALL}.json"), "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2)
    os.makedirs(WORD_TIMINGS_OUT, exist_ok=True)
    with open(
        os.path.join(WORD_TIMINGS_OUT, f"{DEMO_CALL}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {"call_id": DEMO_CALL, "turns": demo["turns"]},
            f,
            indent=2,
        )
    os.makedirs(AUDIO_OUT, exist_ok=True)
    shutil.copy(DEMO_AUDIO_SRC, os.path.join(AUDIO_OUT, f"{DEMO_CALL}.mp3"))
    index.append(summarize(DEMO_CALL, demo))
    print(f"[{DEMO_CALL}] copied verbatim (demo call, already patched)")

    for call_id in call_ids:
        if call_id == DEMO_CALL:
            continue
        out = export_call(call_id, manifest)
        index.append(summarize(call_id, out))

    with open(INDEX_OUT, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nExported {len(index)} calls -> {CALLS_OUT}")
    print(f"Index -> {INDEX_OUT}")


if __name__ == "__main__":
    main()
