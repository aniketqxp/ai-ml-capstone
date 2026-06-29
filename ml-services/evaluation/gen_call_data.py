"""
Build the frontend's call_data.json from transcript_data.json:
  - groups words into speaker turns (agent/customer)
  - chapters the call via the LiteLLM router
  - emits turns + chapters with numeric seconds, ready for the React player

Output: frontend/src/call_data.json
  { call, duration, turns:[{i,speaker,text,start,end}],
    chapters:[{index,label,summary,start,end,start_turn}] }
"""
import os, json, time
from env_util import load_env
from segment import SYSTEM, SKELETON, numbered_transcript
from router import chat_json_routed
from extract import strip_fences

load_env()

FRONTEND = r"d:\Desktop\Main\Projects\ai-ml-capstone\frontend"
TRANSCRIPT = os.path.join(FRONTEND, "transcript_data.json")
OUT = os.path.join(FRONTEND, "src", "call_data.json")


def build_turns(data):
    """Merge agent+customer words into time-ordered speaker turns."""
    merged = []
    for speaker, key in (("AGENT", "agent"), ("CUSTOMER", "customer")):
        for w in data.get(key, []):
            if w.get("start") is None:
                continue
            merged.append((w["start"], w.get("end", w["start"]),
                           speaker, w["word"]))
    merged.sort(key=lambda x: x[0])

    turns = []
    cur = None
    for start, end, sp, word in merged:
        if cur is None or sp != cur["speaker"]:
            if cur:
                turns.append(cur)
            cur = {"speaker": sp, "text": word, "start": start, "end": end}
        else:
            cur["text"] += " " + word
            cur["end"] = end
    if cur:
        turns.append(cur)

    for i, t in enumerate(turns):
        t["i"] = i
        t["start"] = round(t["start"], 2)
        t["end"] = round(t["end"], 2)
    return turns


def repair_chapters(seeds, turns, duration):
    n = len(turns)
    cleaned, seen = [], set()
    for s in seeds:
        try:
            st = max(0, min(int(s["start_turn"]), n - 1))
        except (KeyError, ValueError, TypeError):
            continue
        if st in seen:
            continue
        seen.add(st)
        cleaned.append({"start_turn": st,
                        "label": str(s.get("label", "Untitled")).strip(),
                        "summary": str(s.get("summary", "")).strip()})
    cleaned.sort(key=lambda x: x["start_turn"])
    if not cleaned:
        cleaned = [{"start_turn": 0, "label": "Full Call", "summary": ""}]
    if cleaned[0]["start_turn"] != 0:
        cleaned[0]["start_turn"] = 0

    chapters = []
    for i, c in enumerate(cleaned):
        st = c["start_turn"]
        start = turns[st]["start"]
        end = (turns[cleaned[i + 1]["start_turn"]]["start"]
               if i + 1 < len(cleaned) else duration)
        chapters.append({"index": i + 1, "label": c["label"],
                         "summary": c["summary"], "start": round(start, 2),
                         "end": round(end, 2), "start_turn": st})
    return chapters


def main():
    with open(TRANSCRIPT, encoding="utf-8") as f:
        data = json.load(f)

    turns = build_turns(data)
    duration = round(max(t["end"] for t in turns), 2)
    print(f"call={data.get('call')} | {len(turns)} turns | {duration:.0f}s")

    user = f"{numbered_transcript(turns)}\n\n{SKELETON}"
    t0 = time.time()
    raw, served = chat_json_routed(SYSTEM, user, max_tokens=1500, return_meta=True)
    seeds = json.loads(strip_fences(raw)).get("chapters", [])
    chapters = repair_chapters(seeds, turns, duration)
    print(f"chaptered by {served} in {time.time()-t0:.1f}s -> {len(chapters)} chapters")
    for c in chapters:
        print(f"  [{c['index']}] {c['start']:6.1f}-{c['end']:6.1f}  {c['label']}")

    out = {"call": data.get("call", "call_1"),
           "model": data.get("model", ""),
           "duration": duration, "turns": turns, "chapters": chapters}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
