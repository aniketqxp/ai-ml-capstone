"""
Run the BASELINE evaluator (extract.py's rubric + router) on the frontend's
call_data.json, then ANCHOR every evidence quote to a real timestamp by locating
it in the transcript turns (don't trust the LLM's timestamp -- same robustness
principle as chaptering).

Merges an "evaluation" block into frontend/src/call_data.json so the React
compliance panel can light up each check at the moment its evidence occurs.

Usage:  python eval_call_data.py
"""
import os, json, time
from env_util import load_env
from extract import SYSTEM, SKELETON, strip_fences
from router import chat_json_routed
from rubric import CallEvaluation, RUBRIC_VERSION

load_env()
CALL_DATA = r"d:\Desktop\ai-ml-capstone\frontend\src\call_data.json"


def mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def build_packet(turns, duration, domain="banking"):
    header = (f"CALL METADATA: domain={domain} | duration={mmss(duration)} "
              f"| turns={len(turns)}\n\nROLE-MAPPED TRANSCRIPT:\n")
    body = "\n".join(f"[{t['speaker']} {mmss(t['start'])}] {t['text']}" for t in turns)
    return header + body


def _norm(s):
    return " ".join(s.lower().split())


def anchor(quote, turns):
    """Locate a quote in the transcript -> (start_sec, turn_index). None if unfound."""
    if not quote:
        return None, None
    q = _norm(quote)
    for t in turns:
        if q and q in _norm(t["text"]):
            return t["start"], t["i"]
    # fuzzy fallback: first 6 words of the quote
    frag = " ".join(q.split()[:6])
    if frag:
        for t in turns:
            if frag in _norm(t["text"]):
                return t["start"], t["i"]
    return None, None


def anchor_evidence(obj, turns, stats):
    """Recursively attach 'sec' + 'turn' to every evidence object with a quote."""
    if isinstance(obj, dict):
        if isinstance(obj.get("quote"), str) and obj["quote"]:
            sec, ti = anchor(obj["quote"], turns)
            obj["sec"] = sec
            obj["turn"] = ti
            stats["total"] += 1
            if sec is not None:
                stats["anchored"] += 1
        for v in obj.values():
            anchor_evidence(v, turns, stats)
    elif isinstance(obj, list):
        for item in obj:
            anchor_evidence(item, turns, stats)


def main():
    with open(CALL_DATA, encoding="utf-8") as f:
        cd = json.load(f)
    turns = cd["turns"]
    duration = cd["duration"]

    packet = build_packet(turns, duration)
    user = f"{packet}\n\n{SKELETON}"

    t0 = time.time()
    raw, served = chat_json_routed(SYSTEM, user, max_tokens=4000, return_meta=True)
    dt = time.time() - t0

    data = json.loads(strip_fences(raw))
    data["rubric_version"] = RUBRIC_VERSION
    data["metadata"] = {"call_id": cd.get("call", "call_1"), "domain": "banking",
                        "duration_seconds": duration,
                        "transcript_model": cd.get("model")}
    ev = CallEvaluation.model_validate(data)
    ev_dict = ev.model_dump()

    stats = {"total": 0, "anchored": 0}
    anchor_evidence(ev_dict, turns, stats)
    ev_dict["served_by"] = served

    cd["evaluation"] = ev_dict
    with open(CALL_DATA, "w", encoding="utf-8") as f:
        json.dump(cd, f, indent=2)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\nevaluated by {served} in {dt:.1f}s  |  "
          f"evidence anchored {stats['anchored']}/{stats['total']}")
    print("=" * 64)
    print("  COMPLIANCE")
    c = ev_dict["compliance"]
    for key, name in [("name_announced", "Name announced"),
                      ("company_announced", "Company announced"),
                      ("recording_disclosure", "Recording disclosure"),
                      ("identity_verified", "Identity verified"),
                      ("resolution_provided", "Resolution provided"),
                      ("transfer_next_steps", "Transfer next-steps")]:
        item = c[key]
        mark = {True: "PASS", False: "FAIL", None: "N/A "}[item["passed"]]
        evd = item.get("evidence")
        when = f" @{mmss(evd['sec'])}" if evd and evd.get("sec") is not None else ""
        quote = f'  "{evd["quote"][:48]}"' if evd else (f"  ({item.get('note')})" if item.get("note") else "")
        print(f"    [{mark}]{when:>7} {name:<22}{quote}")
    if c.get("identity_method"):
        print(f"              method: {c['identity_method']}")

    print("\n  QUALITY")
    q = ev_dict["quality"]
    for key, name in [("efficiency", "Efficiency"), ("problem_resolution", "Problem resolution"),
                      ("clarity", "Clarity"), ("professionalism", "Professionalism"),
                      ("empathy", "Empathy")]:
        d = q[key]
        aud = " (+audio)" if d["requires_audio"] else ""
        print(f"    {d['score']}/5  {name:<20}{aud}")

    e = ev_dict["escalation"]
    print(f"\n  ESCALATION: {e['risk_level'].upper()} | emotion: {e['customer_emotion_text']}"
          f" | flags: {', '.join(e['red_flags']) if e['red_flags'] else 'none'}")
    print(f"\n  SUMMARY: {ev_dict.get('overall_summary')}")
    print("=" * 64)
    print(f"\nMerged 'evaluation' into {CALL_DATA}")


if __name__ == "__main__":
    main()
