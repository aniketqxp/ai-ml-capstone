"""
Phase 1 -- Monolithic Structured Extraction.

One LLM call turns the role-mapped transcript packet into a validated
CallEvaluation JSON (compliance booleans + quality scores + escalation), every
judgment backed by a verbatim quote. Pydantic validates; one retry on failure.

MODALITY DISCIPLINE is enforced in the prompt: the text model judges only what
text can prove, and flags acoustic items requires_audio rather than guessing.

Usage:
  python extract.py --call_id en_CA_Banking_1592237 [--provider github] [--model openai/gpt-4o]
"""
import os, json, argparse, time
from pydantic import ValidationError

from env_util import load_env
from assemble import load_manifest, build_packet
from rubric import CallEvaluation, CallMetadata, RUBRIC_VERSION
from llm_client import chat_json

load_env()
DATA = r"d:\Desktop\ai-ml-capstone\data\na_testset"

SYSTEM = f"""You are a meticulous call-center QA analyst. You evaluate a single customer-service call against a fixed rubric and output ONLY a JSON object. Rubric version {RUBRIC_VERSION}.

You are given a ROLE-MAPPED transcript: every line is tagged [AGENT mm:ss] or [CUSTOMER mm:ss]. The channel identity is authoritative -- never attribute the agent's words to the customer or vice versa.

EVALUATE THESE ITEMS:

COMPLIANCE (passed = true / false / null). null ONLY when the item does not apply to this call.
- name_announced: agent stated their own name.
- company_announced: agent stated the company/bank name.
- recording_disclosure: agent said the call may be recorded/monitored. If never stated, passed=false (do NOT assume). Use null only if clearly not required.
- identity_verified: agent verified the customer BEFORE discussing the account (DOB, last 4 digits, account number, customer ID). Set identity_method to the method used.
- resolution_provided: agent gave a resolution or clear next steps before closing.
- transfer_next_steps: if the call was transferred, agent explained next steps. passed=null if there was no transfer.

QUALITY (score 1-5 from TEXT signals; list signals_present / signals_absent):
- efficiency: first-call resolution, no unnecessary holds/transfers.
- problem_resolution: offered options, explained pros/cons and WHY, respected customer choice, solved the actual issue.
- clarity: open-ended questions, confirms understanding ("does that make sense?"), recap at end.
- professionalism: grammar, complete sentences, no slang. (Tone/pace are ACOUSTIC -> set requires_audio=true.)
- empathy: validation phrases ("I understand your frustration"), uses customer name, acknowledges concern. (Warmth/flat tone are ACOUSTIC -> set requires_audio=true.)

ESCALATION:
- red_flags: any of manager_requested, competitor_switch, repeat_attempts, issue_too_complex, explicit_dissatisfaction.
- customer_emotion_text: infer from WORDS only (calm/mild_frustration/frustrated/angry/distressed).
- risk_level: none / review / escalate.

HARD RULES:
1. EVIDENCE: every compliance true/false and every quality score must cite a verbatim quote copied EXACTLY from the transcript, with speaker and timestamp. Never invent quotes.
2. MODALITY: judge ONLY from text. For acoustic qualities (tone, pace, energy, whether the customer "sounds" satisfied) do not guess -- set requires_audio=true and score from text signals alone.
3. CONSERVATISM: do not mark a compliance item true unless the transcript explicitly shows it.
4. OUTPUT: a single JSON object, no markdown, no commentary."""

SKELETON = """Return EXACTLY this JSON shape (fill values; keep keys):
{
  "compliance": {
    "name_announced":      {"passed": true, "evidence": {"quote": "...", "speaker": "AGENT", "timestamp": "00:12"}, "note": null},
    "company_announced":   {"passed": true, "evidence": {"quote": "...", "speaker": "AGENT", "timestamp": "00:12"}, "note": null},
    "recording_disclosure":{"passed": false, "evidence": null, "note": "never stated"},
    "identity_verified":   {"passed": true, "evidence": {"quote": "...", "speaker": "AGENT", "timestamp": "00:42"}, "note": null},
    "identity_method": "date of birth",
    "resolution_provided": {"passed": true, "evidence": {"quote": "...", "speaker": "AGENT", "timestamp": "10:07"}, "note": null},
    "transfer_next_steps": {"passed": null, "evidence": null, "note": "no transfer"}
  },
  "quality": {
    "efficiency":         {"score": 4, "signals_present": ["first-call resolution"], "signals_absent": [], "evidence": [{"quote":"...","speaker":"AGENT","timestamp":"08:38"}], "requires_audio": false},
    "problem_resolution": {"score": 4, "signals_present": ["explained why"], "signals_absent": ["no explicit options A/B/C"], "evidence": [], "requires_audio": false},
    "clarity":            {"score": 3, "signals_present": [], "signals_absent": ["no recap at end"], "evidence": [], "requires_audio": false},
    "professionalism":    {"score": 4, "signals_present": ["complete sentences"], "signals_absent": [], "evidence": [], "requires_audio": true},
    "empathy":            {"score": 2, "signals_present": [], "signals_absent": ["no validation phrase"], "evidence": [], "requires_audio": true}
  },
  "escalation": {
    "red_flags": [],
    "customer_emotion_text": "calm",
    "risk_level": "none",
    "evidence": [],
    "requires_audio": true
  },
  "overall_summary": "2-3 sentence plain-language summary."
}"""


def strip_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s[3:]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    return s.strip()


def extract(call_id, provider="github", model=None, results_dir="results_channels"):
    manifest = load_manifest()
    meta = manifest[call_id]
    result_path = os.path.join(DATA, results_dir, meta["accent"], call_id + ".json")
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    packet, stats = build_packet(result, meta)

    authoritative_meta = CallMetadata(
        call_id=call_id, domain=result.get("domain", meta.get("domain", "unknown")),
        accent=meta.get("accent"), duration_seconds=stats["duration_s"],
        transcript_model=result.get("model"))

    user = f"{packet}\n\n{SKELETON}"
    system = SYSTEM
    last_err = None
    served = provider

    for attempt in (1, 2):
        t0 = time.time()
        if provider == "router":
            from router import chat_json_routed
            raw, served = chat_json_routed(system, user, return_meta=True)
        else:
            raw = chat_json(provider, system, user, model=model)
            served = model or provider
        dt = time.time() - t0
        try:
            data = json.loads(strip_fences(raw))
            data["rubric_version"] = RUBRIC_VERSION
            data["metadata"] = authoritative_meta.model_dump()
            evaluation = CallEvaluation.model_validate(data)
            return evaluation, dt, served
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            print(f"  [attempt {attempt}] validation failed ({type(e).__name__}); retrying...")
            user = (f"{packet}\n\n{SKELETON}\n\nYour previous output was invalid: "
                    f"{str(e)[:400]}. Return corrected JSON only.")
    raise RuntimeError(f"Extraction failed after 2 attempts: {last_err}")


def print_report(ev: CallEvaluation, dt, served=None):
    print("\n" + "=" * 68)
    tag = f"  [{dt:.1f}s | {served}]" if served else f"  [{dt:.1f}s]"
    print(f"  EVALUATION  {ev.metadata.call_id}  ({ev.metadata.domain}){tag}")
    print("=" * 68)
    print("\n  COMPLIANCE")
    c = ev.compliance
    for name, item in [("Name announced", c.name_announced), ("Company announced", c.company_announced),
                       ("Recording disclosure", c.recording_disclosure), ("Identity verified", c.identity_verified),
                       ("Resolution provided", c.resolution_provided), ("Transfer next-steps", c.transfer_next_steps)]:
        mark = {True: "PASS", False: "FAIL", None: "N/A "}[item.passed]
        ev_str = f'  "{item.evidence.quote[:60]}"' if item.evidence else (f"  ({item.note})" if item.note else "")
        print(f"    [{mark}] {name:<22}{ev_str}")
    if c.identity_method:
        print(f"           method: {c.identity_method}")
    print("\n  QUALITY (1-5, text-derived)")
    for name, q in [("Efficiency", ev.quality.efficiency), ("Problem resolution", ev.quality.problem_resolution),
                    ("Clarity", ev.quality.clarity), ("Professionalism", ev.quality.professionalism),
                    ("Empathy", ev.quality.empathy)]:
        aud = " (+audio needed)" if q.requires_audio else ""
        print(f"    {q.score}/5  {name:<20}{aud}")
        if q.signals_absent:
            print(f"          missing: {', '.join(q.signals_absent[:3])}")
    e = ev.escalation
    print(f"\n  ESCALATION: {e.risk_level.upper()}  | emotion(text): {e.customer_emotion_text}"
          f" | flags: {', '.join(e.red_flags) if e.red_flags else 'none'}")
    print(f"\n  SUMMARY: {ev.overall_summary}")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    from llm_client import list_providers
    ap.add_argument("--call_id", required=True)
    ap.add_argument("--provider", default="router",
                    choices=["router"] + list_providers())
    ap.add_argument("--model", default=None)
    ap.add_argument("--results_dir", default="results_channels")
    args = ap.parse_args()

    ev, dt, served = extract(args.call_id, args.provider, args.model, args.results_dir)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.call_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ev.model_dump(), f, indent=2)

    print_report(ev, dt, served)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
