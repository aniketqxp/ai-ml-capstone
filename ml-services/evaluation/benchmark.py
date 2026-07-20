"""
Model benchmark for call QA extraction.

Runs the same structured extraction task across candidate models on two
contrasting calls, then auto-scores each output on four criteria:

  1. Schema validity   -- does Pydantic accept it?
  2. Evidence fidelity -- are quoted strings verbatim substrings of the transcript?
  3. Score spread      -- does the model discriminate between the two calls?
  4. Conservative bias -- does it avoid PASS without explicit evidence?

Produces a ranked comparison table so you can make a principled model
selection before wiring up routing.

Usage:
  python benchmark.py
  python benchmark.py --calls en_CA_Banking_1592237 en_US_General_Health_1587175
  python benchmark.py --providers github mistral cohere
"""

import os, json, time, argparse, re
from typing import Optional
from pydantic import ValidationError

import paths
from env_util import load_env
from assemble import load_manifest, build_packet
from rubric import CallEvaluation, RUBRIC_VERSION
from llm_client import chat_json
from extract import SYSTEM, SKELETON, strip_fences

load_env()

DATA = str(paths.NA_TESTSET)

# ── Candidate models (provider, model_override or None for default) ───────────
CANDIDATES = [
    ("github",     None),                           # GPT-4o-mini
    ("gemini",     "gemini-2.0-flash"),
    ("mistral",    "mistral-small-latest"),
    ("sambanova",  "Meta-Llama-3.3-70B-Instruct"),
    ("sambanova",  "DeepSeek-V3.1"),
    ("cohere",     "command-r7b-12-2024"),
]

DEFAULT_CALLS = [
    "en_CA_Banking_1592237",       # clean, high-scoring banking call
    "en_CA_Banking_1588683",       # weaker banking call
]


# ── Auto-scoring helpers ──────────────────────────────────────────────────────

def check_schema(raw: str) -> tuple[bool, Optional[str]]:
    """Try to parse + validate. Returns (valid, error_snippet)."""
    try:
        data = json.loads(raw)
        data["rubric_version"] = RUBRIC_VERSION
        data["metadata"] = {"call_id": "x", "domain": "x"}
        CallEvaluation.model_validate(data)
        return True, None
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        return False, str(e)[:120]


def check_evidence_fidelity(raw: str, packet: str) -> tuple[int, int]:
    """Count evidence quotes that are verbatim substrings of the transcript.
    Returns (n_verbatim, n_total)."""
    try:
        data = json.loads(raw)
    except Exception:
        return 0, 0

    # collect all quote fields recursively
    quotes = []
    def harvest(obj):
        if isinstance(obj, dict):
            if "quote" in obj and isinstance(obj["quote"], str):
                quotes.append(obj["quote"])
            for v in obj.values():
                harvest(v)
        elif isinstance(obj, list):
            for item in obj:
                harvest(item)
    harvest(data)

    if not quotes:
        return 0, 0

    # normalise packet to lowercase for matching (avoid case mismatches)
    packet_lower = packet.lower()
    n_verbatim = sum(1 for q in quotes if q.lower().strip() in packet_lower)
    return n_verbatim, len(quotes)


def check_conservative_bias(raw: str) -> tuple[int, int]:
    """Count compliance items marked PASS and how many have evidence attached.
    Returns (n_pass_with_evidence, n_pass_total)."""
    try:
        data = json.loads(raw)
    except Exception:
        return 0, 0

    compliance = data.get("compliance", {})
    n_pass = n_with_ev = 0
    for key, item in compliance.items():
        if not isinstance(item, dict):
            continue
        if item.get("passed") is True:
            n_pass += 1
            if item.get("evidence") and item["evidence"].get("quote"):
                n_with_ev += 1
    return n_with_ev, n_pass


def quality_scores(raw: str) -> list[int]:
    """Extract the 5 quality scores as a list."""
    try:
        data = json.loads(raw)
        q = data.get("quality", {})
        return [q.get(k, {}).get("score", 0)
                for k in ("efficiency", "problem_resolution", "clarity",
                           "professionalism", "empathy")]
    except Exception:
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(calls: list[str], providers: Optional[list[str]]):
    manifest = load_manifest()

    candidates = CANDIDATES
    if providers:
        candidates = [(p, m) for p, m in CANDIDATES if p in providers]

    # pre-build packets
    packets = {}
    for cid in calls:
        meta = manifest[cid]
        result_path = os.path.join(
            DATA, "results_channels", meta["accent"], cid + ".json")
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        packet, _ = build_packet(result, meta)
        packets[cid] = packet

    # results[provider_model][call_id] = {raw, scores_dict, timing}
    rows = []

    print(f"\nBenchmarking {len(candidates)} models × {len(calls)} calls\n")

    for provider, model_override in candidates:
        label = f"{provider}/{model_override or 'default'}"
        row = {"label": label, "provider": provider,
               "model": model_override, "calls": {}}

        for cid in calls:
            packet = packets[cid]
            user = f"{packet}\n\n{SKELETON}"
            t0 = time.time()
            try:
                raw = chat_json(provider, SYSTEM, user,
                                model=model_override, max_tokens=4000)
                elapsed = time.time() - t0

                valid, err = check_schema(raw)
                n_verb, n_total = check_evidence_fidelity(raw, packet)
                n_pass_ev, n_pass = check_conservative_bias(raw)
                scores = quality_scores(raw)

                row["calls"][cid] = {
                    "ok":       valid,
                    "elapsed":  elapsed,
                    "schema":   valid,
                    "fidelity": (n_verb, n_total),
                    "conserv":  (n_pass_ev, n_pass),
                    "scores":   scores,
                    "err":      err,
                    "raw":      raw,
                }
                status = "OK " if valid else "ERR"
                fid = f"{n_verb}/{n_total}" if n_total else "--"
                con = f"{n_pass_ev}/{n_pass}" if n_pass else "--"
                sc  = ",".join(str(s) for s in scores) if scores else "--"
                print(f"  [{status}] {label:<38} {cid[-10:]:<14} "
                      f"{elapsed:5.1f}s  fidelity={fid}  conserv={con}  "
                      f"scores=[{sc}]")

            except Exception as e:
                elapsed = time.time() - t0
                row["calls"][cid] = {
                    "ok": False, "elapsed": elapsed,
                    "err": str(e)[:120], "raw": ""}
                print(f"  [FAIL] {label:<38} {cid[-10:]:<14} "
                      f"{elapsed:5.1f}s  {str(e)[:80]}")

        rows.append(row)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUMMARY  ({len(calls)} calls)")
    print(f"{'='*90}")
    print(f"  {'Model':<38} {'Valid':>5} {'Fidelity':>9} {'Conserv':>8} "
          f"{'Spread':>7} {'Avg ms':>8}")
    print(f"  {'-'*85}")

    scored = []
    for row in rows:
        valid_calls = [v for v in row["calls"].values() if v.get("ok")]
        n_valid = len(valid_calls)

        # fidelity: avg across calls
        fid_nums = [(v["fidelity"][0], v["fidelity"][1])
                    for v in valid_calls if v.get("fidelity", (0,0))[1] > 0]
        fid_str = (f"{sum(x[0] for x in fid_nums)}/{sum(x[1] for x in fid_nums)}"
                   if fid_nums else "--")

        # conservatism: evidence on PASS items
        con_nums = [(v["conserv"][0], v["conserv"][1])
                    for v in valid_calls if v.get("conserv", (0,0))[1] > 0]
        con_str = (f"{sum(x[0] for x in con_nums)}/{sum(x[1] for x in con_nums)}"
                   if con_nums else "--")

        # score spread: std-dev of scores across calls (higher = more discriminating)
        all_scores = [s for v in valid_calls for s in v.get("scores", [])]
        if len(all_scores) >= 2:
            mean = sum(all_scores) / len(all_scores)
            spread = (sum((s - mean)**2 for s in all_scores) / len(all_scores)) ** 0.5
            spread_str = f"{spread:.2f}"
        else:
            spread = 0.0
            spread_str = "--"

        avg_ms = (sum(v["elapsed"] for v in row["calls"].values()) /
                  len(row["calls"]) * 1000) if row["calls"] else 0

        print(f"  {row['label']:<38} {n_valid}/{len(calls):>3}  "
              f"{fid_str:>9}  {con_str:>8}  {spread_str:>7}  {avg_ms:>7.0f}ms")

        scored.append((row["label"], n_valid, fid_nums, con_nums, spread, avg_ms))

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n  COLUMN GUIDE:")
    print(f"  Valid    = schema-valid outputs / total calls")
    print(f"  Fidelity = verbatim quotes found in transcript / total quotes cited")
    print(f"  Conserv  = PASS items with supporting evidence / total PASS items")
    print(f"  Spread   = score std-dev across calls (higher = more discriminating)")
    print(f"  Avg ms   = average latency per call")
    print(f"\n  Best fidelity  = quotes least likely to be hallucinated")
    print(f"  Best conserv   = least likely to grant unearned compliance passes")
    print(f"  Best spread    = actually differentiates call quality")
    print(f"{'='*90}")

    # save raw results
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        # strip raw transcripts from saved output to keep file small
        clean = []
        for row in rows:
            r = dict(row)
            r["calls"] = {cid: {k: v for k, v in cv.items() if k != "raw"}
                          for cid, cv in row["calls"].items()}
            clean.append(r)
        json.dump(clean, f, indent=2)
    print(f"\n  Full results saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", nargs="+", default=DEFAULT_CALLS)
    ap.add_argument("--providers", nargs="+", default=None,
                    help="filter to these providers only")
    args = ap.parse_args()
    run_benchmark(args.calls, args.providers)


if __name__ == "__main__":
    main()
