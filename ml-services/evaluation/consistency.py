"""
Consistency harness for the graph evaluation pipeline.

A scoring system that produces different results on the same input across
runs is not dependable. This harness runs the SAME call through the graph N
times and measures agreement:

  - compliance : per-check agreement rate (fraction of runs matching the
                 modal pass/fail/null verdict)
  - quality    : per-dimension score min / max / range / mean / stdev
  - escalation : risk_level + emotion agreement rate
  - anchoring  : evidence anchor rate per run

Usage:
  python consistency.py --call_id en_CA_Banking_1586889 --runs 5
"""
import os
import json
import time
import argparse
import statistics
from collections import Counter

from graph import build_graph

COMPLIANCE_KEYS = [
    "name_announced", "company_announced", "recording_disclosure",
    "identity_verified", "resolution_provided", "transfer_next_steps",
]
QUALITY_KEYS = [
    "efficiency", "problem_resolution", "clarity",
    "professionalism", "empathy", "customer_satisfaction",
]


def agreement(values):
    """Fraction of values matching the mode. 1.0 = perfect agreement."""
    if not values:
        return None
    most_common = Counter(values).most_common(1)[0][1]
    return most_common / len(values)


def run_sweep(call_id, runs, results_dir):
    graph = build_graph()
    evaluations = []

    for i in range(1, runs + 1):
        print(f"\n--- run {i}/{runs} " + "-" * 50)
        t0 = time.time()
        final = graph.invoke({"call_id": call_id, "results_dir": results_dir})
        dt = time.time() - t0
        ev = final["evaluation"]
        evaluations.append(ev)
        stats = ev.get("_anchor_stats", {})
        print(f"--- run {i} done in {dt:.1f}s "
              f"(anchored {stats.get('anchored')}/{stats.get('total')})")

    return evaluations


def analyze(evaluations):
    report = {"n_runs": len(evaluations), "compliance": {}, "quality": {},
              "escalation": {}, "anchoring": {}}

    # compliance agreement per check
    for key in COMPLIANCE_KEYS:
        verdicts = [str(ev["compliance"][key]["passed"]) for ev in evaluations]
        report["compliance"][key] = {
            "verdicts": verdicts,
            "agreement": agreement(verdicts),
        }

    # quality score variance per dimension
    for key in QUALITY_KEYS:
        scores = [ev["quality"][key]["score"] for ev in evaluations
                  if ev["quality"].get(key)]
        if not scores:
            continue
        report["quality"][key] = {
            "scores": scores,
            "min": min(scores),
            "max": max(scores),
            "range": max(scores) - min(scores),
            "mean": round(statistics.mean(scores), 2),
            "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
        }

    # escalation agreement
    risks = [ev["escalation"]["risk_level"] for ev in evaluations]
    emotions = [ev["escalation"]["customer_emotion_text"] for ev in evaluations]
    report["escalation"] = {
        "risk_levels": risks,
        "risk_agreement": agreement(risks),
        "emotions": emotions,
        "emotion_agreement": agreement(emotions),
    }

    # anchor rates
    rates = []
    for ev in evaluations:
        s = ev.get("_anchor_stats", {})
        if s.get("total"):
            rates.append(round(s["anchored"] / s["total"], 3))
    report["anchoring"] = {"rates": rates}

    return report


def print_report(report, call_id):
    n = report["n_runs"]
    print("\n" + "=" * 68)
    print(f"  CONSISTENCY REPORT  {call_id}  ({n} runs)")
    print("=" * 68)

    print("\n  COMPLIANCE (agreement with modal verdict)")
    worst_c = 1.0
    for key, r in report["compliance"].items():
        a = r["agreement"]
        worst_c = min(worst_c, a)
        flag = "" if a == 1.0 else f"  <-- UNSTABLE {r['verdicts']}"
        print(f"    {a * 100:5.1f}%  {key}{flag}")

    print("\n  QUALITY (score spread across runs)")
    worst_range = 0
    for key, r in report["quality"].items():
        worst_range = max(worst_range, r["range"])
        flag = "" if r["range"] <= 1 else "  <-- UNSTABLE"
        print(f"    {key:<24} scores={r['scores']}  "
              f"range={r['range']}  stdev={r['stdev']}{flag}")

    e = report["escalation"]
    print(f"\n  ESCALATION  risk agreement: {e['risk_agreement'] * 100:.1f}% "
          f"{e['risk_levels']}")
    print(f"              emotion agreement: {e['emotion_agreement'] * 100:.1f}% "
          f"{e['emotions']}")

    rates = report["anchoring"]["rates"]
    if rates:
        print(f"\n  ANCHOR RATES per run: {rates}")

    print("\n  VERDICT: ", end="")
    if worst_c == 1.0 and worst_range <= 1 and e["risk_agreement"] == 1.0:
        print("stable (all compliance unanimous, quality range <= 1, "
              "risk unanimous)")
    else:
        print("instability detected -- see UNSTABLE flags above")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser(
        description="Run the graph N times on one call, measure agreement")
    ap.add_argument("--call_id", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--results_dir", default="results_channels")
    args = ap.parse_args()

    evaluations = run_sweep(args.call_id, args.runs, args.results_dir)
    report = analyze(evaluations)
    print_report(report, args.call_id)

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results", f"{args.call_id}_consistency.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
