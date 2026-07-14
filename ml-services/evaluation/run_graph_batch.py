"""
Batch-run the graph evaluation over every call that has audio sentiment data.

Purpose (v0.4.0 validation):
  1. Ground the fusion constants — collect text-vs-acoustic score pairs and
     escalation trajectories across calls so thresholds can be fitted to the
     observed distributions instead of asserted.
  2. Probe the workflow track for leakage/leniency — if expected-step pass
     rates are ~100% on every call including badly-handled ones, the checklist
     generation or the audit is compromised.

Usage:  python run_graph_batch.py [--results_dir results] [--only banking]
Output: results/{call_id}_graph.json per call + results/batch_summary.json
"""
import os
import json
import time
import argparse
import traceback

from graph import build_graph
from rubric import RUBRIC_VERSION_GRAPH

SENTIMENT_ROOT = r"d:\Desktop\Main\Projects\ai-ml-capstone\data\sentiment"


def calls_with_sentiment(only=None):
    out = []
    for domain in sorted(os.listdir(SENTIMENT_ROOT)):
        if only and domain != only:
            continue
        d = os.path.join(SENTIMENT_ROOT, domain)
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and not f.endswith("_segments.json"):
                out.append(f[:-5])
    return out


def summarize(ev):
    """Extract the batch-level facts we want to analyze."""
    q = ev.get("quality", {})
    dims = {}
    for name, d in q.items():
        if isinstance(d, dict) and "score" in d:
            h = d.get("hybrid") or {}
            dims[name] = {"fused": d["score"],
                          "text": h.get("text_score", d["score"]),
                          "acoustic": h.get("acoustic_score"),
                          "coverage": h.get("coverage")}
    e = ev.get("escalation", {})
    eh = e.get("hybrid") or {}
    wf = ev.get("workflow") or {}
    steps = wf.get("expected_steps", [])
    return {
        "quality": dims,
        "risk": e.get("risk_level"),
        "risk_text": eh.get("text_risk"),
        "risk_acoustic": eh.get("acoustic_risk"),
        "esc_late_mean": eh.get("late_mean_escalation"),
        "esc_peak": eh.get("peak_escalation"),
        "esc_method": eh.get("method"),
        "emotion": e.get("customer_emotion_text"),
        "red_flags": e.get("red_flags", []),
        "wf_subject": wf.get("subject"),
        "wf_total": len(steps),
        "wf_met": sum(1 for s in steps if s.get("met") is True),
        "wf_missed": sum(1 for s in steps if s.get("met") is False),
        "anchor": ev.get("_anchor_stats"),
        "investigated": ev.get("investigation") is not None,
        "wall_clock": (ev.get("_pipeline") or {}).get("wall_clock"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results",
                    help="transcription dir under na_testset (default: results "
                         "— the flattened backend format covers all 22 calls)")
    ap.add_argument("--only", default=None, help="restrict to one domain")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    calls = calls_with_sentiment(args.only)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    graph = build_graph()
    summary, failures = {}, {}
    t_start = time.time()

    print(f"Batch: {len(calls)} calls, rubric {RUBRIC_VERSION_GRAPH}, "
          f"results_dir={args.results_dir}\n" + "=" * 68)

    for i, call_id in enumerate(calls, 1):
        out_path = os.path.join(out_dir, f"{call_id}_graph.json")
        if args.skip_existing and os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("rubric_version") == RUBRIC_VERSION_GRAPH:
                summary[call_id] = summarize(prev)
                print(f"[{i:2}/{len(calls)}] {call_id} — cached")
                continue

        print(f"[{i:2}/{len(calls)}] {call_id}")
        t0 = time.time()
        try:
            final = graph.invoke({"call_id": call_id,
                                  "results_dir": args.results_dir})
            ev = final["evaluation"]
            ev["_pipeline"] = {
                "nodes": final.get("node_meta", {}),
                "anchor_passes": final.get("anchor_attempts", 1),
                "wall_clock": round(time.time() - t0, 1),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2)
            summary[call_id] = summarize(ev)
            s = summary[call_id]
            print(f"    ok in {s['wall_clock']}s | risk={s['risk']} "
                  f"(text={s['risk_text']}, ac={s['risk_acoustic']}) | "
                  f"wf {s['wf_met']}/{s['wf_total']} met, {s['wf_missed']} missed")
        except Exception as e:
            failures[call_id] = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {failures[call_id]}")
            traceback.print_exc(limit=2)

    with open(os.path.join(out_dir, "batch_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"rubric_version": RUBRIC_VERSION_GRAPH,
                   "results_dir": args.results_dir,
                   "wall_clock_total": round(time.time() - t_start, 1),
                   "calls": summary, "failures": failures}, f, indent=2)

    print("=" * 68)
    print(f"Done: {len(summary)} ok, {len(failures)} failed, "
          f"{round(time.time() - t_start, 1)}s total")
    print(f"Summary -> {os.path.join(out_dir, 'batch_summary.json')}")


if __name__ == "__main__":
    main()
