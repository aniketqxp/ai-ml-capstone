"""
Batch WER evaluation across banking, health, and telecom calls.

Reads reference transcripts from data/na_testset/manifest.json and
hypothesis word-level JSONs from data/na_testset/results/.

Reports per-call accuracy (raw + normalised) and domain-level aggregates.
Writes a machine-readable results file to data/wer_results.json.

Usage:
    python evaluate_wer_batch.py [--domains banking health telecom]
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import jiwer

# make eval_common importable from the scripts directory
sys.path.insert(0, str(Path(__file__).parent))
from eval_common import normalise, normalise_raw

MANIFEST  = Path(__file__).parent.parent.parent / "data" / "na_testset" / "manifest.json"
RESULTS   = Path(__file__).parent.parent.parent / "data" / "na_testset" / "results"
OUT_JSON  = Path(__file__).parent.parent.parent / "data" / "wer_results.json"
DOCS_OUT  = Path(__file__).parent.parent.parent / "docs" / "wer_accuracy_report.md"

TARGET_DOMAINS = {"banking", "health", "telecom"}


def wer_score(ref: str, hyp: str):
    out = jiwer.process_words(ref, hyp)
    return out.wer, len(ref.split())


def evaluate_call(meta: dict, results_dir: Path) -> dict | None:
    call_id = meta["call_id"]
    accent  = meta["accent"]

    # find the result JSON
    result_path = results_dir / accent / f"{call_id}.json"
    if not result_path.exists():
        return None

    with open(result_path, encoding="utf-8") as f:
        hyp_data = json.load(f)

    hyp_agent    = " ".join(w["word"] for w in hyp_data.get("agent", []))
    hyp_customer = " ".join(w["word"] for w in hyp_data.get("customer", []))
    ref_agent    = meta.get("agent_transcript", "")
    ref_customer = meta.get("customer_transcript", "")

    results = {}
    for label, norm_fn in [("raw", normalise_raw), ("normalised", normalise)]:
        wa, na = wer_score(norm_fn(ref_agent),    norm_fn(hyp_agent))
        wc, nc = wer_score(norm_fn(ref_customer), norm_fn(hyp_customer))
        total_words = na + nc
        overall_wer = (wa * na + wc * nc) / total_words if total_words else 0
        results[label] = {
            "agent_wer":    round(wa, 4),
            "customer_wer": round(wc, 4),
            "overall_wer":  round(overall_wer, 4),
            "overall_acc":  round(1 - overall_wer, 4),
            "ref_words":    total_words,
        }

    return {
        "call_id": call_id,
        "domain":  meta.get("domain", "unknown"),
        "accent":  accent,
        **{f"{l}_{k}": v for l, d in results.items() for k, v in d.items()},
        "raw":        results["raw"],
        "normalised": results["normalised"],
    }


def aggregate(rows: list[dict], key: str = "normalised") -> dict:
    total_words = sum(r[key]["ref_words"] for r in rows)
    weighted_wer = (
        sum(r[key]["overall_wer"] * r[key]["ref_words"] for r in rows) / total_words
        if total_words else 0
    )
    return {
        "n_calls":      len(rows),
        "total_words":  total_words,
        "overall_wer":  round(weighted_wer, 4),
        "overall_acc":  round(1 - weighted_wer, 4),
        "pass_90":      (1 - weighted_wer) >= 0.90,
    }


def write_docs(all_rows: list[dict], domain_aggs: dict, overall: dict):
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# WER Accuracy Report",
        f"*Generated: {ts} | Model: faster-whisper small.en/int8 + stereo channel attribution*",
        "",
        "## Overall",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Calls evaluated | {overall['n_calls']} |",
        f"| Total reference words | {overall['total_words']:,} |",
        f"| Normalised WER | {overall['overall_wer']*100:.1f}% |",
        f"| **Normalised accuracy** | **{overall['overall_acc']*100:.1f}%** |",
        f"| Target (>=90%) | {'PASS' if overall['pass_90'] else 'NEEDS WORK'} |",
        "",
        "## By Domain",
        "| Domain | Calls | Accuracy (normalised) | Pass? |",
        "|--------|-------|-----------------------|-------|",
    ]
    for domain, agg in sorted(domain_aggs.items()):
        status = "yes" if agg["pass_90"] else "no"
        lines.append(
            f"| {domain} | {agg['n_calls']} | {agg['overall_acc']*100:.1f}% | {status} |"
        )

    lines += [
        "",
        "## Per-Call Results (normalised accuracy)",
        "| Call ID | Domain | Acc (norm) | Acc (raw) | Agent WER | Customer WER |",
        "|---------|--------|------------|-----------|-----------|--------------|",
    ]
    for r in sorted(all_rows, key=lambda x: (x["domain"], x["call_id"])):
        n = r["normalised"]
        rw = r["raw"]
        lines.append(
            f"| {r['call_id']} | {r['domain']} "
            f"| {n['overall_acc']*100:.1f}% "
            f"| {rw['overall_acc']*100:.1f}% "
            f"| {n['agent_wer']*100:.1f}% "
            f"| {n['customer_wer']*100:.1f}% |"
        )

    lines += [
        "",
        "## Notes",
        "- **Raw WER**: lowercase + strip punctuation only (conservative, no number normalisation).",
        "- **Normalised WER**: Whisper `EnglishTextNormalizer` — canonicalises numbers,",
        "  contractions, British/American spelling, and strips disfluency annotations `(uh)`, `(um)`, `word~`.",
        "- Accuracy = 1 − WER. Target is ≥ 90% normalised accuracy.",
        "- Weighted by reference word count across calls within each aggregate.",
    ]

    DOCS_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDocumentation written -> {DOCS_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+",
                    default=sorted(TARGET_DOMAINS),
                    choices=sorted(TARGET_DOMAINS))
    args = ap.parse_args()
    domains = set(args.domains)

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    rows = []
    skipped = []
    for meta in manifest:
        if meta.get("domain", "").lower() not in domains:
            continue
        result = evaluate_call(meta, RESULTS)
        if result is None:
            skipped.append(meta["call_id"])
            continue
        rows.append(result)

        n = result["normalised"]
        r = result["raw"]
        status = "PASS" if n["overall_acc"] >= 0.90 else "----"
        print(f"  [{status}] {result['call_id']:<42} "
              f"norm {n['overall_acc']*100:5.1f}%  raw {r['overall_acc']*100:5.1f}%"
              f"  (A:{n['agent_wer']*100:.1f}% C:{n['customer_wer']*100:.1f}%)")

    if skipped:
        print(f"\n  Skipped (no result file): {', '.join(skipped)}")

    print()
    domain_aggs = {}
    for domain in sorted(domains):
        domain_rows = [r for r in rows if r["domain"] == domain]
        if not domain_rows:
            continue
        agg = aggregate(domain_rows)
        domain_aggs[domain] = agg
        print(f"  {domain:<10}  {agg['n_calls']} calls  "
              f"acc {agg['overall_acc']*100:.1f}%  "
              f"({'PASS' if agg['pass_90'] else 'NEEDS WORK'})")

    overall = aggregate(rows)
    print(f"\n  OVERALL    {overall['n_calls']} calls  "
          f"acc {overall['overall_acc']*100:.1f}%  "
          f"({'PASS' if overall['pass_90'] else 'NEEDS WORK'})")

    # save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated": datetime.now().isoformat(),
        "model": "faster-whisper small.en/int8 + stereo channel attribution",
        "domains": sorted(domains),
        "overall": overall,
        "by_domain": domain_aggs,
        "calls": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults JSON -> {OUT_JSON}")

    write_docs(rows, domain_aggs, overall)


if __name__ == "__main__":
    main()
