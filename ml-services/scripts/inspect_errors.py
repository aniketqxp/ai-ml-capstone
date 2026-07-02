"""
Per-call aligned error inspector. Shows the actual ref->hyp word-level diffs so we
can SEE the failure mode (hallucination, boundary mashing, GT filler, real misses).
Low-CPU: pure evaluation, no transcription.

Usage:
  python inspect_errors.py <call_id> [--model small|medium|large|API|LLM] [--channel agent|customer|both]
"""
import os, sys, json, argparse, jiwer
from eval_common import normalise, normalise_raw

DATA      = r"d:\Desktop\ai-ml-capstone\data\na_testset"
MANIFEST  = os.path.join(DATA, "manifest.json")

DIRS = {
    "small":  ("results_channels",        "words"),
    "medium": ("results_channels_medium", "words"),
    "large":  ("results_channels_large",  "words"),
    "API":    ("results_api_vad",         "text"),
    "LLM":    ("results_channels_llm",    "text"),
}


def load(model, accent, cid):
    d, fmt = DIRS[model]
    path = os.path.join(DATA, d, accent, cid + ".json")
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    if fmt == "words":
        return (" ".join(w["word"] for w in j["agent"]),
                " ".join(w["word"] for w in j["customer"]))
    return j["agent_text"], j["customer_text"]


def show_diff(ref, hyp, label):
    r, h = normalise(ref), normalise(hyp)
    out = jiwer.process_words(r, h)
    al = out.alignments[0]
    refs, hyps = out.references[0], out.hypotheses[0]
    nS = sum(1 for c in al if c.type == "substitute")
    nD = sum(1 for c in al if c.type == "delete")
    nI = sum(1 for c in al if c.type == "insert")
    n = len(refs)
    acc = (1 - (nS+nD+ (sum(c.hyp_end_idx-c.hyp_start_idx for c in al if c.type=='insert')))/n)*100 if n else 100
    print(f"\n  === {label} ===  acc {acc:.1f}%  | S={nS} D={nD} I={nI}  ({n} ref words)")
    for c in al:
        if c.type == "equal":
            continue
        rspan = " ".join(refs[c.ref_start_idx:c.ref_end_idx])
        hspan = " ".join(hyps[c.hyp_start_idx:c.hyp_end_idx])
        if c.type == "substitute":
            print(f"    SUB  ref[{rspan}]  ->  hyp[{hspan}]")
        elif c.type == "delete":
            print(f"    DEL  ref[{rspan}]   (model missed)")
        elif c.type == "insert":
            print(f"    INS  hyp[{hspan}]   (model added)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call_id")
    ap.add_argument("--model", default="small")
    ap.add_argument("--channel", default="both", choices=["agent", "customer", "both"])
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {m["call_id"]: m for m in json.load(f)}
    m = manifest[args.call_id]
    accent = m["accent"]
    ag, cu = load(args.model, accent, args.call_id)

    print(f"Call {args.call_id}  ({accent} {m['domain']})  model={args.model}")
    if args.channel in ("agent", "both"):
        show_diff(m["agent_transcript"], ag, "AGENT")
    if args.channel in ("customer", "both"):
        show_diff(m["customer_transcript"], cu, "CUSTOMER")


if __name__ == "__main__":
    main()
