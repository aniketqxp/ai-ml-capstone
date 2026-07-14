"""
Acoustic-text fusion (v0.4.0) -- combines the audio sentiment model's
per-sentence output with the text LLM's rubric scores.

Design: DETERMINISTIC POST-HOC FUSION, not prompt injection.
The LLM judges what text can prove; the acoustic model contributes what text
cannot see (tone, emotion, escalation intensity); a documented formula
combines them. This keeps the LLM's variance out of the acoustic signal and
makes every fused score reproducible and explainable.

Math
----
Valence map: Positive=+1, Neutral/Mixed=0, Negative=-1.

Quality dimension fusion (empathy, professionalism <- AGENT channel;
customer_satisfaction <- CUSTOMER channel, final third weighted 2x):
    acoustic_score a = 3 + 2 * weighted_mean(valence)        # maps [-1,1] -> [1,5]
    w_a = W_ACOUSTIC_CAP * coverage                          # coverage = processed/total
    fused = clamp(round((1 - w_a) * text + w_a * a), 1, 5)

Text stays the primary signal (cap 0.4) because the rubric's behaviours are
text-defined; acoustics modulate. Zero acoustic data -> w_a = 0 (text only).

Escalation fusion (CUSTOMER channel escalation_score trajectory):
    late_mean = mean(score over final third), peak = max(score)
    acoustic tier (v0.4.1, fitted on the 22-call batch -- see threshold
    comments): escalate if late_mean >= 0.50, review if late_mean >= 0.30,
    none otherwise. peak is recorded as context but does NOT set a tier.
    Merge (v0.4.1): tiers that agree merge deterministically ("agreement");
    tiers that disagree are left to the arbitrate LLM node ("disputed" ->
    resolved as "llm_arbitration"), because disagreement is where a fixed
    formula has no information to prefer one channel over the other.

Data layout: data/sentiment/{domain}/{call_id}.json          (model output)
             data/sentiment/{domain}/{call_id}_segments.json (the EXACT
             sentence segmentation the model ran against -- seq_ids are only
             meaningful against their own segmentation)
"""
import os
import json

SENTIMENT_ROOT = r"d:\Desktop\Main\Projects\ai-ml-capstone\data\sentiment"

W_ACOUSTIC_CAP = 0.4          # max acoustic weight at 100% coverage
LATE_FRACTION = 1 / 3         # "end of call" window for trajectory metrics
LATE_WEIGHT = 2.0             # extra weight on final-third customer valence

# Acoustic escalation tiers -- fitted on the 22-call batch (2026-07,
# batch_summary.json, n=21 calls with paired acoustic data):
#   late_mean: min 0.166, p25 0.219, median 0.227, p75 0.257, p90 0.281,
#              max 0.406 (single outlier; next highest 0.281).
#   -> review at 0.30 sits above the calm cluster's p90 with margin; only the
#      outlier call crosses it. escalate at 0.50 is deliberately outside the
#      observed range: nothing in this batch warrants an acoustic-only
#      escalate, and the bar should stay high until data demands lowering it.
#   peak: median 0.558, p90 0.696 -- even calm calls routinely spike past the
#      old 0.60 trigger, so peak has no discriminating power as a threshold.
#      It is kept as recorded context (arbitrator/investigator input) only.
ESC_THRESHOLDS = {
    "escalate": {"late_mean": 0.50},
    "review":   {"late_mean": 0.30},
}

VALENCE = {"Positive": 1.0, "Negative": -1.0, "Neutral": 0.0, "Mixed": 0.0}
RISK_ORDER = {"none": 0, "review": 1, "escalate": 2}

# which quality dimension is informed by which channel
FUSED_DIMS = {
    "professionalism":       {"channel": "AGENT", "late_weighted": False},
    "empathy":               {"channel": "AGENT", "late_weighted": False},
    "customer_satisfaction": {"channel": "CUSTOMER", "late_weighted": True},
}


def load_acoustic(call_id, domain):
    """
    Join the sentiment model output with the segmentation it ran against.
    Returns list of rows {seq_id, pos, speaker, valence, emotion,
    escalation_score, ok} or None if no acoustic data exists for this call.
    """
    d = os.path.join(SENTIMENT_ROOT, domain.lower())
    sent_path = os.path.join(d, f"{call_id}.json")
    seg_path = os.path.join(d, f"{call_id}_segments.json")
    if not (os.path.exists(sent_path) and os.path.exists(seg_path)):
        return None

    with open(sent_path, encoding="utf-8") as f:
        sentiment = json.load(f)
    with open(seg_path, encoding="utf-8") as f:
        segments = json.load(f)

    speakers = {s["seq_id"]: s["speaker"] for s in segments["sentences"]}
    n = max(speakers) or 1

    rows = []
    for s in sentiment["segments"]:
        sid = s["seq_id"]
        ok = s.get("processing_status") == "success" and s.get("sentiment")
        rows.append({
            "seq_id": sid,
            "pos": sid / n,                       # 0..1 position in call
            "speaker": speakers.get(sid),
            "valence": VALENCE.get(s.get("sentiment"), 0.0) if ok else None,
            "emotion": s.get("dominant_emotion"),
            "escalation_score": s.get("escalation_score"),
            "ok": bool(ok),
        })
    return rows


def _channel_valence(rows, channel, late_weighted):
    """(weighted mean valence, coverage) for one speaker channel."""
    chan = [r for r in rows if r["speaker"] == channel]
    if not chan:
        return None, 0.0
    ok = [r for r in chan if r["ok"]]
    coverage = len(ok) / len(chan)
    if not ok:
        return None, 0.0
    wsum = vsum = 0.0
    for r in ok:
        w = LATE_WEIGHT if (late_weighted and r["pos"] >= 1 - LATE_FRACTION) else 1.0
        wsum += w
        vsum += w * r["valence"]
    return vsum / wsum, coverage


def _acoustic_escalation(rows):
    """Acoustic risk tier from the customer escalation_score trajectory."""
    cust = [r for r in rows
            if r["speaker"] == "CUSTOMER" and r["ok"]
            and r["escalation_score"] is not None]
    if not cust:
        return None, None, None
    late = [r["escalation_score"] for r in cust if r["pos"] >= 1 - LATE_FRACTION]
    late_mean = sum(late) / len(late) if late else 0.0
    peak = max(r["escalation_score"] for r in cust)

    for tier in ("escalate", "review"):
        if late_mean >= ESC_THRESHOLDS[tier]["late_mean"]:
            return tier, round(late_mean, 3), round(peak, 3)
    return "none", round(late_mean, 3), round(peak, 3)


def trajectory(rows, buckets=48):
    """
    Downsample the per-sentence acoustic signal into fixed position buckets
    for the frontend timeline: [{p, esc, cust_val, agent_val}, ...].
    p = bucket midpoint (0..1); esc = mean customer escalation_score;
    *_val = mean valence per channel. Fields are null where a bucket has no
    processed sentences for that channel.
    """
    out = []
    for b in range(buckets):
        lo, hi = b / buckets, (b + 1) / buckets
        cell = [r for r in rows if r["ok"] and lo <= r["pos"] < hi]
        esc = [r["escalation_score"] for r in cell
               if r["speaker"] == "CUSTOMER" and r["escalation_score"] is not None]
        cv = [r["valence"] for r in cell
              if r["speaker"] == "CUSTOMER" and r["valence"] is not None]
        av = [r["valence"] for r in cell
              if r["speaker"] == "AGENT" and r["valence"] is not None]
        out.append({
            "p": round((lo + hi) / 2, 4),
            "esc": round(sum(esc) / len(esc), 3) if esc else None,
            "cust_val": round(sum(cv) / len(cv), 3) if cv else None,
            "agent_val": round(sum(av) / len(av), 3) if av else None,
        })
    return out


def fuse_evaluation(ev, rows):
    """
    Mutate an evaluation dict in place: fuse acoustic signal into the three
    audio-informed quality dimensions and the escalation risk level.
    Every dimension gets a `hybrid` provenance block (method "text_only"
    when acoustics did not contribute) so the UI can explain each score.
    Returns a short summary dict for node_meta.
    """
    fused_dims = []

    for name, dim in (ev.get("quality") or {}).items():
        if not isinstance(dim, dict) or "score" not in dim:
            continue
        text_score = dim["score"]
        cfg = FUSED_DIMS.get(name)
        vbar, coverage = (_channel_valence(rows, cfg["channel"], cfg["late_weighted"])
                          if (cfg and rows) else (None, 0.0))
        if cfg and vbar is not None:
            a = 3 + 2 * vbar
            w_a = W_ACOUSTIC_CAP * coverage
            fused = max(1, min(5, round((1 - w_a) * text_score + w_a * a)))
            dim["score"] = fused
            dim["hybrid"] = {
                "text_score": text_score,
                "acoustic_score": round(a, 2),
                "text_weight": round(1 - w_a, 2),
                "acoustic_weight": round(w_a, 2),
                "coverage": round(coverage, 2),
                "channel": cfg["channel"],
                "method": "weighted_mean",
            }
            fused_dims.append(name)
        else:
            dim["hybrid"] = {
                "text_score": text_score,
                "acoustic_score": None,
                "text_weight": 1.0,
                "acoustic_weight": 0.0,
                "coverage": None,
                "channel": cfg["channel"] if cfg else None,
                "method": "text_only",
            }

    # Escalation merge (v0.4.1): agreement is trivially deterministic;
    # disagreement carries no information about which channel to trust, so it
    # is NOT resolved here -- the graph routes disputed calls to the arbitrate
    # LLM node, which weighs both signals against the transcript. Until (or
    # unless) arbitration runs, the text tier stands.
    disputed = False
    esc = ev.get("escalation") or {}
    text_risk = esc.get("risk_level")
    a_risk, late_mean, peak = _acoustic_escalation(rows) if rows else (None, None, None)
    if text_risk and a_risk is not None:
        disputed = a_risk != text_risk
        esc["hybrid"] = {
            "text_risk": text_risk,
            "acoustic_risk": a_risk,
            "late_mean_escalation": late_mean,
            "peak_escalation": peak,
            "method": "agreement" if not disputed else "text_only",
            "arbitration_rationale": None,
        }
    elif text_risk:
        esc["hybrid"] = {
            "text_risk": text_risk,
            "acoustic_risk": None,
            "late_mean_escalation": None,
            "peak_escalation": None,
            "method": "text_only",
            "arbitration_rationale": None,
        }

    # downsampled acoustic timeline for the frontend Agent-findings lanes
    if rows:
        ev["_acoustic"] = {"trajectory": trajectory(rows),
                           "n_sentences": len(rows)}

    return {
        "acoustic_available": bool(rows),
        "fused_dims": fused_dims,
        "acoustic_risk": a_risk,
        "disputed": disputed,
    }
