# Evaluation Pipeline — Development Journal

*Banking Call Compliance Automation | Capstone Project*
*AI/ML Engineering | 2025–2026*

---

## Overview

[`transcription_development.md`](transcription_development.md) ends at Stage 8 (2026-06-16): per-channel transcription locked at 90.2% normalized accuracy on the 22-call banking/health/telecom evaluation set, and 3,101 sentence-level segments written for the downstream emotion model. This document picks up exactly there and covers everything built on top of that transcript: the LLM rubric evaluation graph, its fusion with the acoustic sentiment model, and the React frontend that renders the result.

**A note on how this document was assembled.** Unlike the transcription journal, this arc was not logged stage-by-stage as it happened. It is reconstructed here from `rubric.py`'s own internal version history (`RUBRIC_VERSION_GRAPH`), the current structure of `ml-services/evaluation/`, commit history, and the existing `docs/weekly_progress_eval_fusion.tex` writeup. Where a single commit bundles several logically distinct milestones — which happens more than once below — the stage boundaries follow the rubric's version markers rather than one-stage-per-commit, and that bundling is called out explicitly rather than smoothed over.

---

## Stage 1 — Monolithic Baseline (`extract.py`, rubric v0.1.0) — 2026-06-09

The first evaluation pass (`bea7c8f`) sent a single LLM prompt covering three unrelated judgment types at once: a 6-item compliance checklist, 5 quality dimensions (1–5 scale), and escalation risk. `rubric.py` formalized the domain expert's QA rubric as a Pydantic v2 schema from the start, with two decisions that shaped everything downstream:

- **Modality split.** Every rubric field is tagged text-derivable or audio/paralinguistic. Audio-dependent judgments (tone, pace, "sounds satisfied") are represented in the schema but flagged `requires_audio=True` and left null by the text model — explicitly reserved for a later acoustic fusion stage rather than guessed from text.
- **Evidence-or-it-didn't-happen.** Every leaf judgment carries an `Evidence` object (verbatim quote + speaker + timestamp), so no compliance pass or quality score is ungrounded.

The same day, `segment.py` (LLM chaptering) and the first React player (`a518651`) landed together — transcription, evaluation, and a working visualization all existed as one vertical slice on a single demo call before any batch evaluation was attempted.

---

## Stage 2 — LangGraph Decomposition (v0.2.0) — 2026-06-29

`414f7ff` replaced the monolithic prompt with a LangGraph `StateGraph` fanning out to four parallel LLM nodes (compliance, quality, escalation, chapters), each with its own focused prompt and its own Pydantic fragment to validate. A malformed compliance response no longer invalidates the quality or escalation extraction for the same call — each node retries independently.

`customer_satisfaction` was added as a sixth quality dimension, `Optional` with `default=None` so v0.1.0 evaluation JSON already on disk keeps validating unchanged. Wired into the React player the same day (`9252f90`): the panel guards against null for backward compatibility and prints the rubric version in its footer.

---

## Stage 3 — Evidence Anchoring, Conditional Investigation, and Acoustic Fusion (v0.3.0 → v0.4.0)

**This is the coarsest commit boundary in the project.** The work described in this stage and the next spans three version bumps in `rubric.py`'s comments (0.3.0, 0.4.0, 0.4.1) but landed in git as a single commit, `d05ed9f` ("refit escalation fusion thresholds, add arbitrate node on tier disagreement"), alongside `2b18a6b` (batch runner + consistency harness) — both dated 2026-07-14, three weeks after the v0.2.0 checkpoint.

**v0.3.0 — evidence anchoring loop.** `anchor_node` attaches real transcript timestamps to every evidence quote and, critically, reports *which sections* cited a quote that couldn't be located verbatim in the transcript. `route_after_anchor` uses that feedback to re-run only the offending node(s) (`MAX_ANCHOR_PASSES = 2`) with a revision note quoting the unanchored text, rather than discarding the whole evaluation or trusting an ungrounded quote.

**v0.3.0 — conditional investigation.** `investigate_node` runs only when the anchored `risk_level != "none"`. It receives the first-pass escalation screening result and produces a manager-ready `InvestigationReport`: a 2–3 sentence summary, contributing factors, a recommended action, and a low/medium/high priority — its own evidence quotes get anchored too.

**v0.4.0 — acoustic-text fusion (`fusion.py`).** Deterministic post-hoc fusion, not prompt injection: the acoustic sentiment model's per-sentence output is combined with the text LLM's scores through a documented formula, so the acoustic signal's variance can never perturb the text model's judgment and every fused score stays reproducible. Three quality dimensions are fused — `professionalism` and `empathy` from the agent channel, `customer_satisfaction` from the customer channel with the final third of the call weighted 2× — via
```
a = 3 + 2 * weighted_mean(valence)              # maps [-1,1] -> [1,5]
w_a = 0.4 * coverage                            # coverage = processed / total sentences
fused = clamp(round((1 - w_a) * text + w_a * a), 1, 5)
```
Text stays primary (acoustic weight capped at 0.4) because the rubric's behaviours are text-defined; acoustics modulate rather than override. A call with no acoustic data gets `w_a = 0` and a `method: "text_only"` provenance tag.

**v0.4.0 — expected-workflow track.** `workflow_node` runs three strictly isolated LLM phases: (a) extract the call's *subject* — the request only, never the outcome — from the transcript; (b) generate 4–8 expected checklist steps from domain + subject **alone, without the transcript**, so expectations can't be shaped by what actually happened; (c) audit the transcript against that independent checklist. The isolation in (b) is the entire point of the design.

---

## Stage 4 — Threshold Fitting and Arbitration (v0.4.1) — 2026-07-14

Also inside `d05ed9f`, alongside `2b18a6b`'s new batch runner (`run_graph_batch.py`) and consistency harness (`consistency.py`).

Escalation fusion originally merged text and acoustic risk tiers by a fixed rule (ordinal max — more severe tier wins). v0.4.1 replaced that with a two-path design: tiers that **agree** merge deterministically; tiers that **disagree** are routed to a new `arbitrate_node`, an LLM call given both verdicts, the trajectory, and the transcript, because disagreement is exactly the case where a fixed formula has no basis for preferring one channel over the other.

The acoustic tier's thresholds were fitted empirically rather than asserted, using the batch runner's output across the 22-call set (`n=21` calls with paired acoustic data):

| Statistic | late_mean | peak |
|---|---|---|
| min | 0.166 | — |
| p25 | 0.219 | — |
| median | 0.227 | 0.558 |
| p75 | 0.257 | — |
| p90 | 0.281 | 0.696 |
| max | 0.406 (single outlier) | — |

`review` was set at 0.30 — just above the calm cluster's p90, so only the outlier crosses it. `escalate` was set at 0.50, deliberately outside the observed range, since nothing in the batch warranted an acoustic-only escalate and the bar stays high until data says otherwise. `peak` was evaluated as a second trigger and **rejected**: even calm, well-handled calls routinely spike past what would have been its threshold (median 0.558, p90 0.696), giving it no discriminating power. It is retained only as context handed to the arbitrator.

Validating on the full batch: of 22 calls, 18 fused by deterministic agreement, 1 had no acoustic data, and 3 were routed to arbitration — all three resolved by favoring the acoustic signal's read of the call's trajectory, including one case where a text-calm transcript was escalated to `review` on a late-call acoustic spike, which itself then triggered the investigation node.

---

## Stage 5 — Workflow Correctness Fixes and the Single-Item Recheck (v0.4.2)

Two bugs were found and fixed in the same `d05ed9f` commit that shipped the workflow track itself:

1. **Subject leakage.** The subject-extraction prompt was allowed to surface call-specific figures (dollar amounts, account numbers, dates), which made generated checklist items trivially satisfiable by the same figures already sitting in the transcript. Fixed by excluding call-specific detail from the extracted subject.
2. **False-vs-null asymmetry.** The step-checking prompt gave the model an escape hatch to mark a step `null` with no matching guidance for when to affirmatively conclude `false`, biasing it toward avoiding determinations. Fixed with an explicit rule: `null` is reserved for steps the transcript genuinely can't decide; a step that plainly never happened is `false`.
3. **Silent completeness failure.** Under retry pressure, a re-issued LLM call could return fewer checklist entries than steps given; the prior index-based join silently padded the gap with `met = null`, and a truncated response still passed schema validation. Fixed by validating the returned entry count against the given step count and raising (triggering the existing retry loop) on mismatch.

After all three fixes, a 22-call validation batch showed workflow pass rates spread meaningfully from 38% to 100% — previously an artificially high, largely null-contaminated result — with zero null entries remaining across the batch.

A fourth pass shipped separately on 2026-07-17 (`499fdbb`, commit-labeled v0.4.2): a single-item recheck fires only on steps the main pass marked `met = false`, isolated so it can never perturb a step the main pass already got right. It catches a specific failure mode — the main pass expects a direct question and misses that the value was actually established later via a confirmation or read-back. **Note:** the commit message calls this v0.4.2, but `RUBRIC_VERSION_GRAPH` in `rubric.py` and the docstring in `graph.py` were never actually bumped past `"0.4.1"` — the version constant and the commit label have drifted apart. Recorded here as a fact rather than corrected, since fixing it isn't part of this document's scope.

---

## Stage 6 — LLM Provider Evaluation and Routing — 2026-07-14

Before wiring a fixed provider into extraction, six candidate provider/model pairs were benchmarked (`benchmark.py`) on the same structured extraction task against two contrasting calls (one clean high-scoring banking call, one weaker one), scored on four criteria: schema validity, evidence fidelity (fraction of quoted evidence that is a verbatim transcript substring), score spread (does scoring actually discriminate between the two calls), and conservative bias (do `PASS` compliance items carry supporting evidence).

| Provider / Model | Latency | Evidence Fidelity | Outcome |
|---|---|---|---|
| Mistral `mistral-small-latest` | moderate | high, conservative | **Primary** — only candidate whose scores discriminated between the two calls ([5,4,3,5,2] vs [4,4,3,4,2]) |
| SambaNova `Meta-Llama-3.3-70B-Instruct` | fastest (2.6s) | 75% | **Fallback** — fastest, best raw fidelity, perfectly conservative |
| GitHub `gpt-4o-mini` | moderate | reliable JSON | **Safety** — always-on free baseline |
| SambaNova `DeepSeek-V3.1` | moderate | 50% | **Excluded** — fabricates quotes, the worst failure mode for a compliance system |
| Gemini `gemini-2.0-flash` | — | — | Benchmarked, not selected |
| Cohere `command-r7b-12-2024` | — | — | Benchmarked, not selected |

The retained tiers were wired into a LiteLLM `Router` (`router.py`) with strict priority fallback — `qa-primary` (Mistral) → `qa-fallback` (Llama-3.3-70B) → `qa-safety` (GPT-4o-mini) — `num_retries=2` per tier before falling back, `timeout=90s`, `cooldown_time=60s` for a failing deployment. A 429, timeout, or error on the primary transparently drops to the next tier instead of failing the pipeline. The routing policy is derived directly from the benchmark table, not chosen independently of it.

---

## Stage 7 — Frontend: From Single Demo Player to Routed Multi-Call Dashboard

**2026-06-09 (`a518651`)** — first React player: one hardcoded demo call (`call_data.json`), a chapter strip, and a live evaluation panel.

**2026-06-29 (`9252f90`)** — wired the v0.2.0 graph output in: `customer_satisfaction` rendering added, guarded against null for v0.1.0 back-compat, rubric version shown in the panel footer.

**2026-07-14 (`889be82`)** — `AgentLanes` gained a workflow lane, an escalation sparkline, and a valence heat ribbon — the visual surface for the fusion and workflow work landing that same week.

**2026-07-17 (`88777d6`)** — `export_frontend.py`, the actual bridge from `ml-services` output to static frontend data. For each evaluated call: turns are rebuilt from the per-channel word-level transcript with their own word-grouping logic (`assemble.py`'s `assemble_turns()` drops the per-turn `end` field the frontend player needs, so it isn't reused here); chapters are generated fresh via `segment.segment()` since they were never persisted anywhere; the evaluation JSON is copied verbatim from `results/{call_id}_graph.json`; agent and customer audio are merged into one 2-channel mp3 via ffmpeg. Run across all 22 evaluated calls, writing `calls_index.json`, `calls/{call_id}.json`, and `audio/{call_id}.mp3`. The demo call (`en_CA_Banking_1586889`) was deliberately copied verbatim from the pre-existing `call_data.json` instead of being regenerated, since it already carried a hand-verified evidence patch from the recheck-node bugfix.

**2026-07-17 (`9342f59`)** — the single-page app was split into routed `Dashboard` (list, filter, sort) and `CallDetail` (parameterized player) pages via `react-router-dom`, alongside dark mode (`ThemeProvider`/`ThemeToggle`, Tailwind `darkMode: 'class'`) applied across every component.

**2026-07-17 (`fe48427`)** — a standalone printable report view (`report.html` + `src/report/*`) was added but is not wired into the router or the Vite multi-page build config — a disconnected sub-feature as of this writing.

**Dashboard polish (this week, not yet reflected in a rubric/graph version bump since it's frontend-only).** A hero stats row (total calls, average compliance/workflow rate, needs-review count) and pagination (10 calls per page) were added on top of the routed dashboard. One bug surfaced and was fixed during this work: each row's "Call NN" label was computed from its position in the current filtered/sorted/paginated view rather than from the call's identity, so the same call could show a different number depending on how the table was sorted. Fixed by assigning every `call_id` a fixed rank once from the original `calls_index.json` order and looking it up by id, independent of view state.

---

## Challenges Faced and How They Were Solved

| Challenge | What was tried | What worked |
|---|---|---|
| One monolithic prompt covering three unrelated judgment types | Larger prompts, more instructions | Parallel LangGraph nodes — one focused prompt + Pydantic schema per rubric section, independent retry |
| LLM cites evidence that isn't actually in the transcript | Trusting the model's quotes as given | `anchor_node` verifies every quote against the real transcript; sections with unanchored quotes are looped back for a bounded re-run (max 2 passes) with the failing quotes as a revision note |
| Feeding acoustic signal into the LLM prompt directly | Prompt injection of sentiment scores | Deterministic post-hoc fusion formula outside the LLM call — keeps model variance out of the acoustic channel, keeps every fused score reproducible |
| Text and acoustic escalation tiers disagree with no principled way to merge | A fixed ordinal-max rule (more severe tier always wins) | Conditional `arbitrate_node` — only invoked on disagreement, given both verdicts plus the acoustic trajectory |
| Escalation acoustic thresholds picked by intuition | Round numbers (e.g. a flat 0.60 peak trigger) | Fitted the `late_mean` threshold to the observed 22-call percentile distribution; measured `peak` directly and dropped it once its median/p90 showed no discriminating power |
| Workflow checklist items trivially satisfiable by call-specific figures | — | Excluded dollar amounts / account numbers / dates from the subject-extraction prompt |
| Model avoided committing to `false` on workflow steps, defaulted to `null` | — | Explicit rule separating "genuinely undeterminable" (`null`) from "plainly never happened" (`false`) |
| Retried LLM call silently returned fewer checklist entries under rate-limit pressure | Index-based join padded missing entries with `met = null` (passed schema validation) | Checker validates returned entry count against the given step count, raises and retries on mismatch |
| Main-pass workflow check missed steps satisfied via a later confirmation/read-back | — | Isolated single-item recheck pass, fires only on `met = false` steps, cannot perturb steps already marked true |
| Choosing an LLM provider without evidence | Picking one provider up front | Benchmarked 6 candidates on schema validity, evidence fidelity, score spread, and conservative bias against 2 contrasting calls before wiring a router |
| A single provider outage or rate limit could fail the whole pipeline | — | LiteLLM `Router` with strict priority fallback chain and per-tier cooldown |
| Confirming the graph produces stable verdicts on identical input | Assuming determinism from a temperature=0.1 setting | `consistency.py` reruns one call N times and reports per-check, per-dimension, and risk-level agreement, flagging `UNSTABLE` on any disagreement |
| Dashboard's "Call NN" label changed identity when the table was sorted or filtered | Computing rank from the row's position in the currently displayed page | Assigned a fixed rank per `call_id` once from the original dataset order, looked up rather than recomputed |

---

## Files and Scripts

| File | Location | Purpose |
|---|---|---|
| `rubric.py` | `ml-services/evaluation/` | Pydantic v2 schema for both the legacy (v0.1.0) and graph (`RUBRIC_VERSION_GRAPH`) evaluation contracts |
| `prompts.py` | `ml-services/evaluation/` | Focused system prompts + JSON skeletons for every graph node |
| `nodes.py` | `ml-services/evaluation/` | All node functions, shared LLM retry helper, anchor/investigate/fuse/arbitrate/workflow logic |
| `graph.py` | `ml-services/evaluation/` | `StateGraph` wiring, conditional routing, CLI entry point |
| `fusion.py` | `ml-services/evaluation/` | Acoustic sentiment loading + deterministic quality/escalation fusion formulas |
| `consistency.py` | `ml-services/evaluation/` | N-run agreement harness for one call |
| `run_graph_batch.py` | `ml-services/evaluation/` | Runs the graph across every call with acoustic data, writes `batch_summary.json` |
| `benchmark.py` | `ml-services/evaluation/` | LLM provider/model comparison on schema validity, evidence fidelity, score spread, conservative bias |
| `router.py` | `ml-services/evaluation/` | LiteLLM `Router` — priority fallback chain derived from `benchmark.py` |
| `extract.py` | `ml-services/evaluation/` | Legacy monolithic v0.1.0 evaluation path (still functional, untouched by the graph work) |
| `segment.py` | `ml-services/evaluation/` | LLM call chaptering, reused directly by `chapter_node` and `export_frontend.py` |
| `assemble.py` | `ml-services/evaluation/` | Transcript packet + turn assembly shared by every node |
| `eval_call_data.py` | `ml-services/evaluation/` | `anchor_evidence()` — locates evidence quotes in transcript turns, attaches timestamps |
| `export_frontend.py` | `ml-services/evaluation/` | Exports all evaluated calls into the frontend's static data format + merged audio |
| `sentence_segments.py` / `batch_sentence_segments.py` | `ml-services/evaluation/` | Word→sentence segmentation (single-call / batch); feeds the acoustic sentiment model |

---

*Last updated: 2026-07-17*
*Current graph rubric: `RUBRIC_VERSION_GRAPH = "0.4.1"` in code (v0.4.2 workflow recheck shipped without a version bump — see Stage 5)*
*22-call evaluation batch: 18/22 escalation tiers agree by fusion, 1/22 text-only (no acoustic data), 3/22 resolved by arbitration*
*Frontend: routed Dashboard + CallDetail, dark mode, hero stats + pagination, all 22 calls exported and playable*
*Next: wire the standalone report view into the router or retire it; reconcile the `RUBRIC_VERSION_GRAPH` constant with the v0.4.2 label*
