# Transcription Pipeline — Development Journal

*Banking Call Compliance Automation | Capstone Project*
*AI/ML Engineering | 2025–2026*

---

## Overview

This document traces the full development arc of the call transcription pipeline — from first attempt to working baseline to planned improvements. It is written chronologically and is intended to serve as a reference for the project report, academic presentation, and as a record of the engineering decisions, dead ends, and breakthroughs along the way.

The broader project is a banking call compliance automation system. The transcription component is the first stage of that pipeline: given a recorded call between a bank agent and a customer, produce an accurate, speaker-attributed, word-level-timestamped transcript that downstream compliance and analytics components can reason about.

---

## Stage 1 — The Starting Point and the First Failure

### What we had

The dataset is `apptek-com/apptek_callcenter_dialogues` on HuggingFace — a collection of real call-center recordings across 14 accents and 16 domains. Each call is stored as two separate mono WAV files: one for the agent (channel 1) and one for the customer (channel 2). The ground-truth transcripts are included per channel.

For North American accents, the dataset contains 59 **en-CA** (Canadian English) calls and 67 **en-US_General** (General American English) calls spanning domains including banking, aviation, retail, health, telecom, and others — 126 calls total.

### First approach

The initial transcription approach used the **OpenAI Whisper API** (`whisper-1` model) with a plan to use **WhisperX** locally for forced word alignment. WhisperX is a library that wraps Whisper and adds wav2vec2-based forced alignment, promising tighter word-level timestamps.

### Problems encountered

**Problem 1 — No punctuation from the API.** The OpenAI API's whisper-1 endpoint returns transcript text with no punctuation, making speaker attribution and downstream NLP unreliable.

**Problem 2 — Timestamp drift.** A specific and very concrete failure was observed: a speaker said *"Is that possible"* at approximately 5:02 in the recording. The transcript marked it at **5:14** — twelve seconds late. This is not a rounding error; it is a structural problem.

**Root cause identified:** The transcription was being run on the isolated customer channel (channel 2). That channel contains only the customer's voice — meaning roughly **53–84% of the audio is silence**, with the customer's speech scattered across it. Whisper uses Silero VAD internally to skip silence, but when VAD filters most of a file, the internal timestamp counter drifts relative to the original audio clock. Timestamps become untrustworthy.

**Problem 3 — WhisperX is not usable with the API.** WhisperX requires local access to both the audio and the model weights; it cannot wrap an API call. So the two tools cannot be combined as initially planned. Getting better timestamps would require going fully local.

---

## Stage 2 — Rethinking the Architecture

### The key insight

The silence problem is caused by transcribing isolated channels directly. The fix required rethinking which audio signal to give the model.

**The mono mix approach:** Instead of feeding Whisper one isolated channel at a time, both channels are **summed to a mono mix** before transcription. Because both speakers contribute, the mix has far less silence — approximately 42% vs. 53–84% on isolated channels. Whisper's VAD no longer drifts, and timestamps are accurate relative to the original recording timeline.

**Speaker attribution via channel energy:** Knowing which channel has more acoustic energy during each transcribed segment tells us who is speaking. Since the two channels are physically separate microphone tracks (not a diarization problem — the speakers literally never appear in each other's channel), comparing RMS energy on ch0 vs. ch1 for each segment reliably identifies the speaker.

The formula is simple:

```
RMS(chunk, ch0) >= RMS(chunk, ch1)  →  agent spoke this segment
```

This is not diarization — it is a lookup. The stereo structure of the data makes speaker separation a solved problem in principle; the challenge is executing it correctly.

### Why this matters architecturally

This decision — transcribe the blend, attribute using the channels — is a deliberate trade-off. It trades away the theoretical cleanliness of per-channel transcription in exchange for stable, drift-free timestamps on a single model pass. The attribution via energy works well when channel isolation is clean, which it is for the majority of calls in this dataset (correlation between channels is essentially zero when checked — the microphones are truly separate).

The remaining limitation is that the energy comparison is a per-segment decision, not a word-level one. If a segment contains speech from both sides (overlap), the whole segment goes to whichever channel was louder. This is the source of the remaining attribution errors.

---

## Stage 3 — Building the Local Pipeline

### Tech stack chosen

**faster-whisper** is a reimplementation of OpenAI's Whisper using CTranslate2, an optimized inference engine. It produces identical output to the original Whisper but runs significantly faster and supports int8 quantization on CPU, making it practical without a GPU.

The full local stack:

| Component | Choice | Reason |
|---|---|---|
| ASR model | `faster-whisper small.en` | English-only, 244M params, good accuracy/speed balance |
| Quantization | `int8` on CPU | Runs without GPU; ~3× speed vs. float32 |
| Word timestamps | Built-in DTW alignment | No second model required; avoids the WhisperX memory issue |
| VAD | Silero (via `vad_filter=True`) | Suppresses silence before feeding frames to Whisper |
| Domain conditioning | `initial_prompt` | Seeds the model with banking vocabulary to improve domain-specific accuracy |
| Audio I/O | `soundfile` | Loads WAV directly to numpy, no ffmpeg dependency |

The pipeline in `ml-services/scripts/transcribe_mixed.py`:

1. Load both channel WAVs with `soundfile`
2. Stack to stereo array in memory (no mixed file written to disk)
3. Compute mono mix as the mean of the two channels
4. Feed mono mix to faster-whisper with `word_timestamps=True`, `vad_filter=True`, and `initial_prompt`
5. For each transcribed segment, compare RMS energy on ch0 vs. ch1 to assign speaker
6. Output agent and customer word arrays with `{word, start, end}` per word

Runtime: approximately 170 seconds per 10-minute call on CPU (roughly 3× real-time).

### Evaluation methodology

Measuring accuracy on speech transcription requires care. Word Error Rate (WER) is the standard metric — it counts substitutions, deletions, and insertions relative to a reference transcript and expresses the result as a fraction of total reference words. **Accuracy = 1 − WER**, so higher accuracy is better and 90% is the target.

Two normalization levels are always reported:

- **Raw accuracy** — only punctuation and case are stripped
- **Normalized accuracy** — the **Whisper EnglishTextNormalizer** is applied, which canonicalizes number words (`"four hundred"` = `"400"` = `"$400"`), contractions, spelling variants, etc.

Early in development, a hand-rolled number normalizer was used that was specifically tuned to the numbers appearing in one particular call. This produced an inflated 94.5% on that call. When replaced with the canonical Whisper normalizer — which generalizes to any number format — the honest number was **92.8%**. The lesson: evaluation methodology must be general-purpose and locked down before drawing conclusions about model performance.

The evaluation scripts use `jiwer.process_words()` for WER computation, shared normalization logic in `eval_common.py`, and always print both raw and normalized metrics side by side to make the methodology transparent.

---

## Stage 4 — First Real Result

Running `small.en/int8` on the first test call (a Canadian English banking call, approximately 10 minutes) produced:

| Metric | Score |
|---|---|
| Raw accuracy | 92.0% |
| Normalized accuracy | **92.8%** |

An error analysis on the remaining 7.2% of errors showed:

- ~60% were **measurement artifacts** — number format differences, smart apostrophe encoding issues in the reference, disfluency annotation mismatches (`"ohh"` vs `"oh"`, `"(um)"` in reference not transcribed)
- ~40% were **genuine transcription errors** — mostly minor: wrong plural, wrong tense, a proper noun misspelled

The ground truth itself contained typos and transcriber annotations (`"tranfer"`, `"piece of mind"`, doubled words) that create a ceiling below 100% even for perfect transcription. These are not errors in our system — they are errors in the reference.

**Conclusion from Stage 4:** The pipeline works. Single model, fully local, no API, accurate timestamps, correct speaker attribution on clean audio.

---

## Stage 5 — Scaling to 126 Calls

### Building the batch pipeline

To evaluate performance across the full North American accent set rather than a single cherry-picked call, three new scripts were built:

- **`build_na_testset.py`** — Downloads the parquet index from HuggingFace, filters to en-CA and en-US_General, pairs channel1/channel2 into calls, downloads all WAV files, and writes a manifest. Resumable (skips files already on disk).
- **`run_batch.py`** — Loads the model once, transcribes every call in the manifest, writes per-call JSON results. Resumable.
- **`evaluate_batch.py`** — Reads manifest references and result hypotheses, computes per-call WER (both raw and normalized), aggregates by accent and domain, and reports distribution statistics.

The 126-call full run took **~7.75 hours** on CPU.

### Results across 126 calls

| Accent | Calls | Raw accuracy | Normalized accuracy |
|---|---|---|---|
| en-CA | 59 | 79.2% | 81.5% |
| en-US_General | 67 | 76.6% | 79.6% |
| **Overall** | **126** | **77.9%** | **80.6%** |

The result is lower than the single-call baseline of 92.8%, but that baseline call turned out to be one of the best-performing calls in the entire set. The distribution across 126 calls is:

| Accuracy band | Calls | Share |
|---|---|---|
| ≥ 90% (passing) | 35 | 28% |
| 80–90% | 35 | 28% |
| 70–80% | 37 | 29% |
| < 70% | 19 | 15% |

### Domain breakdown (normalized accuracy)

The two extremes were Telecom at 88.8% and Aviation at 74.4%. Banking — the target domain — sat at **83.4%**, with individual calls ranging from 64.9% to 94.3%.

### Root cause of variance

The key finding from error analysis: **the dominant failure mode is speaker attribution, not transcription quality.** The worst banking call (`en_CA_Banking_1588683`, 64.9%) had 284 agent-attributed words in the hypothesis versus 482 in the reference — approximately 200 words were transcribed correctly but landed in the wrong speaker bucket. Every one of those counts as a deletion from the agent and an insertion into the customer, catastrophically inflating WER even though Whisper produced correct text.

The 35 calls that already pass 90% demonstrate that when attribution is correct, the small.en pipeline is comfortably on target. The problem is reliability of attribution, not accuracy of transcription.

---

## Stage 6 — Error Taxonomy and the Road Forward

### What the 80.6% baseline really means

For a post-processing pipeline, the question is not whether raw accuracy hits a threshold but whether the output is **salvageable** — coherent enough to serve as input to correction stages.

The answer is yes. The transcripts are readable, semantically coherent, and structurally intact. The errors break down as:

| Error type | Approximate share | Recoverable? |
|---|---|---|
| Number / currency format | ~15% | Yes — normalizer |
| Proper nouns (bank names, places) | ~10% | Yes — domain glossary |
| Disfluency handling differences | ~10% | Yes — cleanup filter |
| Speaker misattribution | ~20% | Yes — per-channel transcription |
| Ground-truth reference defects | ~10% | N/A — not real errors |
| Genuine wrong words | ~35% | Partially — LLM correction |

Only ~35% of errors are genuine wrong-word substitutions, and many of those are correctable by context (e.g., `"Critics Union"` → `"Credit Union"`).

### Planned improvements, sequenced by expected payoff

The pipeline improvement plan is deliberately ordered by payoff divided by effort — avoiding the trap of reaching for complex solutions before simple ones are exhausted.

**Phase 0 — Instrumentation (no accuracy gain, but enables everything)**
Capture and store per-segment confidence signals (`avg_logprob`, `no_speech_prob`, per-word probability) from faster-whisper. These are available in the output already and currently discarded. They become the signal for intelligent routing in later phases.

**Phase 1 — Per-channel transcription**
The current approach transcribes the mono mix and uses energy to guess speaker. The more principled approach: apply VAD per channel, transcribe each channel independently, and let the channel identity be the speaker label. Attribution is no longer a heuristic — it is a structural property of the data. Expected to recover the bottom 15% of calls (those with attribution failures) from ~65–75% accuracy into the high 80s.

**Phase 2 — Loudness normalization**
Normalize each channel's loudness before mixing or per-channel transcription. This is cheap, safe, and directly addresses cases where a quiet agent channel has its energy outcompeted by the customer channel during attribution comparisons.

**Phase 2b — Denoising (experimental)**
Libraries like `noisereduce` can perform spectral gating to reduce background noise. This is listed as experimental because Whisper was trained on deliberately noisy audio and is robust to it; aggressive denoising can introduce artifacts the model was not trained on and actually hurt performance. Any denoising applied must be measured against a held-out sample, not assumed to help.

**Phase 3 — Confidence-based model routing**
Use the confidence signals from Phase 0 to selectively escalate: low-confidence calls or segments get re-run through `medium.en` rather than running the larger model on everything. This delivers the accuracy benefit of a bigger model at a fraction of the compute cost.

**Phase 4 — Larger model (medium.en)**
As a fallback if Phase 3 is not sufficient: switch the full pipeline to `medium.en/int8`. Historically, medium provides a 3–5 percentage-point accuracy improvement over small. The trade-off is approximately 3× the runtime per call.

**Phase 5 — LLM cleanup pass**
A domain-primed language model pass over the raw transcript, gated by confidence scores so it only touches spans where the ASR was uncertain. Banking domain priming makes this highly effective for proper nouns and product names. Applied last because (a) LLM corrections are more reliable when the underlying transcript is cleaner, and (b) over-correction risk is real — the LLM may "fix" words that were correct.

**Phase 6 — Fine-tuning (capstone stretch goal)**
Fine-tuning `small.en` or `medium.en` on a held-out slice of in-domain data. Requires a strict train/test split from a dataset separate from the one used for evaluation — using the apptek evaluation set for fine-tuning would invalidate all benchmarks.

### Realistic accuracy ceiling

For context on expectations: world-class ASR on clean read speech achieves 95–97% accuracy. On spontaneous, two-party, telephone-quality audio with disfluencies, overlapping speech, and variable recording conditions, **90–93% is genuinely strong and 95% is exceptional**. Human transcribers disagree with each other by 2–4% on this type of audio. The reference transcripts in this dataset contain their own errors and annotation conventions that create a ceiling below 100%.

A realistic target for the full pipeline, post-improvements, is **92–95% normalized accuracy on the banking domain** with good generalization to the broader NA accent set.

---

## Execution Log — Phases 0–2 and Error Analysis (2026-05-30)

This section records what actually happened when the planned phases were executed — including a phase that was tested and **rejected**, which is itself a result worth documenting.

### Methodology: the frozen probe set

Running the full 126-call set takes 7.75 hours. Iterating on it is impossible. So a **frozen 12-call probe** (`data/na_testset/probe_set.json`) was hand-picked to span the entire quality range — 4 "catastrophic" calls (29–48% baseline), 2 "bad banking", 2 "mid banking", 2 "good banking", 2 "good other" — and weighted toward the banking target domain. Every pipeline change is validated on the probe (~43 min) before any full run is considered. This single decision is what made rapid, methodical iteration possible.

### Phase 0 + 1 executed — per-channel transcription: the structural win

Phase 0 (confidence capture) and Phase 1 (per-channel transcription) were implemented together (`transcribe_channels.py`, `run_probe.py`). Instead of transcribing the mono mix and *guessing* the speaker by energy, each channel is transcribed independently and the channel identity **is** the speaker label. faster-whisper's built-in VAD (`vad_filter=True`) handles the silence-drift problem that originally pushed us to the mono mix, remapping timestamps back onto the true timeline.

Result on the probe — **per-channel vs mono-mix, same 12 calls:**

| Tier | mono-mix | per-channel | Delta |
|---|---|---|---|
| Catastrophic (4 calls) | 29–48% | 87–91% | **+42 to +59 pts** |
| Bad banking (2) | 65–70% | 89–91% | +21 to +25 pts |
| Mid banking (2) | 77–80% | 90–95% | +13 to +15 pts |
| Good banking (2) | 92–94% | 92–95% | flat (±1%) |
| Good other (2) | 90–94% | 92–93% | flat (±1%) |
| **Probe overall** | **69.5%** | **91.3%** | **+21.8 pts** |

This confirmed the central hypothesis: **speaker attribution, not transcription quality, was the dominant error source.** Every struggling call was rescued; nothing good regressed. The +21.8 point probe gain is the single largest improvement in the project. (The probe overall is lower than the 80.6% full-set number because the probe is deliberately stacked with hard calls.)

Phase 0 also paid off immediately: a smoke test caught a Whisper silence-hallucination — *"Thank you for calling outside airlines"* — flagged by `no_speech_prob = 0.88`.

### Phase 2 executed — preprocessing tested and REJECTED (a negative result)

Phase 2 (high-pass filter + loudness normalization, `audio_preprocess.py`) was implemented and probed. It was **rejected as a default**, for evidence-based reasons:

- **High-pass filter — near-neutral.** Spectral analysis showed these recordings already carry ~0% energy below 80 Hz (high-passed upstream by the telephony stack). The filter is correct, cheap insurance but moves nothing on this data. Honesty over theater.
- **Loudness normalization — does not help.** Channel levels vary 20 dB (−17 to −37 dBFS active RMS), so normalizing seemed promising. But a **gate sweep** (see below) showed boosting quiet channels is neutral at best (the quietest call gained +0.1%) and harmful at worst (a mid-banking call regressed −5.9% when its quiet agent channel was amplified). The reason is structural: **Whisper's encoder internally normalizes its mel-spectrogram by its own peak, so it is already largely gain-invariant** — amplifying the waveform gives the model nothing new and occasionally perturbs decoding on carefully-articulated digits.

**Methodology highlight — simulating the gate without re-running.** The gated pipeline's output is, per channel, *already known*: a channel above the gate ≈ its Phase 1 transcript, below the gate = its Phase 2 transcript. So the optimal gate threshold was found by **assembling results from the two existing runs** (`simulate_gate.py`) and sweeping — zero new transcription, seconds of compute. Every gate setting landed within noise of pure Phase 1 (91.3%). Decision: **reject loudness normalization; Phase 1 stands.**

One real benefit was observed but judged insufficient: preprocessing roughly halved the hallucination signal (58 → 30 high-`no_speech` segments). Since that did not translate into better WER, the cleaner signal was not worth the regression risk.

### Error analysis — what remains, and how it reframes the roadmap

With Phase 1 locked at 91.3%, the remaining errors were dissected (`error_analysis.py`) to decide the next move. Two findings reshaped the plan:

**Q1 — What are the errors?** (normalized S=479, D=658, I=187 over the probe)

| Category | Share of errors | LLM-fixable? |
|---|---|---|
| Function / filler words (`i`, `okay`, `and`, `ohh`) | 58.8% | No — cannot restore deleted backchannel |
| Content deletions | 17.5% | No — words aren't present to fix |
| Numbers (account / phone digits) | 11.6% | Formatting only |
| Lexical (homophones, proper nouns) | 8.3% | Partly — but ~half are ground-truth defects |
| Morphology (plural / tense) | 3.8% | Partly |

**Deletions are ~50% of all errors**, and most are filler/backchannel the reference transcribed and Whisper reasonably omitted (`okay` ×43, `ohh` ×11). Several "lexical errors" are actually reference defects where our output is *correct* (`precription → prescription` — the reference is misspelled). The WER is therefore somewhat *deflated* by artifacts that are not real transcription failures.

**Q2 — Does confidence predict errors? Mostly no.** Low-confidence words (prob < 0.5) have a 5× higher error rate (20% vs 3.9%) — a real signal — **but they hold only 14% of all errors while being 4% of words.** The other 86% of errors are in *confident* words, because the biggest error bucket (deletions) has no hypothesis token to be low-confidence in the first place.

**Roadmap consequences:**
- **Phase 3 (confidence-based routing): DROPPED.** It would catch at most 14% of errors. The evidence does not support it.
- **Phase 5 (LLM cleanup): REFRAMED.** It cannot reach 99% — the realistically addressable slice (homophones, proper nouns, number formatting) is ~1–1.5% absolute WER. Its true value is **downstream transcript quality for compliance** (clean proper nouns, readable numbers), not a WER chase. Built conservatively (`llm_cleanup.py`) with a strict minimal-edit prompt and a WER-before/after guardrail to catch over-correction.
- **Phase 4 (medium.en): now the primary WER lever.** Reducing substitutions/deletions requires a stronger acoustic model, not post-processing. This is the machine-heavy job to queue.

### Where the pipeline stands

**Locked:** faster-whisper `small.en/int8`, **per-channel** transcription with built-in VAD, confidence captured per word. **Rejected:** loudness normalization, confidence routing. **Probe accuracy: 91.3% normalized.** Next definitive step: full 126-call per-channel run (confirm the number), then a `medium.en` probe to size the real WER lever.

---

## Challenges Faced and How They Were Solved

| Challenge | What we tried | What worked |
|---|---|---|
| Timestamp drift (5:14 instead of 5:02) | WhisperX forced alignment via wav2vec2 | Mono mix transcription — eliminated silence drift at source |
| WhisperX requires local model + GPU | Two-model pipeline (API + local alignment) | Single local model (faster-whisper) handles both |
| OOM / paging crashes on isolated channels | Larger models on per-channel audio | Mono mix avoids the numpy padding bug; single smaller model fits in RAM |
| API transcription had no punctuation | Post-processing punctuation restoration | Dropped the API entirely; faster-whisper returns punctuated output |
| Evaluation inflation (hand-rolled normalizer) | Custom number map per file | Whisper's canonical EnglishTextNormalizer, generalizes to any number format |
| Smart apostrophe encoding split contractions | Manual text preprocessing | Normalizer converts all curly quotes to straight apostrophes |
| Speaker attribution failures (200 words in wrong bucket) | Tighter energy thresholds on mono mix | Per-channel transcription — attribution becomes structural, +21.8 pts on probe |
| 7-hour batch runs with no early validation | Running full 126-call set speculatively | Frozen 12-call probe set spanning the quality range — validate in ~43 min |
| Deciding if preprocessing helps without 8h re-runs | Re-transcribing per setting | Assembled gated output from two existing runs (`simulate_gate.py`) — swept thresholds in seconds |
| Knowing whether to build confidence routing | Assuming low confidence finds errors | Measured it — confidence flags only 14% of errors; dropped the phase |
| Local LLM cleanup blocked on RAM | qwen2.5:7b (needs ~5 GB) | Model-agnostic script; falls back to smaller model / runs when RAM is freed |

---

## Files and Scripts

| File | Location | Purpose |
|---|---|---|
| `eval_common.py` | `ml-services/scripts/` | Shared Whisper normalizer — single source of truth for all scoring |
| `build_na_testset.py` | `ml-services/scripts/` | Downloads 126 NA-accent calls from HuggingFace, writes manifest |
| `transcribe_mixed.py` | `ml-services/scripts/` | **(deprecated)** mono-mix + energy attribution — superseded by per-channel |
| `run_batch.py` | `ml-services/scripts/` | Mono-mix batch runner (produced the 80.6% baseline in `results/`) |
| `transcribe_channels.py` | `ml-services/scripts/` | **Phase 1** per-channel transcriber + Phase 0 confidence capture |
| `audio_preprocess.py` | `ml-services/scripts/` | **Phase 2** high-pass + gated loudness norm (tested, not default) |
| `run_probe.py` | `ml-services/scripts/` | Runs a method over the frozen 12-call probe (~43 min) |
| `simulate_gate.py` | `ml-services/scripts/` | Finds optimal loudness gate from existing runs — no re-transcription |
| `compare_probe.py` | `ml-services/scripts/` | A/B any two result dirs + no_speech diagnostics + confidence-filter sweep |
| `error_analysis.py` | `ml-services/scripts/` | Error categorization + confidence-vs-error correlation |
| `llm_cleanup.py` | `ml-services/scripts/` | **Phase 5** conservative LLM correction pass (Ollama), WER-guarded |
| `evaluate_batch.py` | `ml-services/scripts/` | Aggregate WER by accent/domain, worst-call ranking, CSV export |
| `probe_set.json` | `data/na_testset/` | Frozen 12-call probe spanning the quality range |
| `results_summary.csv` | `data/na_testset/` | Per-call accuracy for all 126 calls (mono-mix baseline) |

Result directories under `data/na_testset/`: `results/` (mono-mix, full 126), `results_channels/` (per-channel, 12 probe), `results_channels_pp/` (per-channel + preprocessing, 12 probe), `results_channels_llm/` (LLM-cleaned, pending RAM).

---

*Last updated: 2026-05-30*
*Current pipeline: faster-whisper small.en / int8 / CPU / **per-channel** transcription + built-in VAD + per-word confidence*
*Probe accuracy (12 hard calls): **91.3% normalized** | mono-mix baseline (126 calls): 80.6% | rejected: loudness norm, confidence routing*
*Next: full 126-call per-channel run (definitive number) → medium.en probe (real WER lever)*
