# Evaluator v2 Acoustic Dynamics

## Purpose

Phase 4 derives temporal acoustic observations without assigning business
meaning to them. It answers whether the customer's delivery changed, persisted,
recovered, or remained elevated near the end. It does not decide whether a
call needs attention.

All Phase 4 signals and episodes use `internal` visibility and `limited`
reliability. Later validation determines whether they can support a finding.

## Inputs

The derivation consumes Clara's sentence-aligned outputs through the v2
`SignalBundle` adapter:

- model escalation score
- mean pitch
- mean volume
- RMS energy
- pause ratio
- speech rate
- segment timestamps, speaker labels, processing status, and quality flags

Imported call-level trajectory fields are not reused. Phase 4 recomputes its
observations from segment-level evidence.

## Reliability Gates

Each feature is evaluated independently for each speaker. A damaged pitch
measurement therefore cannot disable a usable pause or emotion-model series.

The current engineering gates require:

- at least 12 usable observations
- usable coverage of at least 50 percent for `usable`
- usable coverage from 35 through 50 percent for `limited`
- lower coverage to be `unavailable`

These are data-availability gates, not escalation thresholds. They can be
changed for engineering reasons without claiming that an acoustic value is
dangerous.

Feature-specific quality flags exclude measurements before normalization.
For example, `missing_pitch` excludes pitch but does not exclude speech rate.
A feature with no within-speaker variation is also omitted because it has no
defensible normalization scale.

## Normalization

Values are normalized within speaker and call using median/MAD robust
z-scores, with an IQR fallback when MAD is zero. This avoids comparing a
customer's pitch directly with an agent's pitch or treating one speaker's
natural baseline as universally high.

The output is a relative change within the current call. It is not a
population percentile and must not be displayed as one.

## Episode Candidates

A segment becomes an elevation candidate only when multiple signal families
agree:

- the emotion-model escalation value is elevated and at least one acoustic
  family corroborates it; or
- at least two acoustic families are strongly elevated

Volume and energy count as one intensity family because they are closely
related measurements. Pitch, intensity, speech rate, and pause behavior are
the other families.

An isolated segment cannot form an episode. A timestamped episode requires at
least two nearby elevated customer segments spanning at least two seconds.
The resulting object is named an acoustic elevation candidate, not anger,
escalation, dissatisfaction, or misconduct.

## Trajectory

Early and late windows are thirds of usable customer speech, not thirds of
wall-clock call duration. Agent speech, silence, and customer segments without
usable normalized evidence do not stretch these windows.

The derivation records:

- usable customer seconds
- early and late elevated-speech ratios
- early-to-late model-escalation change
- trajectory direction
- local recovery candidate
- unresolved-ending candidate

Recovery and overall direction describe different comparisons. A call may
drop after a local episode while still end above an unusually calm opening.

## Current Batch Observation

The pinned 22-call enriched batch forms no episode on 18 calls. Three calls
form one episode and one call forms two. Two calls contain a recovery candidate
and one contains an unresolved-ending candidate.

Customer pitch is unavailable on 20 calls and customer volume and energy are
unavailable on 19 because quality flags and usable coverage fail the gates.
Model escalation, pause ratio, and speech rate are usable on most calls.

This is an availability finding, not validation of the episode semantics.
The clean corpus lacks enough genuine escalation to estimate sensitivity or
false-positive rates. Phase 8 must compare these candidates with human labels
and controlled challenge calls before any acoustic result can affect an
attention decision.

## Reproduction

Run the batch against the enriched branch artifacts:

```powershell
python -m v2.run_acoustic_dynamics_batch `
  --sentiment-ref origin/integration/full-pipeline-test `
  --transcript-ref HEAD `
  --output ml-services/evaluation/v2/research/acoustic_dynamics_0_1.json
```

Run from `ml-services/evaluation` so the `v2` package resolves normally.
