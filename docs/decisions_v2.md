# Evaluator v2 Attention and Actions

## Purpose

Phase 6 converts Phase 5 findings into one deterministic call decision. It
does not calculate an overall quality score and does not ask an LLM to choose
whether a call needs attention.

The attention rule is:

```text
attention_required = any(finding qualifies for attention)
```

Positive findings, excluded findings, and the number of non-qualifying
findings cannot average away one qualifying negative finding.

## Decision Status

`attention_required = false` means that no available negative finding
qualified. It does not necessarily mean that the call was cleared.

The separate `decision_status` field distinguishes:

- `complete`: no applicable requirement-assessment gaps remain
- `partial`: findings exist, but one or more applicable requirements remain
  unassessed or uncertain
- `insufficient_evidence`: requirement gaps remain and no findings are
  available

This prevents an incomplete pipeline run from presenting a false pass.

## Finding Qualification

Each negative finding receives one trace entry.

A finding qualifies when:

- severity is `review` or `critical`
- reliability is `usable` or `limited`
- visibility is not `internal`

Informational, internal, unavailable, and unusable findings do not qualify.
Uncertainties are retained but do not become negative findings and therefore
cannot directly trigger attention.

Limited findings may trigger human attention because the permitted actions
either create a review artifact or require approval. They do not authorize a
consequential automatic action.

## Precedence

Attention is an OR, while precedence selects which qualifying finding controls
the recommended action. The policy order is:

1. critical escalation
2. critical required control
3. critical outcome
4. other critical findings
5. review outcome
6. review required control
7. review escalation
8. review process
9. review agent behavior
10. review data quality

All findings in the highest applicable group remain controlling. Lower groups
remain in the decision as supporting detail; they are not discarded.

## Recovery

`escalation.recovery_observed` is recorded as `context_only`. It credits a
supported improvement in customer delivery but does not cancel an explicit
manager request, formal complaint, missed control, or unresolved outcome.

Phase 8 may test a narrower suppression rule for review-level dissatisfaction.
No such rule is assumed before labeled evidence exists.

## Action Policy

| Controlling finding | Action | Execution |
| --- | --- | --- |
| Any critical finding | Create review case | Automatic |
| Review escalation or required control | Create review case | Automatic |
| Review outcome | Request customer follow-up | Requires approval |
| Repeated unresolved contact | Request customer follow-up | Requires approval |
| Review process or agent behavior | Recommend coaching | Requires approval |
| Other review finding | Create review case | Automatic |

Creating a review case is the only automatic action. Coaching and customer
contact remain recommendations requiring human approval. A recommended action
may cite only the controlling findings.

## Policy Trace

Every `CallDecision` stores:

- policy identifier and version
- literal `any_qualifying_finding` aggregation
- one qualification result for every negative finding
- controlling finding identifiers
- recovery finding identifiers and effect
- SHA-256 of the exact `SignalBundle`

Contract validation rejects a decision when its Boolean, action references,
recovery references, or signal-bundle hash contradict this trace.

Presentation selection remains empty in Phase 6. Phase 10 decides which
validated findings appear on the first call view.

## Current Banking Batch

The ten banking calls currently have no v2 requirement assessments and no
explicit-text escalation findings. Their decisions are therefore:

- zero calls requiring attention
- ten calls with `insufficient_evidence`
- ten `none` actions
- 103 retained requirement uncertainties

These calls are not classified as clean. They remain unassessed.

## Reproduction

From `ml-services/evaluation`:

```powershell
python -m v2.run_decisions_batch `
  --profile-ref HEAD `
  --transcript-ref HEAD `
  --sentiment-ref origin/integration/full-pipeline-test `
  --output ml-services/evaluation/v2/research/decisions_0_1.json
```

Use `--assessment-dir` after grounded requirement-assessment files exist.
