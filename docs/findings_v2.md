# Evaluator v2 Findings

## Purpose

Phase 5 translates grounded observations into business findings. It does not
decide whether a call needs attention and does not select dashboard elements.

The implementation keeps three layers separate:

1. The domain plan decides which requirements apply.
2. A grounded assessment states whether each applicable requirement was met,
   missed, or uncertain.
3. Deterministic finding logic assigns the business category, polarity,
   severity, evidence, and visibility.

The semantic assessor cannot invent requirement names, categories, severities,
or actions.

## Requirement Assessments

Each `RequirementAssessment` contains:

- one applicable `requirement_id`
- a `met`, `missed`, or `uncertain` verdict
- a rationale
- supporting evidence
- counter-evidence
- reliability and provenance

Supporting evidence must reference a segment, signal, or episode in the call's
`SignalBundle`. Quoted transcript evidence must be a verbatim substring of the
referenced segment, and the speaker must match.

Assessments for non-applicable requirements are rejected. Missing assessments,
uncertain verdicts, and unusable assessments become explicit uncertainties;
they never become failures.

## Requirement Findings

The domain profile controls the translation:

- critical missed requirements become critical negative findings
- required missed requirements become review findings
- met requirements become informational positive findings when the profile
  defines a positive finding type
- supportive requirements without a failure type cannot create a negative
  finding

Security, authorization, and transaction-accuracy requirements map to required
controls. Process, outcome, and service requirements retain their own business
categories.

Multiple applicable requirements that produce the same finding type are merged.
Their evidence and counter-evidence are retained, while duplicate outputs are
recorded in `suppressed_duplicates`.

## Explicit Text Escalation

The deterministic text detector deliberately covers only explicit customer
language:

- direct manager or supervisor requests
- stated formal complaints
- stated legal, regulatory, or ombudsman escalation
- explicit repeated-contact reports
- direct statements of strong dissatisfaction

Generic negative sentiment and inferred emotion do not create these findings.
For example, a sentence mentioning that a manager previously reviewed an
account does not match the manager-request rule.

The next agent response is retained as counter-evidence so the event is not
presented without the immediate handling context.

## Acoustic Support

A persistent Phase 4 acoustic-elevation episode may be attached to an explicit
text escalation finding when it occurs within 20 seconds. It corroborates the
same finding rather than creating a second one.

Acoustic elevation alone cannot create an escalation finding. Its semantics and
thresholds remain provisional until the Phase 8 comparison against labeled and
controlled calls.

An acoustic recovery candidate becomes an internal positive finding only when:

- an explicit text escalation finding exists
- that finding has nearby persistent acoustic support
- the Phase 4 recovery candidate is true

Recovery describes delivery after the event, not proof that the customer's
business issue was resolved.

## Evidence Policy

Every Phase 5 finding contains both evidence and counter-evidence. Findings use
`details` visibility by default, while provisional acoustic recovery remains
`internal`. Phase 10 will determine which validated findings earn primary-page
visibility.

No finding score is averaged with another finding, and Phase 5 does not produce
an overall score.

## Current Banking Batch

The pinned ten-call banking batch contains 103 applicable requirement checks.
No v2 requirement-assessment files exist yet, so all 103 are recorded as
uncertainties instead of being imported from the older blanket compliance and
generated-workflow outputs.

The high-precision explicit-text rules produce no escalation findings on the
ten clean calls. This is expected for the current data and does not establish
sensitivity. Challenge calls and human labels are required in Phases 7 and 8.

## Reproduction

From `ml-services/evaluation`:

```powershell
python -m v2.run_findings_batch `
  --profile-ref HEAD `
  --transcript-ref HEAD `
  --sentiment-ref origin/integration/full-pipeline-test `
  --output ml-services/evaluation/v2/research/findings_0_1.json
```

Use `--assessment-dir` when grounded assessment batches become available. The
summary records a SHA-256 digest for each supplied assessment file.
