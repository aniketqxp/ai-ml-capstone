# Banking Domain Profile

## Purpose

`banking-demo-v1` is the first evaluator v2 domain profile. It converts the
banking test calls into explicit intent, applicability, requirement, and
workflow definitions.

It is a capstone project policy in `research` status. It is not a bank policy,
legal interpretation, or regulatory compliance standard.

## Supported Intents

The ten current banking calls contain four supported intents:

| Intent | Calls |
|---|---:|
| Recurring account transfer | 6 |
| One-time account transfer | 3 |
| Checking-account opening | 1 |
| Agricultural-equipment loan inquiry | 1 |

One call contains both one-time and recurring transfers, so the counts exceed
ten.

The runtime evaluator may select more than one intent. It cannot create an
intent that is absent from the profile.

## Requirements And Workflow

Each business requirement is defined once. Workflow branches contain ordered
references to those requirements rather than separate copies.

For example:

- `transfer.amount_confirmed` defines the business meaning, evidence
  expectation, severity, and finding types.
- `transfer.core` references that requirement at the appropriate point in the
  transfer sequence.

This prevents separate "compliance" and "workflow" findings from reporting the
same omission twice.

The profile currently contains:

- 23 requirements
- 10 workflow branches
- 23 ordered workflow references

## Requirement Levels

- `critical`: a missed applicable requirement may support attention by itself,
  subject to evidence and the later decision policy.
- `required`: expected business handling whose effect depends on outcome and
  surrounding evidence.
- `supportive`: useful positive behavior that does not independently require
  attention when absent.

Agent identification is supportive. Identity verification, transaction
details, recurring schedule, and authorization are critical where applicable.

## Applicability Facts

Conditional rules use structured facts:

- whether a transfer is being executed
- whether it completed during the call
- whether an account is being opened
- whether an existing product should be linked
- whether a loan inquiry advances into an application

If a required fact is missing, the resolver records unresolved applicability.
It does not mark the requirement as failed and does not silently apply it.

Examples:

- A general loan inquiry does not require applicant identity verification.
- Identity verification becomes applicable when a loan application begins.
- Product-linkage requirements apply only when linkage is requested.
- Transfer completion confirmation applies only when completion occurred.

## Deliberate Exclusions

The profile does not contain a blanket recording-disclosure requirement.

It also does not require:

- an exact loan rate during a general inquiry
- a reference number when another clear transfer confirmation exists
- identity verification for information-only loan questions
- unspecified terms invented by an LLM

Adding such requirements later requires a profile version change and an
identified policy source.

## Runtime Boundary

The runtime model may:

- select supported intents
- select structured applicability facts
- locate evidence for resolved requirements

It may not:

- invent requirements
- change requirement levels
- add workflow steps
- describe project rules as regulatory obligations
- treat unresolved applicability as a failure

The profile and ten intent-selection fixtures are stored in
`ml-services/evaluation/v2/profiles/`.
