# Evaluator v2 result communication

## Purpose

The call evaluator page answers six questions:

1. Does this call need attention?
2. Why?
3. Where is the evidence?
4. What was handled well?
5. What action follows?
6. Is the evaluation complete?

Anything that does not improve one of these answers stays in details,
diagnostics, or the underlying call record.

## Decision language

Attention is a categorical decision, not a score:

- `Needs attention` means at least one deterministic finding qualified under
  the decision policy.
- `No review finding identified` means the applicable assessed rules produced
  no qualifying negative finding. It does not mean the call passed every
  possible standard.
- `Evaluation incomplete` means applicable requirements remain unassessed. The
  call is not presented as cleared.

The page does not show an overall percentage, a five-point quality rating, a
friction score, or confidence. Those values would combine unlike claims or
imply calibration that the project has not established.

## Information order

### Primary

The first reading path contains:

- one attention state;
- no more than two controlling reasons;
- one next action when attention is required;
- no more than two positive highlights; and
- an incompleteness notice when coverage is partial or insufficient.

The state is a literal text label. Reasons are plain-language finding rows, not
numeric scales. Positive findings use checkmarks because they are completed
behaviors or outcomes, not points toward a total.

The action is shown as a command with its execution rule. `Automatic` means the
policy permits automated execution; it does not claim that the backend already
performed the action. `Manager approval required` distinguishes a recommendation
from an executed action.

### Evidence

Each visible reason and positive highlight links to selected evidence. A
transcript quote includes speaker and timestamp and seeks the audio when a
timestamp exists. Acoustic evidence is labelled as supporting context and
cannot appear as an independent verdict.

There is one evidence timeline. Separate compliance, workflow, quality,
sentiment, and escalation lanes are not shown because they repeat locations and
force the user to reconcile internal evaluator categories. The full transcript
and audio remain available as the source record.

### Details

A disclosure contains:

- non-controlling negative findings;
- additional positive findings;
- counter-evidence availability;
- uncertainty messages;
- the bounded Supervisor context note; and
- evaluator, policy, and domain-profile versions.

Severity, detection thresholds, qualification traces, and reliability mechanics
remain internal. Limited reliability is exposed only as a local note on the
affected finding.

### Internal

Raw sentiment probabilities, pitch, volume, energy, speech rate, normalized
acoustic values, model thresholds, hashes, and decision traces are diagnostic
inputs. They are not business conclusions and are never page-level elements.

## Page composition

The Phase 11 page should use this reading order:

1. attention state and incompleteness notice;
2. primary reasons and the next action;
3. handled-well checkmarks;
4. selected evidence aligned to the audio timeline;
5. full conversation; and
6. one evaluation-details disclosure.

The primary section should not repeat evidence text already present in the
timeline. Evidence timestamps answer where a finding occurred, so there is no
separate location section. Additional finding counts may indicate hidden detail,
but no collapsed section should compete visually with the primary decision.

## Structural validation

The controlled challenge suite checks that every projected scenario has:

- a literal attention state;
- at most two primary reasons;
- evidence for every primary reason;
- at most two positive highlights;
- exactly one action when attention is required and none otherwise;
- an explicit incomplete state or notice;
- no repeated finding or evidence item; and
- no field named as a score, percentage, rating, or confidence.

These checks test whether the contract contains one answer path. They do not
measure whether an unfamiliar person understands the rendered page.

## Human comprehension protocol

After the Phase 11 page is interactive, unfamiliar participants should inspect
representative clean, attention, recovery, and incomplete calls. Without
coaching, each participant answers the six questions in the Purpose section and
points to the page element used.

The page should be revised when participants:

- infer that no finding means a universal pass;
- cannot locate the evidence for a reason;
- mistake acoustic support for an independent decision;
- cannot distinguish a recommended action from an executed one; or
- overlook incomplete evaluation coverage.

The current report records this study as `not_run`; controlled structural checks
must not be reported as human comprehension results.
