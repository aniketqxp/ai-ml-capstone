# Evaluator v2 bounded Supervisor

Phase 9 defines the boundary between the deterministic evaluator and any LLM
used to select supporting context. The Supervisor is not an arbitrator and
cannot change attention, controlling findings, or the permitted action.

The old v1 graph and its arbitration node remain unchanged. Runtime adoption
belongs to the later v2 integration phase.

## Immutable decision lock

`build_supervisor_context` copies these fields from `CallDecision` into a
hashed lock:

- call and evaluator version;
- decision status;
- attention required;
- controlling finding identifiers;
- the single permitted action, execution policy, cited findings, and approval
  requirements;
- SHA-256 of the complete deterministic decision.

`SupervisorResult` always receives this lock from the trusted context. No
field returned by an LLM can replace it.

## Bounded context

The default packet contains at most:

- eight triggered findings, with controlling findings selected first;
- four positive findings;
- three evidence items per finding;
- two contradiction items per finding;
- four uncertainty records;
- 320 characters per text field;
- 16,000 serialized characters overall.

Only selected evidence snippets are included. The packet does not contain the
full transcript, raw signal arrays, acoustic feature values, or unrestricted
action options.

When the ordinary limits are still too large, the builder keeps compact
controlling-finding records and removes lower-priority context. Every omitted
item is counted. If even the decision lock cannot fit, context construction
fails instead of silently producing an unbounded prompt.

## LLM response

The LLM may return exactly five fields:

```json
{
  "supporting_finding_ids": [],
  "positive_finding_ids": [],
  "evidence_ids": [],
  "uncertainty_codes": [],
  "context_note": null
}
```

All identifiers must exist in the supplied packet. The optional note may
describe cited facts, but decision, action, recommendation, review, approval,
override, and similar verdict language is rejected.

Fields such as `attention_required`, `action_type`, scores, or invented
evidence are forbidden by the strict contract.

## Fallback

`resolve_supervisor_response` returns deterministic copy when the provider:

- returns no response;
- returns malformed JSON;
- violates the strict response contract;
- cites an unknown identifier;
- uses forbidden decision language.

The copy derives its headline, summary, controlling reasons, permitted action,
evidence, positives, and uncertainties from the trusted context. It does not
need another model call.

## Validation results

`v2/research/supervisor_validation_0_1.json` runs seven response cases over
each of the fourteen controlled calls:

- one valid bounded selection;
- attempted attention override;
- attempted action override;
- invented evidence reference;
- forbidden verdict language;
- malformed JSON;
- missing response.

All 98 cases produced the expected acceptance or fallback behavior, and all
98 preserved the deterministic decision lock. Context packets ranged from
6,026 to 7,944 characters under the 16,000-character limit.

## Runtime integration

The runtime sequence must be:

1. Build and validate `CallDecision`.
2. Call `build_supervisor_prompt`.
3. Send only that prompt to the configured provider.
4. Pass the raw provider response to `resolve_supervisor_response`.
5. Store the trusted `SupervisorResult`.

Provider output must never be merged directly into `CallDecision` or written
over its attention and action fields. A provider exception should be passed
as a missing response so the deterministic fallback is stored.
