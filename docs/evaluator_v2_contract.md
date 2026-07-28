# Evaluator v2 Contract

## Scope

Phase 2 defines the boundary between signal production, evaluation decisions,
storage, and presentation. It does not activate evaluator v2, add thresholds,
or change the deployed v1 behavior.

The contract has two top-level payloads:

- `SignalBundle`: normalized transcript, acoustic, and derived observations.
- `CallDecision`: evidence-backed findings, the attention decision, and the
  permitted action.

Language-neutral JSON Schemas are generated in
`ml-services/evaluation/v2/contracts/`.

## Signal Bundle

Every bundle identifies the call, source versions, transcript segments,
signals, temporal episodes, and modality coverage.

Important rules:

- `segment_id` is the canonical internal join key.
- `seq_id` is preserved for transcript compatibility but is not assumed unique.
- Every signal records its producer, producer version, modality, scope,
  reliability, and source location.
- Missing or skipped model outputs are represented through coverage and
  reliability rather than fabricated zeroes.
- Raw signals cannot use `primary` visibility.
- Temporal episodes may reference only segments and signals present in the same
  bundle.

Current acoustic payloads map:

- sentence sentiment label
- dominant emotion label
- escalation score
- model top probability
- negative-emotion probability

The enriched payload from `integration/full-pipeline-test` additionally maps:

- pitch statistics
- RMS energy
- mean volume
- pause duration, count, and ratio
- speech rate
- feature quality flags
- candidate trajectory direction, delta, start, end, and peak
- candidate de-escalation and unresolved-ending indicators

Existing derived trajectory values are marked `limited` with the reason
`imported_unvalidated_derived_signal`. Existing arbitrary empathy and
resolution scores are intentionally not mapped.

## Decision Contract

The primary result remains:

```json
{
  "attention_required": true,
  "triggered_findings": [],
  "positive_findings": [],
  "recommended_action": {}
}
```

Each finding must include:

- business definition
- domain-profile applicability
- versioned detection rule
- evidence
- counter-evidence
- reliability and coverage limitations
- severity and visibility

Contract rules prevent:

- attention without a negative finding
- a visible finding without evidence
- a positive finding in `triggered_findings`
- a negative finding in `positive_findings`
- actions that reference unknown findings
- a non-empty action when attention is false
- raw signals being promoted directly to the primary page

## Visibility

Visibility is data policy, not styling:

- `primary`: eligible for the first call-page view.
- `evidence`: shown while verifying a finding.
- `details`: available through expansion.
- `internal`: retained for evaluation, validation, and diagnostics.

Signals and episodes cannot be `primary`. A detector must translate them into
an evidence-backed finding first. The later presentation phase will decide how
eligible findings are visually represented.

## Actions

The contract currently permits:

- no action
- create a review case
- recommend coaching
- request customer follow-up
- request policy review

Only review-case creation may be automatic. Coaching, customer follow-up, and
policy review require human approval.

## Source Files

- `ml-services/evaluation/v2/schemas.py`: strict Pydantic models and invariants.
- `ml-services/evaluation/v2/adapters.py`: current and enriched payload mapping.
- `ml-services/evaluation/v2/validation.py`: decision-to-signal reference checks.
- `ml-services/evaluation/v2/domain_profiles.py`: profile validation and
  deterministic applicability resolution.
- `ml-services/evaluation/v2/findings.py`: grounded requirement assessments,
  explicit-text rules, acoustic corroboration, and deterministic findings.
- `ml-services/evaluation/v2/run_findings_batch.py`: pinned banking-batch
  finding derivation and assessment-gap reporting.
- `ml-services/evaluation/v2/profiles/`: versioned domain rules and selection
  fixtures.
- `ml-services/evaluation/v2/generate_contract_schemas.py`: JSON Schema export.
- `ml-services/evaluation/v2/test_contracts.py`: contract and mapping tests.

Evaluator v2 remains unregistered in `evaluator_runtime.py`. Setting
`EVALUATOR_VERSION=v2` continues to fail explicitly until its decision logic is
implemented and validated in later phases.
