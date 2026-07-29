# Evaluator v2 integration

## Rollout boundary

Evaluator v2 runs beside v1. It does not replace `EVALUATOR_VERSION=v1`.

Two flags control the rollout:

- `EVALUATOR_V2_SHADOW=1` runs v2 after the v1 graph inside the processing
  worker.
- `VITE_EVALUATOR_V2=1` allows the frontend to render a successful v2
  presentation. Calls without one use the existing v1 page.

The flags are independent. A hosted worker can collect shadow results while the
frontend remains on v1.

## Runtime sequence

For each processed call:

1. v1 produces its graph evaluation as before.
2. The v2 shadow runner normalizes sentence segments and optional acoustic
   output into a `SignalBundle`.
3. Banking calls resolve the research banking profile and a grounded profile
   selection.
4. Findings, attention, action, and presentation are derived deterministically.
5. v1 and v2 attention are compared only when v2 requirement coverage is
   complete.
6. A shadow envelope records the decision, presentation, limitations, legacy
   proxy, and comparison status.

The shadow runner catches and records its own errors. A v2 failure cannot fail
the primary v1 processing job.

## Current coverage

The ten existing banking calls have profile-selection fixtures, so all ten
complete shadow execution. They do not yet have grounded semantic requirement
assessment batches. Their v2 status is therefore `insufficient_evidence`, and
the page states that the evaluation is incomplete.

The initial shadow batch records:

- 10 successful shadow executions;
- 10 v2 `insufficient_evidence` decisions;
- 10 v1 legacy attention proxies;
- 0 v2 attention decisions;
- 0 comparable attention pairs; and
- 0 reported disagreements.

The v1 proxy and v2 attention value use different policies. Reporting the raw
10-versus-0 result as disagreement would be invalid while v2 coverage is
incomplete.

Fresh banking calls need a grounded profile selection before v2 can evaluate
them. Other domains remain `unsupported_domain` until they have their own
profile. Both states are stored rather than inferred as no attention.

## Versioned storage

Supabase Storage receives:

- `evaluation-runs/{call_id}/v1-{content_hash}.json`
- `evaluation-runs/{call_id}/v2-{decision_hash_or_run_id}.json`
- `evaluation-v2/{call_id}.json` as the latest v2 read target

The `evaluation_runs` table stores the job, public call ID, runtime run ID,
evaluator version, mode, status, decision hash, attention value, and complete
payload. Reprocessing the same job replaces that job's matching version row;
separate jobs remain separate run records.

The existing `evaluations` table remains the v1 compatibility record.

## API contract

The artifact API exposes:

- `GET /evaluation-v2/{call_id}.json`
- `GET /calls/{call_id}/evaluation-runs`
- `POST /calls/{call_id}/feedback`
- `GET /calls/{call_id}/feedback`

Feedback writes to `evaluation_feedback` and must cite the current decision
hash. Finding feedback must cite a known finding ID. Action feedback must cite
the permitted action type. Approval is rejected when the action does not
require manager approval.

Supported feedback types are:

- `approve_action`
- `dismiss_decision`
- `dismiss_finding`
- `confirm_finding`
- `action_completed`
- `action_failed`

The API records approval and completion; it does not claim that recording
feedback executes an external business operation.

## Call page

The v2 call page implements the Phase 10 communication contract:

- one literal attention state;
- no more than two reasons;
- one constrained action;
- no more than two positive highlights;
- one selected evidence timeline;
- explicit evaluation completeness;
- the full seekable transcript and audio; and
- one evaluation-details disclosure.

Raw sentiment, acoustic telemetry, quality ratings, compliance percentages,
workflow percentages, and duplicate analysis lanes are absent. Unsupported or
missing v2 runs load the existing v1 call page.

In local static-artifact mode, feedback is saved to browser storage when the
API is unavailable. Hosted builds require the API response to contain a
feedback ID before reporting that server feedback was recorded.

## Verification

For a local static preview:

```powershell
cd frontend
npm run dev:e2e
```

Open `/calls/en_CA_Banking_1586889` to inspect an actual incomplete banking
shadow run. `npm run test:e2e` starts or reuses the same preview server.

The browser suite covers:

- the actual incomplete banking shadow state;
- an attention state with reason, action, positive handling, and evidence;
- local feedback behavior;
- mobile width without horizontal overflow; and
- v1 fallback for an unsupported domain.

Human comprehension testing remains separate. Browser checks establish that the
implemented states render and behave correctly; they do not establish manager
understanding.

## Promotion gate

V2 should remain in shadow mode until:

1. fresh calls receive grounded profile selections;
2. semantic requirement assessments are produced and validated;
3. a meaningful number of complete v1/v2 pairs exist;
4. disagreements are reviewed against adjudicated evidence;
5. manager feedback is inspected for systematic dismissal patterns; and
6. unfamiliar-user comprehension testing is run on the interactive page.
