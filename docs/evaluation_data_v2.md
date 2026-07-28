# Evaluator v2 evaluation data

Phase 7 adds a versioned evaluation dataset for the banking profile. It is
separate from training data and from runtime evaluator output.

## Artifacts

- `ml-services/evaluation/v2/research/evaluation_data_0_1.json` contains ten
  existing-call seed annotations, fourteen controlled challenge calls, and
  seven counterfactual pairs.
- `ml-services/evaluation/v2/research/evaluation_data_summary_0_1.json`
  records annotation status and coverage counts.
- `ml-services/evaluation/v2/evaluation_data.py` defines the contracts and
  validates source hashes, evidence fidelity, provenance, pair references,
  and declared invariants.
- `ml-services/evaluation/v2/build_evaluation_data.py` reproduces the checked
  artifacts from canonical banking transcripts and authored challenge cases.

Run the builder and checks from `ml-services/evaluation`:

```powershell
python -m v2.build_evaluation_data
ruff check v2
python -m unittest discover -s v2 -p "test_*.py"
```

## Existing calls

The ten real-call annotations are researcher seeds, not human ground truth.
Each one records expected attention, expected negative and positive finding
types, exact transcript evidence, and unresolved interpretation notes.

Their provenance is deliberately constrained:

- `annotator_type`: `model_assisted_researcher_seed`
- `status`: `draft`
- `requires_human_review`: `true`
- `ground_truth_basis`: absent

The contract rejects an attempt to mark this provenance as reviewed or
adjudicated. A human reviewer must read each referenced transcript, revise the
label where necessary, and record a human reviewer identity before the labels
can be used for human-agreement or accuracy claims.

The first seed review marks six calls as not requiring attention and four as
requiring attention. This distribution is a review queue, not a reported
accuracy result.

## Controlled challenges

The fourteen authored calls form seven known-answer pairs:

| Axis | Controlled change | Expected behavior |
| --- | --- | --- |
| Transcript | Contextual use of "manager" becomes an explicit request | Attention changes |
| Delivery | Calm delivery becomes persistent elevation | Acoustic support changes; attention does not |
| Outcome | Clear completion becomes ambiguous acknowledgement | Attention changes |
| Control | Identity verification is removed | Attention changes |
| Control | Explicit authorization is removed | Attention changes |
| Recovery | Elevated delivery recovers or remains unresolved | Recovery context changes; attention does not |
| Duration | Segment timing expands with identical words | Decision does not change |

Synthetic known answers are marked `synthetic_author` and `adjudicated`
because the expected difference follows from the authored single-variable
change. They test evaluator sensitivity and invariance; they do not estimate
performance on real calls.

## Phase 8 use

Phase 8 should run text-only, audio-only, and multimodal evaluator variants
against the same dataset version. Report existing-call results separately
from controlled challenge results, and do not calculate human agreement until
the existing-call queue has human-reviewed labels.
