# Evaluator v2 signal validation

Phase 8 compares text-only, audio-only, and multimodal evaluator variants
against the Phase 7 dataset. The experiment keeps real-call observations
separate from controlled known-answer checks because they support different
claims.

## Reproduction

The checked report pins:

- Banking profile and transcripts:
  `03097cb56eeb4234d6aaa3932b3a2e3e3c51cdeb`
- Clara acoustic outputs:
  `a63045fee0a1b98082f951cca584267a6705c50a`
- Evaluation dataset: `banking-evaluator-v2` version `0.1.0`

Run from `ml-services/evaluation`:

```powershell
python -m v2.run_signal_validation `
  --profile-ref 03097cb56eeb4234d6aaa3932b3a2e3e3c51cdeb `
  --transcript-ref 03097cb56eeb4234d6aaa3932b3a2e3e3c51cdeb
```

The detailed and compact outputs are:

- `v2/research/signal_validation_0_1.json`
- `v2/research/signal_validation_summary_0_1.json`

## Controlled results

The controlled population contains fourteen calls in seven counterfactual
pairs. Semantic requirement assessments are known-answer fixtures derived
from each authored manipulation. These results validate deterministic
finding, evidence, decision, and action behavior; they do not measure an
LLM's ability to assess a real transcript.

| Variant | Sensitivity | False-positive rate | Reason recall | Evidence localization | Pair checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text only | 1.00 | 0.00 | 1.00 | 1.00 | 5/7 |
| Audio only | 0.00 | 0.00 | 0.00 | unavailable | 1/7 |
| Multimodal | 1.00 | 0.00 | 1.00 | 1.00 | 7/7 |

Text alone passes the transcript, control, outcome, and duration pairs. It
cannot satisfy the delivery-support or recovery-context expectations because
those intentionally require acoustic evidence.

Multimodal evaluation passes all seven pairs. The acoustic condition adds
evidence to four controlled findings and produces one recovery finding. It
does not change the underlying attention label.

Audio alone produces no attention decisions. This is intentional: an acoustic
episode is not allowed to create a manager request, missed authorization,
unclear outcome, or other business finding without grounded semantic
evidence.

## Existing-call results

The ten existing banking calls use the actual sentence transcripts and
Clara's enriched acoustic output. No grounded requirement-assessment batches
exist for these calls yet, so this run can exercise only the explicit-text and
acoustic-support detectors.

All three variants produce:

- zero attention decisions;
- zero false positives among the six draft no-attention seeds;
- four false negatives against the four draft attention seeds;
- zero matched expected reasons;
- zero qualifying acoustic-elevation episodes in the banking batch.

The resulting 0.60 agreement with researcher seed labels is not accuracy.
It reflects six no-attention matches while the semantic requirement layer is
absent. The missed seeds concern authorization, identity verification,
recurring-change instructions, and dissatisfaction wording outside the
current high-precision text patterns.

Expanding a text rule from one draft example would be fitting to an
unadjudicated label. Phase 8 therefore records the miss without changing the
rule.

## Decisions

Audio remains supporting evidence and recovery context. It is not eligible to
trigger or clear attention independently.

No thresholds were fitted. The real calls have no human-adjudicated labels,
and the challenge calls are authored, so either population would give a
misleading basis for threshold optimization.

Human agreement is unavailable because all ten existing-call labels remain
draft researcher seeds. Once humans adjudicate those calls and grounded
requirement assessments are generated independently, the same harness can
measure semantic sensitivity, reason selection, evidence localization, and
whether acoustic support improves any decision outcome.
