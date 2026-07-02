# Model V1 vs Model V2 Comparison

## Summary

Model V1 was trained on CREMA-D only. Model V2 was trained on the combined CREMA-D + RAVDESS dataset to improve generalization and reduce dependence on one emotional speech dataset.

## Dataset Setup

| Model | Dataset Source | Purpose |
|---|---|---|
| Model V1 | CREMA-D | Baseline supervised emotion model |
| Model V2 | CREMA-D + RAVDESS | Improved/generalized emotion model |

## Metrics

| Metric | Model V1 | Model V2 | Change |
|---|---:|---:|---:|
| Raw emotion accuracy | 80.80% | 81.13% | +0.33 pts |
| Raw emotion macro F1 | 80.86% | 81.13% | +0.27 pts |
| Business sentiment accuracy | 91.70% | 91.54% | -0.16 pts |
| Business sentiment macro F1 | 88.26% | 88.33% | +0.07 pts |

## Interpretation

Model V2 slightly improves raw emotion accuracy and macro F1 compared with Model V1. The improvement may be modest, but Model V2 is trained on a broader emotional speech dataset, which makes it more suitable for generalization than a CREMA-D-only model.

The business-level sentiment metric remains important because the project goal is call-center risk detection. In this setting, confusing fear with sadness is less severe than confusing Negative/Escalated speech with Neutral or Positive/Calm speech.

## Next Improvement Direction

The next accuracy-improvement experiments should focus on hyperparameter tuning, freezing/unfreezing Wav2Vec2 layers, training-only audio augmentation, and dataset-specific evaluation on CREMA-D and RAVDESS separately.

After these supervised experiments, AppTek should be used for realistic call-center inference/demo. If manual labels are added for AppTek segments, it can also be used for domain-specific business sentiment evaluation.
