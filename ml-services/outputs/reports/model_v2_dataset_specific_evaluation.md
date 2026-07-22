# Model V2 Dataset-Specific Evaluation

## Summary

This report evaluates Model V2 separately on CREMA-D and RAVDESS test samples. The goal is to identify whether the combined test score hides dataset-specific weakness.

## Metrics

| Dataset | Samples | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| CREMA-D | 1229 | 80.88% | 80.89% | 80.74% |
| RAVDESS | 260 | 82.31% | 82.34% | 82.75% |
| Combined | 1489 | 81.13% | 81.12% | 81.08% |

## Interpretation

Model V2 performs better on RAVDESS than on CREMA-D. This suggests that adding RAVDESS helped the model learn patterns that transfer well to that dataset, but we need to check if CREMA-D performance dropped.

## Next Improvement Direction

Use this dataset-specific result to decide whether Model V3 should focus on hyperparameter tuning, augmentation, partial unfreezing, or dataset balancing.
