# Final Selected Emotion Model

## Selected Model

Model V5: `model_v5_cremad_ravdess_freeze6_epochs5_lr1e5`

## Reason for Selection

Model V5 was selected as the final emotion model because the capstone use case focuses on call-center business sentiment and escalation risk, not only exact emotion classification. Although Model V8 achieved a slightly higher raw emotion accuracy, Model V5 achieved the strongest business sentiment accuracy while maintaining strong raw emotion performance.

## Dataset

The model was trained and evaluated using the combined CREMA-D + RAVDESS dataset.

## Training Configuration

- Base checkpoint: `Dpngtm/wav2vec2-emotion-recognition`

- Epochs: 5

- Batch size: 2

- Learning rate: 1e-5

- Freeze feature encoder: True

- Freeze transformer layers: 6

- Augmentation: Disabled

## Final Metrics

- Raw emotion accuracy: 83.41%

- Raw emotion macro F1: 83.27%

- Business sentiment accuracy: 91.00%

- Business sentiment macro F1: 88.11%

## Business Label Mapping

- Anger, disgust, fear, sadness → Negative/Escalated

- Neutral → Neutral

- Happy → Positive/Calm

## Next Step

The selected V5 model will be used for realistic call-center inference on AppTek samples to generate emotion predictions, business sentiment, confidence, escalation score, risk level, and sentiment timeline outputs.

