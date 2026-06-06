# Model V2 Error Analysis

## Model Summary

- Model version: model_v2
- Run name: model_v2_cremad_ravdess
- Dataset source: CREMA-D + RAVDESS
- Raw emotion accuracy: 0.8113
- Raw emotion macro F1: 0.8113
- Business sentiment accuracy: 0.9154

## Strongest Emotion Classes

- neutral: recall=0.8996 (215/239 correct)
- anger: recall=0.8840 (221/250 correct)
- fear: recall=0.8320 (208/250 correct)

## Weakest Emotion Classes

- sadness: recall=0.6960 (174/250 correct), most confused with fear (35 cases)
- disgust: recall=0.7760 (194/250 correct), most confused with anger (19 cases)
- happy: recall=0.7840 (196/250 correct), most confused with anger (18 cases)

## Top Emotion Confusions

| Actual Emotion | Predicted Emotion | Count |
|---|---:|---:|
| sadness | fear | 35 |
| fear | sadness | 30 |
| sadness | neutral | 24 |
| disgust | anger | 19 |
| disgust | fear | 18 |
| happy | anger | 18 |
| happy | fear | 15 |
| neutral | sadness | 14 |
| sadness | disgust | 13 |
| disgust | sadness | 12 |

## Top Business Sentiment Confusions

| Actual Sentiment | Predicted Sentiment | Count |
|---|---:|---:|
| Positive/Calm | Negative/Escalated | 42 |
| Negative/Escalated | Neutral | 29 |
| Neutral | Negative/Escalated | 20 |
| Negative/Escalated | Positive/Calm | 19 |
| Positive/Calm | Neutral | 12 |
| Neutral | Positive/Calm | 4 |

## Interpretation

Model V2 improves raw emotion accuracy slightly compared with Model V1, but the remaining errors show that some emotional states are still acoustically close. The largest issue remains confusion among sadness, fear, and some neutral samples. At the business level, the most important remaining issue is Positive/Calm being misclassified as Negative/Escalated.

## Recommended Improvements for Model V3

### Sadness is still confused with fear and neutral.

- Reason: Sadness, fear, and low-energy neutral speech can overlap acoustically.
- Recommended fix: Try augmentation and unfreezing deeper Wav2Vec2 layers. Also evaluate CREMA-D and RAVDESS separately to see which dataset causes this confusion.

### Positive/Calm is still sometimes predicted as Negative/Escalated.

- Reason: Happy speech can be high-energy and acoustically close to anger.
- Recommended fix: Tune thresholds for business sentiment and consider adding acoustic feature support for positive high-energy vs aggressive high-energy speech.

### Raw accuracy improved only slightly after adding RAVDESS.

- Reason: RAVDESS improves data diversity, but the current training strategy may not fully adapt the model.
- Recommended fix: Run controlled hyperparameter experiments and test partial unfreezing instead of only frozen feature encoder training.

### Combined test accuracy may hide dataset-specific weakness.

- Reason: CREMA-D and RAVDESS have different speakers, recording conditions, and acting styles.
- Recommended fix: Evaluate Model V2 separately on CREMA-D test and RAVDESS test before deciding the next training experiment.

## Next Steps

1. Evaluate Model V2 separately on CREMA-D test and RAVDESS test.
2. Run controlled hyperparameter tuning experiments.
3. Try partial unfreezing of Wav2Vec2 layers.
4. Add training-only audio augmentation.
5. Use AppTek for realistic call-center inference/demo.
