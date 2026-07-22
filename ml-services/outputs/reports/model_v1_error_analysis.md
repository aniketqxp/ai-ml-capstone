# Model V1 Error Analysis

## Model Summary

- Model version: model_v1
- Model name: Wav2Vec2 emotion classifier
- Base checkpoint: Dpngtm/wav2vec2-emotion-recognition
- Dataset: CREMA-D
- Raw emotion accuracy: 0.8080
- Business sentiment accuracy: 0.9170

## Strongest Emotion Classes

- anger: recall=0.9381 (197/210 correct)
- neutral: recall=0.8715 (156/179 correct)
- fear: recall=0.8333 (175/210 correct)

## Weakest Emotion Classes

- sadness: recall=0.6714 (141/210 correct), most confused with fear (35 cases)
- happy: recall=0.7667 (161/210 correct), most confused with anger (29 cases)
- disgust: recall=0.7762 (163/210 correct), most confused with anger (19 cases)

## Top Emotion Confusions

| Actual Emotion | Predicted Emotion | Count |
|---|---:|---:|
| sadness | fear | 35 |
| happy | anger | 29 |
| disgust | anger | 19 |
| sadness | neutral | 19 |
| fear | sadness | 16 |
| disgust | fear | 14 |
| disgust | sadness | 12 |
| neutral | anger | 12 |
| fear | anger | 10 |
| sadness | disgust | 10 |

## Top Business Sentiment Confusions

| Actual Sentiment | Predicted Sentiment | Count |
|---|---:|---:|
| Positive/Calm | Negative/Escalated | 41 |
| Negative/Escalated | Neutral | 22 |
| Neutral | Negative/Escalated | 22 |
| Negative/Escalated | Positive/Calm | 8 |
| Positive/Calm | Neutral | 8 |
| Neutral | Positive/Calm | 1 |

## Interpretation

Model V1 performs strongly on the main call-center sentiment objective. The raw 6-class emotion accuracy is lower than the business-level sentiment accuracy because several raw emotion mistakes still fall within the same business category. For example, fear predicted as sadness is wrong at the emotion level but still correct as Negative/Escalated.

## Recommended Improvements for Model V2

### Sadness is weaker than other classes and is often confused with fear or neutral.

- Reason: Sad, worried, tired, or low-energy speech can be acoustically similar.
- Recommended fix: Add RAVDESS examples, apply light augmentation, and check sadness/fear label mapping carefully.

### Happy speech is sometimes confused with anger or negative emotion.

- Reason: Happy and angry speech can both be high-energy with higher pitch or louder voice.
- Recommended fix: Use business-level sentiment accuracy and consider voice-feature support to distinguish positive high energy from aggressive high energy.

### Some negative emotions are confused with each other.

- Reason: Anger, disgust, fear, and sadness are all negative states and can overlap in vocal tone.
- Recommended fix: Report both raw emotion accuracy and business sentiment accuracy, since the business use case mainly needs escalation/negative-state detection.

### The model is trained on acted emotional speech.

- Reason: CREMA-D is useful for labeled emotion learning but not identical to real call-center calls.
- Recommended fix: Use AppTek for realistic inference/demo and manually label a small AppTek subset if accuracy on call-center-like audio is needed.

## Next Steps

1. Add RAVDESS support and standardize labels.
2. Train Model V2 on CREMA-D + RAVDESS.
3. Track V2 experiments with MLflow.
4. Compare V1 and V2 using raw emotion and business sentiment metrics.
5. Use AppTek for realistic call-center inference/demo.
