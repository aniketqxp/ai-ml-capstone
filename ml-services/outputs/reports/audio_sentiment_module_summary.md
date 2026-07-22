# Audio Sentiment Analysis Module Summary

## Module Purpose

The Audio Sentiment Analysis module is responsible for analyzing call audio and converting it into business-ready emotional intelligence for the capstone call-center QA system.

Instead of only predicting a raw emotion label, the module produces outputs that can be used by the backend, dashboard, and workflow analysis components, including:

- Dominant emotion

- Overall business sentiment

- Negative emotion probability

- Escalation score

- Risk level

- Confidence level

- Uncertainty flag

- Sentiment timeline

- Peak emotion moment

- Audio feature summary

This makes the module useful for call-center quality monitoring, escalation detection, and customer experience review.

---

## Supervised Training Datasets

The supervised emotion model was trained and evaluated using two labeled emotional speech datasets:

### CREMA-D

CREMA-D was used as the first supervised emotion dataset. It provided labeled acted speech samples for the main emotion classes used by the project.

### RAVDESS

RAVDESS was added to improve generalization and reduce dependence on one dataset. RAVDESS samples were standardized into the same six-emotion schema used by the project.

### Final Emotion Schema

The final project emotion schema contains six classes:

- anger

- disgust

- fear

- happy

- neutral

- sadness

The business sentiment mapping is:

- anger, disgust, fear, sadness → Negative/Escalated

- neutral → Neutral

- happy → Positive/Calm

---

## Model Development and Experiments

Several Wav2Vec2-based experiments were trained and evaluated. The goal was to improve both raw emotion classification and business-level sentiment usefulness.

### Baseline and Early Models

Model V1 was trained on CREMA-D only and achieved strong initial results.

Model V2 added RAVDESS to create a combined CREMA-D + RAVDESS dataset. This improved generalization and allowed the model to be evaluated across two emotional speech datasets.

### Improvement Experiments

Several improvement directions were tested:

- Increasing training epochs

- Changing learning rate and warmup ratio

- Freezing different numbers of Wav2Vec2 transformer layers

- Adding light training-only audio augmentation

- Reusing compatible classifier weights from the pretrained checkpoint

The strongest improvement came from freezing the Wav2Vec2 feature encoder and freezing the lower six transformer layers. This allowed the model to adapt higher-level layers while preserving useful pretrained speech representations.

---

## Final Selected Model

The final selected model is:

`model_v5_cremad_ravdess_freeze6_epochs5_lr1e5`

### Final Model Configuration

- Base checkpoint: `Dpngtm/wav2vec2-emotion-recognition`

- Dataset: CREMA-D + RAVDESS

- Epochs: 5

- Batch size: 2

- Learning rate: 1e-5

- Freeze feature encoder: True

- Freeze transformer layers: 6

- Augmentation: Disabled

### Final Model Metrics

- Raw emotion accuracy: 83.41%

- Raw emotion macro F1: 83.27%

- Business sentiment accuracy: 91.00%

- Business sentiment macro F1: 88.11%

### Reason for Selecting V5

Model V5 was selected because the project goal is not only exact emotion classification. The main business goal is to identify call-center sentiment and escalation risk.

Although Model V8 achieved a slightly higher raw emotion accuracy, Model V5 achieved the strongest business sentiment accuracy while maintaining strong raw emotion performance. Therefore, V5 was selected as the final model for inference and demo integration.

---

## Final Inference Pipeline

The selected V5 model was connected to the inference pipeline through `EmotionPredictor`.

The inference pipeline now loads:

`outputs/wav2vec2/model_v5_cremad_ravdess_freeze6_epochs5_lr1e5/best_model`

The output schema includes:

- `overall_audio_sentiment`

- `dominant_emotion`

- `negative_emotion_probability`

- `anger_probability`

- `stress_probability`

- `sadness_probability`

- `anxiety_probability`

- `calm_probability`

- `audio_features`

- `emotional_volatility`

- `audio_sentiment_shift`

- `audio_escalation_score`

- `risk_level`

- `prediction_confidence`

- `confidence_level`

- `uncertain_prediction`

- `top_emotion_margin`

- `peak_emotion`

- `sentiment_timeline`

- `model_version`

Both the standalone emotion predictor test and the full sentiment pipeline test confirmed that the selected V5 model is being used successfully.

---

## AppTek Call-Center Inference

After selecting the final model, the module was tested on realistic call-center audio using the AppTek Call Center Dialogues dataset.

AppTek is used for realistic inference and demo, not supervised emotion accuracy, because it does not provide ground-truth emotion labels for this project task.

### AppTek Domains Used

Three industry-relevant call-center domains were selected:

- Banking

- Healthcare

- Telecommunications

The raw AppTek domain names were mapped as follows:

- banking → banking

- health → healthcare

- telecom → telecommunications

### AppTek Sample Setup

- Total calls processed: 9

- Samples per domain: 3

- Processed duration per call: first 60 seconds

- Model used: `model_v5_cremad_ravdess_freeze6_epochs5_lr1e5`

### AppTek Inference Results

Overall sentiment distribution:

- Positive: 7

- Negative: 2

Overall risk distribution:

- Low: 6

- Medium: 3

Average escalation score:

- 0.2642

### Domain-Level Results

| Domain | Calls | Sentiment Distribution | Risk Distribution | Average Escalation | Highest-Risk Call |

|---|---:|---|---|---:|---|

| banking | 3 | Positive: 2, Negative: 1 | Medium: 2, Low: 1 | 0.2693 | APPTEK_BANKING_02 |

| healthcare | 3 | Positive: 3 | Low: 3 | 0.2556 | APPTEK_HEALTHCARE_01 |

| telecommunications | 3 | Positive: 2, Negative: 1 | Low: 2, Medium: 1 | 0.2677 | APPTEK_TELECOMMUNICATIONS_03 |

The highest-risk call was:

- Call ID: `APPTEK_BANKING_02`

- Domain: banking

- Dominant emotion: disgust

- Overall sentiment: Negative

- Risk level: Medium

- Escalation score: 0.3368

---

## Module Contribution to the Full Capstone System

This module contributes the audio intelligence layer of the full call-center QA pipeline.

In the overall system, the module can support:

- Detecting emotionally negative or escalated calls

- Highlighting risky calls for QA review

- Providing evidence for dashboard alerts

- Creating sentiment timelines across long calls

- Supporting root-cause and action agents with emotional context

- Helping managers compare call risk across domains

The final AppTek inference results show that the model can run on realistic call-center audio and generate structured outputs suitable for backend storage, dashboard visualization, and downstream AI agents.

---

## Important Limitation

The supervised accuracy metrics come from CREMA-D + RAVDESS because those datasets include emotion labels.

AppTek is used for realistic call-center inference and demonstration, not supervised accuracy, because AppTek does not provide ground-truth emotion labels for this project task.

Therefore, AppTek results should be presented as inference/demo results, while CREMA-D + RAVDESS results should be presented as supervised model evaluation.

---

## Final Status

The Audio Sentiment Analysis module is now ready for integration with the backend/dashboard pipeline.

Completed items:

- Labeled audio dataset preparation

- RAVDESS integration

- Combined CREMA-D + RAVDESS metadata

- Wav2Vec2 fine-tuning experiments

- MLflow tracking

- Model comparison and error analysis

- Final model selection

- V5 inference integration

- AppTek 3-domain sample preparation

- AppTek inference output generation

- AppTek 3-domain inference report

Next integration step:

Connect the AppTek sentiment summary and JSON outputs to the backend/dashboard so the call-center QA system can display sentiment, risk, timeline, and escalation insights per call.

