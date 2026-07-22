Model V1 is the first locked version of the semantic/audio sentiment analysis model for the capstone project. It is used as the baseline for future improvements such as RAVDESS integration, augmentation, hyperparameter tuning, and AppTek realistic inference testing.


- Model type: Wav2Vec2ForSequenceClassification

- Base checkpoint: Dpngtm/wav2vec2-emotion-recognition

- Framework: Hugging Face Transformers

- Task: 6-class speech emotion classification
## Emotion Classes

- anger

- disgust

- fear

- happy

- neutral

- sadness

## Dataset

- Dataset: CREMA-D

- Input type: audio-only WAV files

- Split strategy: speaker-aware train / validation / test split

- Train samples: 5147

- Validation samples: 1066

- Test samples: 1229

## Training Configuration

- Epochs: 3

- Batch size: 2

- Learning rate: 1e-5

- Device: Apple Silicon MPS

- Fine-tuning method: supervised Hugging Face Trainer fine-tuning

- Feature encoder: frozen in current training setup

- Classifier head: reinitialized for 6 CREMA-D emotion classes

## Results

- Validation accuracy: 81.52%

- Validation macro F1: 81.44%

- Test accuracy: 80.80%

- Test macro F1: 80.86%

## Confusion Matrix Summary

The strongest classes were anger, fear, and neutral. Sadness was the weakest class and was most often confused with fear and neutral. This is expected because sad, worried, and low-energy speech can sound acoustically similar.

## Why This Model Is Locked

This model is now considered Model V1. All future experiments must be compared against this version to prove whether they improve performance or generalization.

Future model versions may include:

- CREMA-D + RAVDESS training

- audio augmentation

- class balancing

- hyperparameter tuning

- freezing/unfreezing experiments

- AppTek realistic call-center inference testing

## Final Inference Output

Model V1 is already connected to the semantic pipeline and produces:

- dominant emotion

- overall audio sentiment

- emotion probabilities

- audio features

- sentiment timeline

- confidence level

- uncertainty flag

- escalation score

- score breakdown

- peak emotion timestamp
