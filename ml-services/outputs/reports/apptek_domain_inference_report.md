# AppTek 3-Domain Inference Report

## Purpose

This report summarizes realistic call-center inference using the selected V5 audio sentiment model on AppTek samples from three domains: banking, healthcare, and telecommunications.

## Selected Model

- Model version: `model_v5_cremad_ravdess_freeze6_epochs5_lr1e5`
- Model role: audio emotion and business sentiment inference

## Dataset Setup

- Total AppTek calls processed: 9
- Samples per selected domain: 3
- Processed duration per call: first 60 seconds
- Domains used:
  - banking: 3 calls
  - healthcare: 3 calls
  - telecommunications: 3 calls

## Overall Results

### Sentiment Distribution

- Positive: 7
- Negative: 2

### Risk Distribution

- Low: 6
- Medium: 3

- Average escalation score: 0.2642

## Domain-Level Summary

| Domain | Calls | Sentiment Distribution | Risk Distribution | Avg Escalation | Highest-Risk Call |
|---|---:|---|---|---:|---|
| banking | 3 | Positive: 2, Negative: 1 | Medium: 2, Low: 1 | 0.2693 | APPTEK_BANKING_02 |
| healthcare | 3 | Positive: 3 | Low: 3 | 0.2556 | APPTEK_HEALTHCARE_01 |
| telecommunications | 3 | Positive: 2, Negative: 1 | Low: 2, Medium: 1 | 0.2677 | APPTEK_TELECOMMUNICATIONS_03 |

## Highest-Risk Call

- Call ID: `APPTEK_BANKING_02`
- Domain: banking
- Dominant emotion: disgust
- Overall sentiment: Negative
- Risk level: Medium
- Escalation score: 0.3368

## Important Limitation

AppTek is used for realistic call-center inference/demo, not supervised emotion accuracy. The supervised accuracy was measured on CREMA-D + RAVDESS. AppTek does not provide ground-truth emotion labels for this project task.

## Output Generated

For each AppTek call, the system generated dominant emotion, business sentiment, confidence level, uncertainty flag, escalation score, risk level, and sentiment timeline.