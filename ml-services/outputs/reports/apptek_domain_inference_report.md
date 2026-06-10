# AppTek 3-Domain Inference Report

## Purpose

This report summarizes realistic call-center inference using the selected V5 audio sentiment model on AppTek samples from three domains: banking, healthcare, and telecommunications.

## Selected Model

- Model version: `model_v5_cremad_ravdess_freeze6_epochs5_lr1e5`
- Model role: audio emotion and business sentiment inference

## Dataset Setup

- Total AppTek calls processed: 322
- Samples per selected domain: 3
- Processed duration per call: first 60 seconds
- Domains used:
  - banking: 128 calls
  - healthcare: 114 calls
  - telecommunications: 80 calls

## Overall Results

### Sentiment Distribution

- Positive: 174
- Negative: 129
- Neutral: 11
- Mixed: 7
- Unknown: 1

### Risk Distribution

- Low: 185
- Medium: 137

- Average escalation score: 0.3010

## Domain-Level Summary

| Domain | Calls | Sentiment Distribution | Risk Distribution | Avg Escalation | Highest-Risk Call |
|---|---:|---|---|---:|---|
| banking | 128 | Positive: 72, Negative: 46, Neutral: 8, Mixed: 2 | Low: 78, Medium: 50 | 0.2938 | APPTEK_BANKING_0090 |
| healthcare | 114 | Positive: 65, Negative: 45, Mixed: 3, Neutral: 1 | Low: 69, Medium: 45 | 0.2970 | APPTEK_HEALTHCARE_0028 |
| telecommunications | 80 | Negative: 38, Positive: 37, Mixed: 2, Neutral: 2, Unknown: 1 | Medium: 42, Low: 38 | 0.3183 | APPTEK_TELECOMMUNICATIONS_0019 |

## Highest-Risk Call

- Call ID: `APPTEK_BANKING_0090`
- Domain: banking
- Dominant emotion: fear
- Overall sentiment: Negative
- Risk level: Medium
- Escalation score: 0.5735

## Important Limitation

AppTek is used for realistic call-center inference/demo, not supervised emotion accuracy. The supervised accuracy was measured on CREMA-D + RAVDESS. AppTek does not provide ground-truth emotion labels for this project task.

## Output Generated

For each AppTek call, the system generated dominant emotion, business sentiment, confidence level, uncertainty flag, escalation score, risk level, and sentiment timeline.