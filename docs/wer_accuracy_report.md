# WER Accuracy Report
*Generated: 2026-06-15 | Model: faster-whisper small.en/int8 + stereo channel attribution*

## Overall
| Metric | Value |
|--------|-------|
| Calls evaluated | 22 |
| Total reference words | 32,059 |
| Normalised WER | 9.8% |
| **Normalised accuracy** | **90.2%** |
| Target (>=90%) | PASS |

## By Domain
| Domain | Calls | Accuracy (normalised) | Pass? |
|--------|-------|-----------------------|-------|
| banking | 10 | 91.1% | yes |
| health | 7 | 86.5% | no |
| telecom | 5 | 93.2% | yes |

## Per-Call Results (normalised accuracy)
| Call ID | Domain | Acc (norm) | Acc (raw) | Agent WER | Customer WER |
|---------|--------|------------|-----------|-----------|--------------|
| en_CA_Banking_1586889 | banking | 92.2% | 91.5% | 7.3% | 9.2% |
| en_CA_Banking_1588683 | banking | 89.8% | 82.6% | 5.2% | 13.0% |
| en_CA_Banking_1590992 | banking | 94.8% | 92.4% | 4.2% | 7.7% |
| en_CA_Banking_1592237 | banking | 94.4% | 93.4% | 5.4% | 6.1% |
| en_US_General_Banking_1584540 | banking | 90.5% | 89.1% | 10.1% | 7.9% |
| en_US_General_Banking_1586157 | banking | 89.2% | 82.9% | 10.7% | 10.9% |
| en_US_General_Banking_1586678 | banking | 95.7% | 90.5% | 3.1% | 6.0% |
| en_US_General_Banking_1586893 | banking | 91.8% | 87.8% | 5.1% | 16.1% |
| en_US_General_Banking_1587139 | banking | 86.6% | 80.5% | 13.9% | 12.6% |
| en_US_General_Banking_1587700 | banking | 88.5% | 85.4% | 6.8% | 21.8% |
| en_CA_Health_1587315 | health | 93.8% | 90.4% | 4.8% | 9.0% |
| en_CA_Health_1588706 | health | 93.3% | 92.1% | 5.3% | 8.9% |
| en_CA_Health_1590851 | health | 74.4% | 72.4% | 12.9% | 37.9% |
| en_US_General_Health_1586594 | health | 91.3% | 87.0% | 7.9% | 10.3% |
| en_US_General_Health_1586726 | health | 92.0% | 88.4% | 4.1% | 12.0% |
| en_US_General_Health_1587175 | health | 88.2% | 83.8% | 10.3% | 14.3% |
| en_US_General_Health_1587922 | health | 86.5% | 83.9% | 9.4% | 20.7% |
| en_CA_Telecom_1590675 | telecom | 95.1% | 92.5% | 3.4% | 10.8% |
| en_CA_Telecom_1590992 | telecom | 93.3% | 92.0% | 6.9% | 6.1% |
| en_US_General_Telecom_1584567 | telecom | 90.6% | 85.5% | 6.1% | 17.3% |
| en_US_General_Telecom_1586751 | telecom | 94.3% | 88.4% | 5.1% | 6.8% |
| en_US_General_Telecom_1586891 | telecom | 93.6% | 88.9% | 4.4% | 13.0% |

## Notes
- **Raw WER**: lowercase + strip punctuation only (conservative, no number normalisation).
- **Normalised WER**: Whisper `EnglishTextNormalizer` — canonicalises numbers,
  contractions, British/American spelling, and strips disfluency annotations `(uh)`, `(um)`, `word~`.
- Accuracy = 1 − WER. Target is ≥ 90% normalised accuracy.
- Weighted by reference word count across calls within each aggregate.