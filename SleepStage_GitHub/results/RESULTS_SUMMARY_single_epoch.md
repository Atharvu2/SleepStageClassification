# Executed Results Summary

## Data processed
- Participants: 5 (SLP001–SLP005)
- EDF recordings: 14 available recordings
- Labelled epochs: 13,628
- Feature matrix: 13,628 × 22
- Evaluation: Leave-One-Subject-Out cross-validation; every epoch from the held-out participant is excluded from training.

## Pooled held-out-subject model scores
| Model | Accuracy | Macro-F1 | Weighted-F1 | Cohen's kappa |
|---|---:|---:|---:|---:|
| RandomForest | 0.8069 | 0.7065 | 0.7949 | 0.7359 |
| SVM_RBF | 0.7741 | 0.7163 | 0.7868 | 0.7026 |
| KNN | 0.7923 | 0.6821 | 0.7777 | 0.7164 |
| ExtraTrees | 0.7940 | 0.7024 | 0.7883 | 0.7220 |
| XGBoost | 0.7636 | 0.7066 | 0.7776 | 0.6890 |

Random Forest has the highest accuracy and weighted-F1 among these runs; RBF SVM has the highest macro-F1. XGBoost was run with a modest 25-tree configuration to keep the pilot computationally lightweight. Its performance should not be interpreted as a tuned or best-possible XGBoost result.

## Important caveats
- These are outputs from the logged experiments in this project; they are not externally validated benchmark scores.
- Only five participants were used; results are preliminary and uncertain.
- Majority voting across expert CSV labels suppresses scorer disagreement.
- Only one EEG channel (C4-A1 where available) and 22 hand-engineered features were used.
- CNN/CNN-LSTM scripts are optional, but deep-learning results have **not** been run and are not included.
- Do not claim clinical readiness or generalisability from this pilot.
