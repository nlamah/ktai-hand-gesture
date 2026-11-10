# Right-hand normalization before feature extraction

Exploratory comparison using the same sources, folds and independently detected RGB views as the mirror study. Detector-Left is reflected to detector-Right; these are not independently verified hand labels.
Primary comparison: accepted original images versus the original 835-input reference. The artificial mixture is a separate distribution.

| Model | Test view | Training/input | Accepted N | Macro-precision (%) | Macro-F1 | Correct |
|---|---|---|---:|---:|---:|---:|
| svm | Original | No change | 60251 | 97.53902 | 0.97524391 | 58894 |
| svm | Original | Mirror training | 60251 | 97.49538 | 0.97492776 | 58868 |
| svm | Original | Right normalization | 60251 | 97.32223 | 0.97319602 | 58758 |
| svm | Mirrored | No change | 60201 | 26.00735 | 0.19248891 | 14027 |
| svm | Mirrored | Mirror training | 60201 | 97.29327 | 0.97250277 | 58676 |
| svm | Mirrored | Right normalization | 60201 | 94.88493 | 0.94802260 | 57239 |
| svm | Fixed mixture | No change | 60225 | 73.56587 | 0.62649875 | 36452 |
| svm | Fixed mixture | Mirror training | 60225 | 97.45079 | 0.97420633 | 58801 |
| svm | Fixed mixture | Right normalization | 60225 | 96.07574 | 0.96047248 | 57987 |
| lightgbm | Original | No change | 60251 | 96.71271 | 0.96691985 | 58454 |
| lightgbm | Original | Mirror training | 60251 | 96.99001 | 0.96961927 | 58581 |
| lightgbm | Original | Right normalization | 60251 | 96.48978 | 0.96459280 | 58304 |
| lightgbm | Mirrored | No change | 60201 | 58.10641 | 0.49977814 | 29779 |
| lightgbm | Mirrored | Mirror training | 60201 | 96.99586 | 0.96940072 | 58488 |
| lightgbm | Mirrored | Right normalization | 60201 | 95.08552 | 0.94959642 | 57322 |
| lightgbm | Fixed mixture | No change | 60225 | 78.73684 | 0.74139200 | 44124 |
| lightgbm | Fixed mixture | Mirror training | 60225 | 97.01177 | 0.96962795 | 58540 |
| lightgbm | Fixed mixture | Right normalization | 60225 | 95.75919 | 0.95678864 | 57798 |

## All attempts, including failed detections

| Model | Test view | Training/input | Attempt N | Macro-precision (%) | Macro-F1 | Correct |
|---|---|---|---:|---:|---:|---:|
| svm | Original | No change | 78000 | 97.53902 | 0.84669832 | 58894 |
| svm | Original | Mirror training | 78000 | 97.49538 | 0.84640358 | 58868 |
| svm | Original | Right normalization | 78000 | 97.32223 | 0.84483115 | 58758 |
| svm | Mirrored | No change | 78000 | 26.00735 | 0.16554426 | 14027 |
| svm | Mirrored | Mirror training | 78000 | 97.29327 | 0.84401102 | 58676 |
| svm | Mirrored | Right normalization | 78000 | 94.88493 | 0.82301442 | 57239 |
| svm | Fixed mixture | No change | 78000 | 73.56587 | 0.53849130 | 36452 |
| svm | Fixed mixture | Mirror training | 78000 | 97.45079 | 0.84559912 | 58801 |
| svm | Fixed mixture | Right normalization | 78000 | 96.07574 | 0.83379477 | 57987 |
| lightgbm | Original | No change | 78000 | 96.71271 | 0.84006268 | 58454 |
| lightgbm | Original | Mirror training | 78000 | 96.99001 | 0.84206142 | 58581 |
| lightgbm | Original | Right normalization | 78000 | 96.48978 | 0.83795342 | 58304 |
| lightgbm | Mirrored | No change | 78000 | 58.10641 | 0.43133157 | 29779 |
| lightgbm | Mirrored | Mirror training | 78000 | 96.99586 | 0.84132148 | 58488 |
| lightgbm | Mirrored | Right normalization | 78000 | 95.08552 | 0.82437168 | 57322 |
| lightgbm | Fixed mixture | No change | 78000 | 78.73684 | 0.64103855 | 44124 |
| lightgbm | Fixed mixture | Mirror training | 78000 | 97.01177 | 0.84176955 | 58540 |
| lightgbm | Fixed mixture | Right normalization | 78000 | 95.75919 | 0.83093481 | 57798 |

## Paired changes

- svm, original, relative to no change: 157 corrected, 293 lost, net -136.
- svm, original, relative to mirror training: 186 corrected, 296 lost, net -110.
- svm, fixed mixture, relative to no change: 21,850 corrected, 315 lost, net +21535.
- svm, fixed mixture, relative to mirror training: 253 corrected, 1,067 lost, net -814.
- lightgbm, original, relative to no change: 285 corrected, 435 lost, net -150.
- lightgbm, original, relative to mirror training: 283 corrected, 560 lost, net -277.
- lightgbm, fixed mixture, relative to no change: 14,032 corrected, 358 lost, net +13674.
- lightgbm, fixed mixture, relative to mirror training: 268 corrected, 1,010 lost, net -742.

## Scope

Primary scores share accepted test images within each presentation. Secondary scores count all 78,000 attempts: failed detections add false negatives, leaving precision unchanged but changing recall/F1. Normalization does not improve detection itself.
Normalization fits one row per accepted original source; mirror training adds RGB derivatives. Detector hand labels are unverified. Reused filename folds and mirror pairs are not independent new participants.
Live normalization applies before feature extraction only when the selected model bundle requests it. Deployment fits are not independent tests.

Evidence: `protocol.json`, `metrics.csv`, `per_letter.csv`, `handedness_metrics.csv`, `paired_changes.csv`, `predictions.csv`, `geometry_verification.json`, `verification.json`.
