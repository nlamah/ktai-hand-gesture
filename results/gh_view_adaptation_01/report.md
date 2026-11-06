# Index-axis G/H roll adaptation

Development experiment using existing Kaggle data and three already saved back-of-hand images. Those three images are calibration **training**, not held-out webcam evidence.

Transformation: 180 degrees around each image-space index MCP-to-tip axis; preserves finger direction; G/H only.

The matched reference is the mirror-trained 26-letter SVM. Original source folds and the fixed 50/50 presentation mix are unchanged. G/H-only rotation copies preserve total source weight; each of the three calibration sources has total weight 20. Both arms use the same preprocessing, gamma and class weights. New G/H predictions require the fixed horizontal-index gate. All other proposals retain the reference prediction.

## All 26 letters

| Presentation | Scope | Reference precision | Candidate precision | Reference F1 | Candidate F1 | Fixed / lost |
|---|---|---:|---:|---:|---:|---:|
| original | all_attempts | 0.974954 | 0.974929 | 0.846404 | 0.846371 | 5 / 7 |
| original | accepted_only | 0.974954 | 0.974929 | 0.974928 | 0.974895 | 5 / 7 |
| mirrored_rgb | all_attempts | 0.972933 | 0.973049 | 0.844011 | 0.843990 | 10 / 11 |
| mirrored_rgb | accepted_only | 0.972933 | 0.973049 | 0.972503 | 0.972548 | 10 / 11 |
| balanced_mix | all_attempts | 0.974508 | 0.974634 | 0.845599 | 0.845637 | 8 / 5 |
| balanced_mix | accepted_only | 0.974508 | 0.974634 | 0.974206 | 0.974290 | 8 / 5 |

Precision and F1 are macro averages over A–Z. All-attempt scores count failed detections as missed attempts; accepted-only scores condition on a valid detection. These are the dataset’s usual viewpoints, not a new back-of-hand test.

## G and H on the balanced presentation mix

| Letter | Model | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| G | baseline | 0.991939 | 0.820333 | 0.898011 |
| H | baseline | 0.985738 | 0.783333 | 0.872957 |
| G | adapted | 0.988353 | 0.820333 | 0.896539 |
| H | adapted | 0.976388 | 0.785667 | 0.870706 |

Non-G/H attempts predicted as G/H: 41 reference, 68 candidate. A horizontal-pose gate does not eliminate false G/H proposals.

## Previously saved camera images

| Intended | Reference | Candidate | Role |
|---|---|---|---|
| C | D | D | previously_inspected_development_control |
| G | H | G | calibration_training |
| H | P | H | calibration_training |
| J | H | H | previously_inspected_development_control |
| K | K | K | previously_inspected_development_control |
| P | N | N | previously_inspected_development_control |
| Q | N | N | previously_inspected_development_control |
| S | E | E | previously_inspected_development_control |
| Z | Q | Q | previously_inspected_development_control |
| O | D | D | previously_inspected_development_control |
| G | H | G | calibration_training |
| C | D | D | previously_inspected_development_control |
| F | W | W | previously_inspected_development_control |
| K | K | K | previously_inspected_development_control |
| N | M | M | previously_inspected_development_control |
| O | D | D | previously_inspected_development_control |
| P | Q | H | previously_inspected_development_control |
| S | E | E | previously_inspected_development_control |
| T | T | T | previously_inspected_development_control |
| X | X | X | previously_inspected_development_control |
| Y | A | A | previously_inspected_development_control |
| H | G | G | previously_inspected_development_control |

The first three back-view G/H cases used for calibration cannot validate this model. All remaining cases were previously inspected and selected for errors; they are development controls, not a representative test. Rotation and calibration are combined here; this comparison does not isolate their contributions. The earlier G/H rotation candidate also used ±90° rotations; this candidate uses only 180°, a mirror-trained reference and explicit calibration.

## Checks and limits

Verified 72 aggregate rows, 312 per-letter rows, 36 paired rows, 260 held-out predictions per arm, 22 capture replays and 6 calibration re-encodings. Source/model/image hashes, fixed folds, unchanged preprocessing and restricted routing pass. See `verification.json`.

The rotation acts on normalized image-landmark geometry. It cannot reconstruct hidden fingers or improve MediaPipe detections. The original source blocks are not verified signer identities. No independent back-view accuracy or general robustness is claimed. This remains an optional live development model.
