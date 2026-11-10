# Standalone G/H y180 augmentation

Question: does adding 180-degree camera-y turns to G/H fitting examples improve the 26-letter SVM, without mirror augmentation, camera calibration or routing rules?

Both arms use the same 60,251 historical accepted original images and frozen five filename-block folds. The source population contains 2,430 G and 2,371 H examples. Only G/H training directions change: `(dx,dy,dz) -> (-dx,dy,-dz)`. Originals are retained. G/H originals and turns have weight 0.5 each; other sources have weight one. The full fit has 65,052 rows. Scaling, resolved gamma and balanced class priors are learned from original fitting rows and shared. Both models are ordinary 26-class SVM pipelines.

Validation uses original images only, never transformed fitting siblings. A freshly fitted baseline exactly reproduces all 60,251 historical world-joint OOF predictions. No source images or camera captures are added.

## Pooled five-fold results

| Training | Accuracy (%) | Macro-precision (%) | Macro-F1 | Correct |
|---|---:|---:|---:|---:|
| Original only | 97.74776 | 97.53902 | 0.97524391 | 58894 / 60251 |
| Original + G/H y180 | 97.74610 | 97.53874 | 0.97523305 | 58893 / 60251 |

Paired changes: **4 fixed, 5 lost, net -1**; 13 other wrong-to-wrong changes. Scores condition on successful historical landmark extraction (60,251 of 78,000 source images). Macro scores average A–Z; this is not a binary G/H task.

| Letter | Reference P | Candidate P | Reference R | Candidate R | Reference F1 | Candidate F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.980036 | 0.980036 | 0.987203 | 0.987203 | 0.983607 | 0.983607 |
| B | 0.991430 | 0.991430 | 0.997278 | 0.997278 | 0.994345 | 0.994345 |
| C | 0.990659 | 0.990659 | 0.998431 | 0.998431 | 0.994530 | 0.994530 |
| D | 0.988056 | 0.988051 | 0.976792 | 0.976384 | 0.982391 | 0.982183 |
| E | 0.985794 | 0.985794 | 0.992201 | 0.992201 | 0.988987 | 0.988987 |
| F | 0.992603 | 0.992254 | 0.985659 | 0.985659 | 0.989119 | 0.988945 |
| G | 0.989353 | 0.987735 | 0.994239 | 0.994239 | 0.991790 | 0.990976 |
| H | 0.986441 | 0.986023 | 0.981864 | 0.981864 | 0.984147 | 0.983939 |
| I | 0.974609 | 0.973373 | 0.967647 | 0.967647 | 0.971115 | 0.970501 |
| J | 0.981883 | 0.981883 | 0.967780 | 0.967780 | 0.974780 | 0.974780 |
| K | 0.991788 | 0.991788 | 0.983710 | 0.983710 | 0.987732 | 0.987732 |
| L | 0.998009 | 0.998804 | 0.996025 | 0.996025 | 0.997016 | 0.997413 |
| M | 0.891107 | 0.891581 | 0.919476 | 0.918851 | 0.905069 | 0.905011 |
| N | 0.913220 | 0.913961 | 0.879687 | 0.879687 | 0.896140 | 0.896497 |
| O | 0.967263 | 0.967249 | 0.978799 | 0.978357 | 0.972997 | 0.972771 |
| P | 0.978166 | 0.978166 | 0.987751 | 0.987751 | 0.982935 | 0.982935 |
| Q | 0.989423 | 0.988958 | 0.977208 | 0.978158 | 0.983278 | 0.983528 |
| R | 0.968737 | 0.968737 | 0.938262 | 0.938262 | 0.953256 | 0.953256 |
| S | 0.971451 | 0.971826 | 0.983978 | 0.983978 | 0.977674 | 0.977864 |
| T | 0.990598 | 0.990598 | 0.984707 | 0.984707 | 0.987644 | 0.987644 |
| U | 0.921063 | 0.921063 | 0.965778 | 0.965778 | 0.942890 | 0.942890 |
| V | 0.983170 | 0.983170 | 0.984326 | 0.984326 | 0.983748 | 0.983748 |
| W | 0.982107 | 0.983306 | 0.983306 | 0.983306 | 0.982706 | 0.983306 |
| X | 0.968591 | 0.969039 | 0.972184 | 0.972184 | 0.970384 | 0.970609 |
| Y | 0.994955 | 0.994955 | 0.993798 | 0.993798 | 0.994377 | 0.994377 |
| Z | 0.989633 | 0.989633 | 0.977806 | 0.977806 | 0.983684 | 0.983684 |

## Fold changes

| Fold | Reference F1 | Candidate F1 | Fixed | Lost | Net |
|---|---:|---:|---:|---:|---:|
| 1 | 0.98137147 | 0.98149750 | 3 | 1 | +2 |
| 2 | 0.93792419 | 0.93793164 | 1 | 1 | +0 |
| 3 | 0.99164126 | 0.99153971 | 0 | 1 | -1 |
| 4 | 0.97810044 | 0.97800902 | 0 | 1 | -1 |
| 5 | 0.97879909 | 0.97874404 | 0 | 1 | -1 |

## Previously selected G/H camera cases

None enter fitting. These cases informed earlier development, so their replay is diagnostic and cannot estimate unseen back-of-hand accuracy. All 22 saved cases are in `capture_replay.csv`.

| Intended | Reference | G/H y180 |
|---|---|---|
| G | C | H |
| H | C | H |
| G | A | H |
| H | G | G |

## Interpretation and scope

This isolates adding G/H y180 fitting copies on the original-data SVM. It does not measure an interaction with mirroring or the value of camera calibration. The earlier multi-angle G/H recipe and the mirror-plus-calibration candidates are different experiments with separate artifacts. This was requested after earlier development and is exploratory, not a newly untouched final test.

The rotation is exact for the descriptor geometry but operates in normalized landmark coordinates with estimated depth. It cannot reveal hidden fingers or simulate MediaPipe errors in real back views. Good original-view scores do not validate back-view recognition. Filename blocks are not verified signers. Historical detector confidence thresholds are not retained. No significance, new-person accuracy or general webcam robustness is claimed.

Checks passed: 12 aggregate rows, 312 letter rows, 6 paired rows, 260 fold-model replays and 22 camera replays, plus source/model/image hashes, disjoint source groups and original-only preprocessing/weights.

## Reproduce and use live

```bash
python -m research.studies.gh_rotation_standalone --stage freeze
python -m research.studies.gh_rotation_standalone --stage cv
python -m research.studies.gh_rotation_standalone --stage train
python -m research.studies.report_gh_rotation_standalone
python -m research.live_camera --model models/asl_shin_angles_GH_y180.joblib
```
