# Fixed ±15° direction augmentation experiment

One predefined candidate: original inputs plus -15°/+15° rotations about the camera-x axis. Only 630 axis-direction features change; 190 normalized distances and 15 joint-angle values stay fixed. All 26 Kaggle classes are augmented. Each of the three rows has sample weight 1/3, preserving total fitting weight per original. The fixed SVM profile is C=10, RBF, gamma=scale, balanced. No webcam captures enter training.

This is rotation of normalized landmark-feature coordinates, not a photorealistic wrist, perspective or detector simulation. No other angle/axis/model variant was searched. Predefined settings and source hashes are in `protocol.json`.

## Matched Kaggle five-fold comparison

| Model | Accuracy | Macro-F1 | Correct / accepted images |
|---|---:|---:|---:|
| Baseline Shin + joint angles | 97.74776% | 0.975244 | 58894/60251 |
| ±15° augmentation | 97.70294% | 0.974686 | 58867/60251 |

Net correct change: -27; 57 formerly incorrect images recovered, 84 formerly correct images lost. Accuracy delta: -0.04481 percentage points. Four folds decline and one improves.

The same frozen filename-group partitions and validation IDs are used. Augmentation is applied after the split to fitting images only; imputation/scaling are fitted on the fitting partition. There are 60,251 OOF predictions per variant, not 180,753 validation observations. Groups are not verified signers; these remain exploratory detection-conditioned scores, not a paper replication or live accuracy estimate.

## Development webcam round: 27 C/D/O images

| Model | Correct | Accuracy | Macro-F1 over C/D/O |
|---|---:|---:|---:|
| Baseline | 17/27 | 62.96% | 0.610035 |
| ±15° augmentation | 18/27 | 66.67% | 0.642439 |

| True letter | Baseline correct | Augmented correct |
|---|---:|---:|
| C | 3/9 | 3/9 |
| D | 8/9 | 8/9 |
| O | 6/9 | 7/9 |

One additional O is recognized; C remains unchanged. One wrong C prediction changes from D to O, so O precision decreases even though O recall improves. This is a small development-set change, not evidence of general improvement.

## Other inspected examples and evaluation status

On the ten previously inspected, manually selected confusions, nine predictions stay unchanged; the friend’s O changes from D to O. This is descriptive evidence from a selected known example, not an independent test.

The evaluation directory contains 0/27 finalized records; no evaluation scores were computed. The model and recipe were fixed before checking the evaluation status.

## Try live

Close the existing camera window first. From the repository root:

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_tilt15.joblib
```

Baseline comparison:

```bash
python -m research.live_camera --model models/asl_shin_angles.joblib
```

Q/Esc closes the native camera window. Snapshots retain the selected model hash. The default model is still the original baseline. This candidate is available for testing; results do not justify replacing the baseline as a generally better model.

## Reproduce

```bash
python -m research.studies.tilt_augmentation --stage train
python -m research.studies.tilt_augmentation --stage cv
python -m research.studies.tilt_augmentation --stage evaluate
python -m research.studies.report_tilt
```

The train command refuses to overwrite the candidate. CV checkpoints can be resumed under the same frozen experiment configuration. To run a separate experiment, provide fresh `--out` and `--candidate` paths. Input and model hashes must match. After a complete reserved evaluation round is collected, rerunning `--stage evaluate` computes its paired result without retraining.
