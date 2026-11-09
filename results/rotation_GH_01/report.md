# G/H direction-rotation augmentation

One fixed experiment augments only the accepted G/H training examples with camera-y rotations of -90, +90 and 180 degrees, alongside the original. The 630 direction features rotate; 190 distances and 15 joint-angle inputs stay unchanged. This is a landmark-feature approximation, not image augmentation or a simulation of hand-back landmark detection.

There are 60,251 original training images and 74,654 fitting rows. Each selected source contributes four rows weighted 0.25; other classes occur once with weight 1. Scaling, resolved RBF gamma and balanced class weights come from original fitting rows. Captures never enter training.

## Frozen five-fold validation

| Model | Accuracy | Macro-F1 | Correct |
|---|---:|---:|---:|
| With joint angles | 97.74776% | 0.97524391 | 58894/60251 |
| G/H rotation | 97.73780% | 0.97516874 | 58888/60251 |

Paired changes: 9 wrong predictions corrected; 15 previously correct predictions lost.

| Letter | Baseline precision | G/H precision | Baseline recall | G/H recall | Baseline F1 | G/H F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 98.00% | 98.01% | 98.72% | 98.81% | 0.9836 | 0.9841 |
| B | 99.14% | 99.14% | 99.73% | 99.73% | 0.9943 | 0.9943 |
| C | 99.07% | 99.17% | 99.84% | 99.84% | 0.9945 | 0.9950 |
| D | 98.81% | 98.81% | 97.68% | 97.76% | 0.9824 | 0.9828 |
| E | 98.58% | 98.58% | 99.22% | 99.22% | 0.9890 | 0.9890 |
| F | 99.26% | 99.37% | 98.57% | 98.53% | 0.9891 | 0.9895 |
| G | 98.94% | 98.65% | 99.42% | 99.42% | 0.9918 | 0.9904 |
| H | 98.64% | 98.47% | 98.19% | 97.93% | 0.9841 | 0.9820 |
| I | 97.46% | 97.21% | 96.76% | 96.76% | 0.9711 | 0.9699 |
| J | 98.19% | 97.96% | 96.78% | 96.78% | 0.9748 | 0.9736 |
| K | 99.18% | 99.18% | 98.37% | 98.37% | 0.9877 | 0.9877 |
| L | 99.80% | 99.88% | 99.60% | 99.56% | 0.9970 | 0.9972 |
| M | 89.11% | 89.21% | 91.95% | 91.89% | 0.9051 | 0.9053 |
| N | 91.32% | 91.40% | 87.97% | 87.97% | 0.8961 | 0.8965 |
| O | 96.73% | 96.81% | 97.88% | 97.84% | 0.9730 | 0.9732 |
| P | 97.82% | 97.91% | 98.78% | 98.78% | 0.9829 | 0.9834 |
| Q | 98.94% | 98.90% | 97.72% | 97.82% | 0.9833 | 0.9835 |
| R | 96.87% | 96.83% | 93.83% | 93.83% | 0.9533 | 0.9531 |
| S | 97.15% | 97.18% | 98.40% | 98.36% | 0.9777 | 0.9777 |
| T | 99.06% | 99.06% | 98.47% | 98.47% | 0.9876 | 0.9876 |
| U | 92.11% | 92.00% | 96.58% | 96.58% | 0.9429 | 0.9423 |
| V | 98.32% | 98.36% | 98.43% | 98.43% | 0.9837 | 0.9839 |
| W | 98.21% | 98.29% | 98.33% | 98.33% | 0.9827 | 0.9831 |
| X | 96.86% | 96.99% | 97.22% | 97.22% | 0.9704 | 0.9711 |
| Y | 99.50% | 99.50% | 99.38% | 99.38% | 0.9944 | 0.9944 |
| Z | 98.96% | 99.01% | 97.78% | 97.74% | 0.9837 | 0.9837 |

## Previously selected webcam errors

These are known, selected failures, useful for diagnosis but not an independent or representative evaluation. The standard guided round contains only C/D/O, not G/H. A new G/H round with different poses and preferably another signer is still needed.

| Intended letter | Joint-angle model | G/H rotation |
|---|---|---|
| C | D | D |
| G | C | H |
| H | C | H |
| J | M | H |
| K | R | H |
| P | N | N |
| Q | N | N |
| S | E | E |
| Z | M | H |
| O | D | D |

## Try live

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_GH_rotation.joblib
```

The initial model caption is G/H rotation. Select it again with button/key 5; All-letter tilt (3) is the separate earlier all-letter model.

## Reproduce

```bash
python -m research.studies.tilt_augmentation --stage train --letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib
python -m research.studies.tilt_augmentation --stage cv --letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib
python -m research.studies.tilt_augmentation --stage evaluate --letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib
python -m research.studies.report_GH_rotation
```

Training refuses to overwrite an existing model. Use fresh candidate/output paths for a fresh run. Historical CSV/JSON candidate fields named tilt15 are retained for compatibility; protocol.json specifies the actual larger G/H rotations. Validation uses original images and frozen filename-block groups, not verified signer groups. This exploratory result does not establish generalization to unseen people or palm/back views.
