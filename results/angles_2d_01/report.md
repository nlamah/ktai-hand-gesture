# Projected 2D angles alongside 3D joint angles

One fixed experiment adds 14 internal joint angles measured in aspect-correct image pixels to the original 835-input Vingerhoeken model. The candidate has 849 inputs: 820 unchanged Shin features, 15 original 3D measurements and 14 projected angles. There is no augmentation and no new contact input. The same training-only median imputation/standardization is applied to the 29 angle inputs. The classifier remains balanced RBF SVM C=10, gamma=scale. Captures never enter training.

## Frozen five-fold comparison

| Model | Accuracy | Macro-F1 | Correct |
|---|---:|---:|---:|
| Vingerhoeken | 97.74776% | 0.97524391 | 58894/60251 |
| Beeld + diepte | 97.69298% | 0.97465145 | 58861/60251 |

Paired changes: 70 wrong predictions fixed; 103 previously correct predictions lost.

| Fold | Vingerhoeken accuracy | Beeld + diepte accuracy | Vingerhoeken F1 | Beeld + diepte F1 |
|---|---:|---:|---:|---:|
| 1 | 98.208% | 98.134% | 0.98137 | 0.98084 |
| 2 | 94.525% | 94.455% | 0.93792 | 0.93732 |
| 3 | 99.219% | 99.096% | 0.99164 | 0.99006 |
| 4 | 98.029% | 98.001% | 0.97810 | 0.97742 |
| 5 | 98.165% | 98.190% | 0.97880 | 0.97930 |

### Choosing the number of folds

Five folds retain the existing comparison protocol and avoid selecting a favorable split. The groups are filename blocks, not verified people. Most fits use approximately 80% of images, but group sizes make the partitions uneven. Increasing to ten folds would not establish generalization to new signers. A held-out signer/session is a more direct test of webcam robustness. The main scores pool all out-of-fold predictions; fold means/standard deviations describe split variation and are not confidence intervals.

Fold accuracy mean ± sample SD: baseline 0.976294 ± 0.017991; candidate 0.975753 ± 0.017973.
Fold macro_f1 mean ± sample SD: baseline 0.973567 ± 0.020651; candidate 0.972988 ± 0.020524.

## All letters

| Letter | Baseline recall | Candidate recall | Baseline F1 | Candidate F1 |
|---|---:|---:|---:|---:|
| A | 98.720% | 98.583% | 0.98361 | 0.98224 |
| B | 99.728% | 99.773% | 0.99435 | 0.99502 |
| C | 99.843% | 99.843% | 0.99453 | 0.99453 |
| D | 97.679% | 97.883% | 0.98239 | 0.98444 |
| E | 99.220% | 98.917% | 0.98899 | 0.98490 |
| F | 98.566% | 98.566% | 0.98912 | 0.98929 |
| G | 99.424% | 99.465% | 0.99179 | 0.99139 |
| H | 98.186% | 98.186% | 0.98415 | 0.98477 |
| I | 96.765% | 96.639% | 0.97112 | 0.96944 |
| J | 96.778% | 96.817% | 0.97478 | 0.97384 |
| K | 98.371% | 98.334% | 0.98773 | 0.98810 |
| L | 99.603% | 99.523% | 0.99702 | 0.99682 |
| M | 91.948% | 91.136% | 0.90507 | 0.90179 |
| N | 87.969% | 88.125% | 0.89614 | 0.89488 |
| O | 97.880% | 97.880% | 0.97300 | 0.97300 |
| P | 98.775% | 98.726% | 0.98294 | 0.98365 |
| Q | 97.721% | 97.816% | 0.98328 | 0.98329 |
| R | 93.826% | 93.472% | 0.95326 | 0.95271 |
| S | 98.398% | 98.437% | 0.97767 | 0.97674 |
| T | 98.471% | 98.428% | 0.98764 | 0.98638 |
| U | 96.578% | 96.498% | 0.94289 | 0.94010 |
| V | 98.433% | 98.433% | 0.98375 | 0.98413 |
| W | 98.331% | 98.290% | 0.98271 | 0.98250 |
| X | 97.218% | 97.265% | 0.97038 | 0.97017 |
| Y | 99.380% | 99.341% | 0.99438 | 0.99380 |
| Z | 97.781% | 97.653% | 0.98368 | 0.98303 |

## Existing webcam development examples

The most recent GH-handrug round contains 11 deliberately selected errors from the G/H-rotation model, including other letters. Replaying these is diagnostic, not a representative or independent accuracy estimate. There are no new H-labeled examples in that round.

| Intended | Saved G/H prediction | Vingerhoeken | Beeld + diepte |
|---|---|---|---|
| G | H | A | A |
| C | D | D | D |
| F | B | B | B |
| K | H | K | K |
| N | M | M | M |
| O | D | D | D |
| P | H | M | M |
| S | E | E | E |
| T | L | L | T |
| X | H | X | X |
| Y | A | A | M |

Correct on these 11 known selected examples: Vingerhoeken 2/11; Beeld + diepte 3/11.

Earlier guided C/D/O development round: Vingerhoeken 17/27 (macro-F1 C/D/O 0.610035); Beeld + diepte 16/27 (macro-F1 C/D/O 0.570820). These are reused development images. There is no completed held-out webcam evaluation.

## Live comparison

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_2d.joblib --round-name "2D-3D-test"
```

Select **Beeld + diepte [6]** in the native tool, or **Vingerhoeken [2]** to compare. Snapshot exports retain 849 feature names/values, separate 3D and projected-angle dictionaries, model hash and round identity.

## Reproduce

```bash
python -m research.studies.compare_image_angles --stage all
python -m research.studies.report_image_angles
```

The all stage resumes the fixed protocol and protected candidate, caches only the 14 new angles, and reuses completed fold predictions. A direct train stage refuses to overwrite an existing candidate. Training geometry comes from the original frozen manifest; live geometry uses the same feature function and detector-image dimensions. View-dependent 2D angles cannot be interpreted as ground-truth anatomy. This run tests one representation with fixed settings, not all possible weights or contact features.
