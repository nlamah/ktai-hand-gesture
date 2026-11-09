# Review of the GH-handrug round — 31 October 2026

Found 11 confirmed confusion captures from 19:18:39 to 19:22:43 (Europe/Amsterdam). All belong to round `20261031T181714Z_df8bf30f`, recorded name **GH-handrug**, round 1, with the same G/H model (SHA256 `1f5a52da8601d3650e388243e70550d26f65883db9c22a2fb5f31e554d0bfc57`). All image checksums were verified and all saved G/H predictions replayed exactly. No model was changed or trained during this review. This round contains no new H-labeled image.

![Hand crops with joint angles](hand_crops.png)

## Which confusions?

| Intended | G/H rotation | Joint angles | All-letter tilt |
|---|---|---|---|
| G | H | A | A |
| C | D | D | D |
| F | B | B | B |
| K | H | K | K |
| N | M | M | M |
| O | D | D | O |
| P | H | M | M |
| S | E | E | E |
| T | L | L | L |
| X | H | X | X |
| Y | A | A | A |

These are deliberately selected errors: 0/11 correct G/H-model predictions is not an estimate of live accuracy. The comparison uses the same stored landmarks without redetection.

## Belangrijkste bevindingen

**G: the middle finger is measured as bent.** Its 3D PIP angle is 95.3°, compared with training medians of 86.7° for G and 166.2° for H. This signal fits G better. However, the index finger 3D PIP angle is 134.9° versus a G training median of 172.6°. In the 2D pixel projection it is 174.5°, explaining why visible landmarks appear straighter than the added 3D angle feature. This does not prove the 3D angle is wrong: perspective and projection change angles. It motivates testing both representations together. G ranks second in the G/H model. Decision scores are not probabilities.

**K and X: regressions from G/H augmentation.** The ordinary Joint angles model recognizes both correctly, while the G/H model predicts H for both. Large rotations have adversely changed decision boundaries here. For K, index and middle fingers are measured as extended (178.5° and 172.8°), so flexion alone does not distinguish K and H. Thumb position and orientation remain relevant.

**F: visible contact does not produce F.** Thumb and index finger meet in the photo. The computed tip distance is 0.13 times wrist-to-middle-MCP length (a palm scale), yet all five models predict B. Index PIP is 2.8° in pixel projection versus 146.3° in the 3D estimate. For folded/overlapping fingers, projected angles are not physical joint angles: this is a representation difference, not evidence of a coding error.

**C and O: G/H rotations do not resolve these errors.** C remains D and O remains D. The earlier all-letter tilt model recognizes this O. The five nearest original training examples in the full Joint angles input space are all C for C; the five nearest using only standardized joint angles are all O for O. This is a diagnostic distance comparison, not evidence that a nearest-neighbor classifier would generalize better.

**N/M, S/E and T/L:** these confusions also occur without G/H augmentation. Bent fingers have similar angles; thumb position and contact are potential distinguishing signals. For this T image, all five nearest full-input training examples are T, while the SVM predicts L.

**P/H and Y/A:** for P, index and middle fingers are measured as less extended in 3D (139° and 145°) than in pixel projection (179° and 176°). Full-input training neighbors are N; angle neighbors are H. For Y, all five nearest training neighbors are Y in both spaces, but models predict M or A. More G/H rotations therefore seem unlikely to provide a general solution.

## Proposed next steps

1. **Use Joint angles as the provisional reference and keep G/H rotation experimental.** New K/X errors and slightly lower five-fold scores do not support generally preferring the G/H model.

2. **Test projected 2D flexion alongside existing 3D angles.** Calculate in true pixel proportions, adding PIP/DIP angles and tip-to-knuckle/finger-length ratios. These may represent the straight G index finger and folded F index finger more explicitly. 2D measurements also depend on viewpoint; treat them as additional signals, not anatomical truth. Compare on the same five folds, learning normalization on training partitions only.

3. **Investigate a small trained G/H hand-shape classifier.** Middle finger flexion is useful here. A second stage for G/H-ambiguous cases might target it. A hard rule from one capture or general H-to-G correction could damage K/X/P. Check stage activation and all 26 letters.

4. **Test explicit contact/thumb features separately.** F/B concerns thumb-index contact; N/M/S/T concerns the thumb relative to folded fingers. Shin already contains pairwise distances, so this is a more targeted representation rather than guaranteed new information. The earlier full categorical extension was inconsistent; use small targeted ablations.

5. **Collect a fixed G/H round with correct and incorrect attempts.** Include both letters, palm/back/intermediate views, repeated attempts and preferably a second person. Use selected errors for development; reserve a new round after fixing the approach. Real additional images may help if 3D landmarks systematically change with viewpoint, since synthetic feature rotations do not simulate detector behavior.

**First recommended experiment:** Joint angles plus projected 2D finger flexion, without large G/H rotations, followed by a separate contact ablation. This targets the measurement difference visible in these images.

## Reproducibility

`analysis.json` contains model hashes, capture IDs, training percentiles, 2D/3D angles and training neighbors. `model_comparison.csv` contains replayed predictions for all five models. `hand_crops.png` shows only hand crops from the eleven captures. Rerun the numerical analysis with `MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.review_GH_captures`. This review includes AI-generated interpretations; causal hypotheses are not independently validated.