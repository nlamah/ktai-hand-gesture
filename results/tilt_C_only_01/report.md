# Selective C-only tilt augmentation

Only the 1,912 accepted Kaggle C images receive -15/+15-degree camera-x variants of their 630 direction features. The other 58,339 images occur once unchanged. Total fitting rows: 64,075 from 60,251 original images. The 190 distance values and 15 joint angles are unchanged. Captures are not used for training.

## Comparison

| Model | Kaggle 5-fold accuracy | Kaggle macro-F1 | Development correct | Development macro-F1 C/D/O |
|---|---:|---:|---:|---:|
| Baseline | 97.74776% | 0.97524391 | 17/27 | 0.610035 |
| All letters +/-15 degrees | 97.70294% | 0.97468587 | 18/27 | 0.642439 |
| Only C +/-15 degrees | 97.74776% | 0.97524465 | 17/27 | 0.610035 |

## Interpretation

The C-only variant has exactly the same 27 development predictions as the baseline: C 3/9, D 8/9 and O 6/9. No development improvement is observed. The previously inspected friend's O remains D with this variant; the all-letter variant recognized it as O.

Kaggle: 0 incorrect examples recovered and 0 correct examples lost. 3 predictions change from one wrong label to another. Accuracy is identical; the macro-F1 difference (+0.00000074) is practically negligible and reflects redistribution of false positives, not additional correct images.

This is a negative result for this limited C-only +/-15-degree camera-x recipe. It does not show that other axes/ranges, physical image augmentation or different representations cannot help. No other recipes were searched in this run.

## Controls and limitations

Each C source has three rows weighted 1/3 each; other rows have weight1. Selective duplication would otherwise affect class frequency. For this run, imputation/scaling, balanced class weights and the resolved gamma=scale value are fitted on original fitting rows. The SVM is then fitted on the augmented rows. Full-fit class weights and gamma match the baseline exactly. Each CV fold applies this procedure only to its fitting partition; validation rows remain original.

The earlier all-letter run used ordinary gamma=scale on its augmented matrix. Therefore, the C-only run is controlled directly against the baseline; the three-way table is not a strict one-factor ablation of the earlier all-letter run's numeric gamma. This preprocessing policy is recorded in protocol.json.

The angle transformation is a rotation in normalized landmark-feature space, not a complete physical wrist/perspective/detector simulation. Kaggle groups are filename blocks, not verified people. Development results use only 27 examples of one signer and three letters. The evaluation session remains incomplete (0/27), so no independent webcam score is available.

## Live test

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_C_tilt15.joblib
```

Close the previous camera window first. The default model is unchanged. This candidate is for comparison, not a demonstrated improvement.

## Reproduce

```bash
python -m research.studies.tilt_augmentation --stage train --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib
python -m research.studies.tilt_augmentation --stage cv --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib
python -m research.studies.tilt_augmentation --stage evaluate --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib
python -m research.studies.report_C_tilt
```

The existing candidate is protected against overwrite; use fresh --candidate and --out paths for another fit. Matched OOF predictions, capture predictions and all-letter precision/recall/F1 are retained beside this report.
