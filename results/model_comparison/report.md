# SVM, LightGBM and the contribution of additional features

The same 60,251 accepted Kaggle ASL Alphabet images and five grouped outer folds as previous experiments. Shin uses 820 values; the combined variant adds 70 category indicators, 26 Excel compatibility scores and 15 joint angles. Additions are unweighted here. The 17,749 failed detections are outside the classifier scores below.

## Model comparison

| Model | Shin accuracy | Met extra features | Difference | Shin macro-F1 | Met extra macro-F1 |
|---|---:|---:|---:|---:|---:|
| svm | 97.637% | 97.628% | -0.008 pp | 0.97386 | 0.97372 |
| lightgbm | 97.097% | 97.164% | +0.066 pp | 0.96784 | 0.96860 |

SVM: RBF, C=10, gamma=scale, balanced. LightGBM: 100 boosting rounds, 31 leaves, learning rate 0.1, minimum 20 examples per leaf, seed 42; other project settings fixed. LightGBM uses four threads for remaining fits, with deterministic and force_col_wise enabled. No new hyperparameter search was performed. Angle imputation and scaling are fitted on training data only.

## Permutation importance

Five repeats per group and fold, capped at 100 validation images per letter per fold. Values below are mean macro-F1 decreases in percentage points. Positive means worse after shuffling; negative means better. Group columns are shuffled jointly; both models use the same permutations and samples. Dispersion is in the CSV.

| Gehusselde groep | SVM: macro-F1-daling | LightGBM: macro-F1-daling |
|---|---:|---:|
| all_extra | +1.263 pp | +0.290 pp |
| categories | +0.049 pp | +0.055 pp |
| angles | +0.570 pp | +0.031 pp |
| excel_scores | +0.008 pp | +0.200 pp |
| thumb | +0.129 pp | +0.014 pp |
| finger_flexion | +0.499 pp | +0.017 pp |
| spread | -0.002 pp | +0.055 pp |
| crossing | +0.008 pp | +0.000 pp |
| contact | +0.000 pp | +0.000 pp |

`all_extra` shuffles all added features together. Other groups overlap, so their importances cannot be added. `thumb` includes thumb categories and angles; `finger_flexion` includes other flexion categories and angles. Excel scores derive from categories. Redundancy can depress importance; impossible shuffled combinations can inflate it. Importance measures model reliance, not independent additional information.

## Ablation: retrain without a feature group

All validation images are used. The table gives pooled accuracy after removal. Delta is relative to the full combined model: positive means improvement after omission.

| Weggelaten groep | SVM accuracy | SVM delta | LightGBM accuracy | LightGBM delta |
|---|---:|---:|---:|---:|
| angles | 97.593% | -0.035 pp | 97.094% | -0.070 pp |
| categories | 97.744% | +0.116 pp | 97.129% | -0.035 pp |
| excel_scores | 97.632% | +0.003 pp | 97.069% | -0.095 pp |

The three groups are omitted individually; all other inputs remain. Removing a group may show little effect when another group supplies the same information. Per-fold macro-F1 and per-letter recall changes are in the detailed files.

## Per-letter recall

| Letter | Images | SVM Shin | SVM extra | LightGBM Shin | LightGBM extra | LightGBM winst door extra |
|---|---:|---:|---:|---:|---:|---:|
| A | 2188 | 98.67% | 98.67% | 98.03% | 97.90% | -0.14 pp |
| B | 2204 | 99.77% | 99.73% | 99.18% | 99.64% | +0.45 pp |
| C | 1912 | 99.84% | 99.84% | 98.74% | 98.95% | +0.21 pp |
| D | 2456 | 97.68% | 97.64% | 97.19% | 97.15% | -0.04 pp |
| E | 2308 | 99.00% | 99.22% | 98.53% | 98.48% | -0.04 pp |
| F | 2859 | 98.60% | 98.57% | 98.36% | 98.67% | +0.31 pp |
| G | 2430 | 99.42% | 99.47% | 98.89% | 98.89% | +0.00 pp |
| H | 2371 | 98.14% | 98.23% | 97.34% | 97.26% | -0.08 pp |
| I | 2380 | 96.68% | 96.72% | 96.34% | 96.43% | +0.08 pp |
| J | 2576 | 96.78% | 96.93% | 96.97% | 96.93% | -0.04 pp |
| K | 2701 | 98.26% | 98.30% | 97.89% | 97.85% | -0.04 pp |
| L | 2516 | 99.64% | 99.56% | 98.45% | 98.17% | -0.28 pp |
| M | 1602 | 90.01% | 90.76% | 88.45% | 88.70% | +0.25 pp |
| N | 1280 | 88.12% | 86.80% | 85.16% | 85.70% | +0.55 pp |
| O | 2264 | 98.19% | 97.48% | 93.90% | 94.39% | +0.49 pp |
| P | 2041 | 98.63% | 98.82% | 98.48% | 98.68% | +0.20 pp |
| Q | 2106 | 97.72% | 97.72% | 98.53% | 98.34% | -0.19 pp |
| R | 2543 | 93.43% | 92.96% | 95.24% | 95.44% | +0.20 pp |
| S | 2559 | 98.09% | 98.40% | 97.73% | 97.69% | -0.04 pp |
| T | 2354 | 98.43% | 98.43% | 98.05% | 98.13% | +0.08 pp |
| U | 2513 | 96.70% | 96.54% | 96.74% | 96.74% | +0.00 pp |
| V | 2552 | 98.24% | 98.24% | 97.73% | 97.81% | +0.08 pp |
| W | 2456 | 98.25% | 98.37% | 98.37% | 98.57% | +0.20 pp |
| X | 2157 | 97.03% | 97.36% | 96.89% | 97.13% | +0.23 pp |
| Y | 2580 | 99.46% | 99.38% | 99.42% | 99.30% | -0.12 pp |
| Z | 2343 | 97.44% | 97.57% | 95.69% | 95.43% | -0.26 pp |

## Interpretation and limitations

Combine three kinds of evidence: gain over Shin, loss after shuffling, and loss after retraining without the group. High permutation importance alone does not establish added value over Shin. Evaluation uses six filename blocks, not verified people; it is development research. References and thresholds were selected earlier on the separate Roboflow set. There is no independent webcam or person test. Standard deviations over folds and repeats are not independent confidence intervals.

Raw results: `model_metrics.csv`, `per_letter.csv`, `permutation_summary_r5_cap100.csv`, `permutation_letters_r5_cap100.csv`, `ablation.csv`, `ablation_letters.csv`, `oof_predictions.csv`. Model-specific files are retained. For current reproduction commands, see `README.md`, sections 4–5.

## Follow-up candidate: SVM without standalone categories

Shin + 15 continuous angles + 26 Excel scores (861 inputs) achieves 58,892/60,251 correct: **97.744%**. That is 65 more correct images than Shin/project (+0.108 percentage points), and 70 more than the full unweighted combination. The 70 explicit one-hot categories are omitted; Excel scores indirectly retain categorical information.

| Letter | Shin recall | Kandidaat recall | Extra correcte beelden |
|---|---:|---:|---:|
| M | 90.01% | 91.64% | +26 |
| R | 93.43% | 93.87% | +11 |
| S | 98.09% | 98.40% | +8 |
| Z | 97.44% | 97.78% | +8 |
| O | 98.19% | 97.92% | -6 |
| N | 88.12% | 87.89% | -3 |
| U | 96.70% | 96.62% | -2 |

This candidate was selected after inspecting ablation variants and requires confirmation on new independent images. This is an accuracy result; stored ablations contain per-fold macro-F1, not pooled macro-F1. All letters are in `candidate_without_categories_per_letter.csv`.
