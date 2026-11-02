# Kaggle ASL Alphabet: nested feature-weight tuning

Dataset source used by Shin: [grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet), confirmed in [the paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8434249/). 78,000 letter images, 60,251 accepted detections; all 26 letters. The dataset source matches, but the detector and split do not exactly reproduce the paper.

5 outer and 3 inner StratifiedGroupKFold; session filename blocks, seed 42; one coordinate pass over 0.25/0.5/1/2. Bound internal search to 150 images/class in each fitting and validation partition; outer training and evaluation use all usable images. Choose inner mean macro-F1 then accuracy. Outer validation never used to select weights. Angle imputation/scaling fitted on inner training partitions and refitted on outer training partition.

Same public dataset source as Shin ASL Alphabet experiment, not the exact published split, detector or a replication of published scores. Filename blocks are not verified subjects. Fixed categorical thresholds/Excel proposals came from separate Roboflow development data; no Kaggle-fold threshold tuning. Only letter classes A-Z. The provided 26 test photos duplicated training pixels and were excluded; no independent final test. Coordinate search is bounded and not exhaustive. Current fixed SVM C/gamma not jointly tuned.

## Main findings

- The dataset source is the Kaggle ASL Alphabet dataset cited by Shin, rather than the small Roboflow set used in earlier comparisons.
- The tuned combination is marginally more accurate: 58,848 correct images versus 58,827 for Shin/project (+21) and 58,842 for the Shin/paper profile (+6).
- Unweighted additions yield 58,822 correct images, five fewer than the project baseline. Weighting helps relative to that unweighted combination (+26).
- The highest macro-F1 remains with the pure Shin paper-profile baseline (0.97469); the tuned combination reaches 0.97434. There is no clear gain across all metrics.
- M recall increases from 90.01% to 91.01% (+16 images); S from 98.09% to 98.36% (+7 images).
- R decreases from 93.43% to 93.08% (−9 images), N from 88.13% to 87.58% (−7), and O from 98.19% to 97.88% (−7).
- Selected category weights are 1; 0.25; 0.25; 0.25; 0.5. More weight is not automatically better.

This small difference is not an established statistically reliable general improvement. Groups are six filename blocks, not known people. Search is capped at 150 images per letter per inner subset, but outer training and evaluation use all 60,251 accepted images. The 17,749 failed detections explain the lower score over all 78,000 images.

| Variant | Accepted-only accuracy | Macro-F1 | All-image accuracy |
|---|---:|---:|---:|
| shin_project | 97.637% | 0.97386 | 75.419% |
| shin_paper_profile | 97.661% | 0.97469 | 75.438% |
| combined_unweighted | 97.628% | 0.97372 | 75.413% |
| combined_tuned | 97.671% | 0.97434 | 75.446% |

`shin_project`: C=10, balanced. `shin_paper_profile`: C=1000, no class weighting. Both use RBF, gamma=scale. Combined models use C=10 and balanced weights to isolate feature weighting. Excel scores are calculated for all 26 candidate letters without the true label as input.

## Selected weights per outer fold

| Fold | Angles | Categories | Excel-scores |
|---|---:|---:|---:|
| 1 | 0.5 | 1.0 | 2.0 |
| 2 | 1.0 | 0.25 | 0.25 |
| 3 | 0.5 | 0.25 | 2.0 |
| 4 | 1.0 | 0.25 | 1.0 |
| 5 | 1.0 | 0.5 | 0.25 |

## Recall per letter

| Letter | Images | Shin project | Shin paper profile | Unweighted | Tuned | Tuned − Shin difference |
|---|---:|---:|---:|---:|---:|---:|
| A | 2188 | 98.67% | 98.58% | 98.67% | 98.72% | +0.05 pp |
| B | 2204 | 99.77% | 99.77% | 99.73% | 99.73% | -0.05 pp |
| C | 1912 | 99.84% | 99.84% | 99.84% | 99.84% | +0.00 pp |
| D | 2456 | 97.68% | 97.88% | 97.64% | 97.72% | +0.04 pp |
| E | 2308 | 99.00% | 98.83% | 99.22% | 99.18% | +0.17 pp |
| F | 2859 | 98.60% | 98.53% | 98.57% | 98.57% | -0.03 pp |
| G | 2430 | 99.42% | 99.30% | 99.47% | 99.42% | +0.00 pp |
| H | 2371 | 98.14% | 98.06% | 98.23% | 98.23% | +0.08 pp |
| I | 2380 | 96.68% | 96.85% | 96.72% | 96.68% | +0.00 pp |
| J | 2576 | 96.78% | 96.93% | 96.93% | 96.74% | -0.04 pp |
| K | 2701 | 98.26% | 98.37% | 98.30% | 98.37% | +0.11 pp |
| L | 2516 | 99.64% | 99.64% | 99.56% | 99.64% | +0.00 pp |
| M | 1602 | 90.01% | 93.38% | 90.76% | 91.01% | +1.00 pp |
| N | 1280 | 88.12% | 88.12% | 86.80% | 87.58% | -0.55 pp |
| O | 2264 | 98.19% | 97.97% | 97.48% | 97.88% | -0.31 pp |
| P | 2041 | 98.63% | 98.78% | 98.82% | 98.82% | +0.20 pp |
| Q | 2106 | 97.72% | 97.67% | 97.72% | 97.63% | -0.09 pp |
| R | 2543 | 93.43% | 94.06% | 92.96% | 93.08% | -0.35 pp |
| S | 2559 | 98.09% | 98.12% | 98.40% | 98.36% | +0.27 pp |
| T | 2354 | 98.43% | 98.22% | 98.43% | 98.43% | +0.00 pp |
| U | 2513 | 96.70% | 95.66% | 96.54% | 96.62% | -0.08 pp |
| V | 2552 | 98.24% | 97.92% | 98.24% | 98.39% | +0.16 pp |
| W | 2456 | 98.25% | 98.53% | 98.37% | 98.37% | +0.12 pp |
| X | 2157 | 97.03% | 95.92% | 97.36% | 97.22% | +0.19 pp |
| Y | 2580 | 99.46% | 99.38% | 99.38% | 99.38% | -0.08 pp |
| Z | 2343 | 97.44% | 97.52% | 97.57% | 97.61% | +0.17 pp |

Detailed results: `per_letter.csv`, `fold_metrics.csv`, `inner_search.csv`, `oof_predictions.csv`, `summary.json`. The historical 97.43% used three folds; do not compare it directly with these new five folds. The appropriate comparison is between models on the current shared folds.

## Reproduction

```bash
python -m research.studies.tune_groups \
  --out experiments/kaggle_nested_feature_weights_02
```

The input cache stores manifest, Excel and threshold hashes. Each outer fold has a checkpoint with predictions and inner search results. Checks confirmed 60,251 unique out-of-fold predictions, five disjoint outer folds, fifteen disjoint inner folds, no outer-validation images in inner selection, and recomputed accuracy/macro-F1 matching saved values. All seven targeted preprocessing, weighting and input tests passed.
