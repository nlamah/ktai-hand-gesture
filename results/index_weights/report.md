# Targeted index finger angle weighting: SVM + Shin + 15 angles

Frozen 5 outer folds; 3 inner grouped folds; at most 150 images/class per inner fitting/validation partition. Choose mean inner macro-F1 then accuracy; coordinate order PIP, MCP, DIP, one pass; weights 0.5/1/2. Full outer training and validation; imputation/scaling fitted only on each training partition. Fixed RBF SVM C=10, gamma=scale, balanced. All other angles and Shin unchanged.

Index hypothesis was chosen after inspecting previous results on these same development folds; nested weight selection does not turn them into a fresh independent test. Filename blocks are not known people.

## Conclusion

The highest tested accuracy is 97.753% with internally selected PIP weighting versus 97.748% unweighted: **only three additional correct images out of 60,251**. Macro-F1 changes from 0.97524 to 0.97526, which is not a convincing general improvement. Sixteen errors are corrected while thirteen previously correct predictions are lost.

Doubling PIP by default yields one net additional correct image but worsens O → D (19 → 23). Tuning all three index finger angles reduces O → D (19 → 15) but loses five correct images overall, including N. This is not a good general solution for O/D.

Recommendation: retain **Shin + 15 angles without extra weighting** as the simple follow-up candidate. Seek confirmation on new, person-separated images before further optimization on the same development folds.

| Variant | Accuracy | Macro-F1 | Correct | Fixed / lost versus unweighted |
|---|---:|---:|---:|---:|
| unweighted | 97.748% | 0.97524 | 58894 | 0 / 0 |
| pip_2 | 97.749% | 0.97524 | 58895 | 22 / 21 |
| tuned_pip | 97.753% | 0.97526 | 58897 | 16 / 13 |
| tuned_index | 97.739% | 0.97507 | 58889 | 21 / 26 |

unweighted: all angle weights 1. pip_2: predefined PIP weight of 2. tuned_pip: only PIP selected internally. tuned_index: PIP, then MCP and DIP tuned internally.

## Weights per outer fold

| Fold | Variant | MCP | PIP | DIP |
|---|---|---:|---:|---:|
| 1 | unweighted | 1.0 | 1.0 | 1.0 |
| 1 | pip_2 | 1.0 | 2.0 | 1.0 |
| 1 | tuned_pip | 1.0 | 2.0 | 1.0 |
| 1 | tuned_index | 0.5 | 2.0 | 1.0 |
| 2 | unweighted | 1.0 | 1.0 | 1.0 |
| 2 | pip_2 | 1.0 | 2.0 | 1.0 |
| 2 | tuned_pip | 1.0 | 2.0 | 1.0 |
| 2 | tuned_index | 1.0 | 2.0 | 0.5 |
| 3 | unweighted | 1.0 | 1.0 | 1.0 |
| 3 | pip_2 | 1.0 | 2.0 | 1.0 |
| 3 | tuned_pip | 1.0 | 0.5 | 1.0 |
| 3 | tuned_index | 0.5 | 0.5 | 1.0 |
| 4 | unweighted | 1.0 | 1.0 | 1.0 |
| 4 | pip_2 | 1.0 | 2.0 | 1.0 |
| 4 | tuned_pip | 1.0 | 0.5 | 1.0 |
| 4 | tuned_index | 2.0 | 0.5 | 1.0 |
| 5 | unweighted | 1.0 | 1.0 | 1.0 |
| 5 | pip_2 | 1.0 | 2.0 | 1.0 |
| 5 | tuned_pip | 1.0 | 0.5 | 1.0 |
| 5 | tuned_index | 1.0 | 0.5 | 1.0 |

## Targeted confusions

| True letter → prediction | Unweighted | PIP 2 | Tuned PIP | All index finger angles tuned |
|---|---:|---:|---:|---:|
| O → D | 19 | 23 | 16 | 15 |
| D → O | 41 | 39 | 42 | 39 |
| M → N | 102 | 101 | 102 | 103 |
| N → M | 141 | 143 | 143 | 146 |
| R → U | 142 | 137 | 137 | 136 |
| U → R | 44 | 44 | 44 | 44 |

## Recall per letter

| Letter | Images | Unweighted | PIP 2 | Tuned PIP | All index finger angles tuned |
|---|---:|---:|---:|---:|---:|
| A | 2188 | 98.72% | 98.72% | 98.72% | 98.67% |
| B | 2204 | 99.73% | 99.73% | 99.73% | 99.73% |
| C | 1912 | 99.84% | 99.84% | 99.84% | 99.84% |
| D | 2456 | 97.68% | 97.72% | 97.64% | 97.72% |
| E | 2308 | 99.22% | 99.18% | 99.18% | 99.13% |
| F | 2859 | 98.57% | 98.50% | 98.57% | 98.57% |
| G | 2430 | 99.42% | 99.42% | 99.42% | 99.42% |
| H | 2371 | 98.19% | 98.14% | 98.14% | 98.19% |
| I | 2380 | 96.76% | 96.81% | 96.81% | 96.81% |
| J | 2576 | 96.78% | 96.86% | 96.86% | 96.82% |
| K | 2701 | 98.37% | 98.37% | 98.37% | 98.37% |
| L | 2516 | 99.60% | 99.60% | 99.60% | 99.60% |
| M | 1602 | 91.95% | 92.01% | 91.89% | 91.82% |
| N | 1280 | 87.97% | 87.81% | 87.81% | 87.50% |
| O | 2264 | 97.88% | 97.70% | 97.97% | 98.01% |
| P | 2041 | 98.78% | 98.78% | 98.78% | 98.78% |
| Q | 2106 | 97.72% | 97.86% | 97.67% | 97.58% |
| R | 2543 | 93.83% | 94.02% | 94.02% | 94.06% |
| S | 2559 | 98.40% | 98.28% | 98.32% | 98.32% |
| T | 2354 | 98.47% | 98.47% | 98.51% | 98.51% |
| U | 2513 | 96.58% | 96.62% | 96.62% | 96.58% |
| V | 2552 | 98.43% | 98.43% | 98.43% | 98.39% |
| W | 2456 | 98.33% | 98.33% | 98.33% | 98.37% |
| X | 2157 | 97.22% | 97.22% | 97.22% | 97.22% |
| Y | 2580 | 99.38% | 99.42% | 99.42% | 99.38% |
| Z | 2343 | 97.78% | 97.74% | 97.74% | 97.70% |

Weights are applied after standardization. A factor of 2 contributes a factor of 4 to the corresponding squared-distance component of the RBF kernel. Selection did not use O/D alone, and no post-hoc per-letter outer-fold choice was applied. This study concerns accepted detections; failed detections are unchanged.

```bash
python -m research.studies.tune_index
```
