# All-letter confusion: SVM with and without joint angles

The same five folds and 60,251 accepted Kaggle images. Rows are true letters; columns are predictions. Counts are out-of-fold, not training predictions.

![Confusion matrices before and after adding angles](confusion_before_after.png)

![Change in confusion](confusion_changes.png)

## Overview by true letter

Each row shows up to three largest incorrect destinations after adding angles. `R: 149 → 142` means the true letter in this row was first predicted as R 149 times and then 142 times. CSV files and matrices contain all destinations.

| True letter | Images | Correct Shin → angles | Largest confusions after angles (Shin → angles) |
|---|---:|---:|---|
| A | 2188 | 2159 → 2160 | X: 9 → 8; T: 7 → 7; O: 6 → 6 |
| B | 2204 | 2199 → 2198 | W: 1 → 2; C: 0 → 1; F: 2 → 1 |
| C | 1912 | 1909 → 1909 | L: 2 → 1; O: 1 → 1; Z: 0 → 1 |
| D | 2456 | 2399 → 2399 | O: 35 → 41; F: 5 → 5; I: 5 → 3 |
| E | 2308 | 2285 → 2290 | S: 8 → 6; I: 3 → 3; X: 2 → 3 |
| F | 2859 | 2819 → 2818 | O: 15 → 16; C: 8 → 8; W: 2 → 6 |
| G | 2430 | 2416 → 2416 | J: 9 → 9; H: 5 → 4; F: 0 → 1 |
| H | 2371 | 2327 → 2328 | J: 30 → 29; G: 9 → 10; M: 1 → 1 |
| I | 2380 | 2301 → 2303 | S: 21 → 23; E: 13 → 11; X: 6 → 8 |
| J | 2576 | 2493 → 2493 | I: 28 → 28; H: 16 → 18; R: 14 → 12 |
| K | 2701 | 2654 → 2657 | U: 19 → 21; V: 12 → 10; B: 4 → 4 |
| L | 2516 | 2507 → 2506 | Y: 4 → 4; T: 2 → 3; C: 2 → 2 |
| M | 1602 | 1442 → 1473 | N: 127 → 102; A: 8 → 11; I: 9 → 6 |
| N | 1280 | 1128 → 1126 | M: 140 → 141; A: 4 → 5; B: 2 → 2 |
| O | 2264 | 2223 → 2216 | D: 12 → 19; Y: 9 → 9; A: 3 → 3 |
| P | 2041 | 2013 → 2016 | Q: 20 → 17; H: 4 → 5; N: 1 → 1 |
| Q | 2106 | 2058 → 2058 | P: 40 → 41; H: 2 → 3; W: 3 → 2 |
| R | 2543 | 2376 → 2386 | U: 149 → 142; K: 8 → 7; J: 4 → 4 |
| S | 2559 | 2510 → 2518 | X: 16 → 15; I: 10 → 9; M: 10 → 6 |
| T | 2354 | 2317 → 2318 | A: 14 → 15; E: 3 → 5; Z: 5 → 5 |
| U | 2513 | 2430 → 2427 | R: 46 → 44; V: 15 → 17; W: 5 → 8 |
| V | 2552 | 2507 → 2512 | U: 12 → 13; W: 12 → 13; K: 16 → 9 |
| W | 2456 | 2413 → 2415 | V: 15 → 13; U: 8 → 10; R: 6 → 4 |
| X | 2157 | 2093 → 2097 | U: 13 → 15; Z: 13 → 14; R: 15 → 9 |
| Y | 2580 | 2566 → 2564 | A: 4 → 7; O: 3 → 3; T: 3 → 3 |
| Z | 2343 | 2283 → 2291 | X: 30 → 22; S: 20 → 20; M: 1 → 2 |

## Largest increases in specific errors

| True letter → prediction | Shin | With angles | Difference |
|---|---:|---:|---:|
| O → D | 12 | 19 | +7 |
| D → O | 35 | 41 | +6 |
| F → W | 2 | 6 | +4 |
| F → M | 2 | 5 | +3 |
| M → A | 8 | 11 | +3 |
| U → W | 5 | 8 | +3 |
| Y → A | 4 | 7 | +3 |
| F → B | 0 | 2 | +2 |
| I → P | 0 | 2 | +2 |
| I → S | 21 | 23 | +2 |

## Implications for feature weighting

Upweighting the three index finger angles is a reasonable hypothesis for O/D, but does not follow automatically from the confusion matrix. More weight can amplify landmark errors and move SVM boundaries for other letters. A feature multiplier of 2 contributes a factor of 4 to the squared-distance component of the RBF kernel.

The largest remaining pairs are M/N and R/U. Thumb and joint finger poses are plausible candidates for M/N; lateral order and crossing for R/U. These are hypotheses, not causal conclusions from the matrix. The earlier crossing feature showed no clear additional contribution; inspect landmarks before increasing its weight.

Next targeted test: weight index finger MCP/PIP/DIP as a group by 0.5 / 1 / 2, selected in inner training folds. Evaluate outer folds on overall macro-F1, O/D and M/N and R/U, rather than O alone. Visually inspect the eight newly incorrect O images. These repeatedly used development folds are not an independent person test.

Files: shin_counts.csv, shin_angles_counts.csv, changes_counts.csv en all_error_pairs.csv.

## Index finger: targeted evidence for O/D

Median measured angles in degrees (not tuned thresholds):

| Groep | Images | MCP | PIP | DIP |
|---|---:|---:|---:|---:|
| O | 2264 | 137.6 | 142.1 | 125.6 |
| D | 2456 | 157.2 | 169.0 | 146.8 |
| Nieuw fout: O naar D | 8 | 146.9 | 133.2 | 131.3 |

The PIP median of newly misclassified O images lies on the O side. This makes upweighting PIP alone a testable hypothesis, not necessarily all three index finger angles together. Medians do not show the full overlap or SVM decision boundary, so improvement is uncertain.
