# Eight feature representations and grouped permutation

D: 190 distances; S: 630 Shin directions; E: 70 one-hot values for 19 own checks; J: 15 world-joint measurements.
Eight representations, two learners and five shared filename-block folds on 60,251 accepted original sources. Workbook-match scores, augmentation and hand normalization are excluded.

| Variant | Inputs | SVM precision (%) | SVM F1 | LightGBM precision (%) | LightGBM F1 |
|---|---:|---:|---:|---:|---:|
| DS: Distances + directions | 820 | 97.38815 | 0.97386235 | 96.80215 | 0.96784341 |
| D: Distances only | 190 | 94.42535 | 0.94376340 | 93.63347 | 0.93550772 |
| DSE: DS + own categories | 890 | 97.35679 | 0.97347720 | 96.78134 | 0.96763271 |
| DE: D + own categories | 260 | 94.63529 | 0.94597400 | 93.71895 | 0.93639522 |
| DSJ: DS + world joints | 835 | 97.53902 | 0.97524391 | 96.71271 | 0.96691985 |
| DJ: D + world joints | 205 | 95.05396 | 0.95010689 | 93.83213 | 0.93765908 |
| DSEJ: DS + categories + joints | 905 | 97.38902 | 0.97376260 | 96.77617 | 0.96752529 |
| DEJ: D + categories + joints | 275 | 94.97488 | 0.94926247 | 93.86942 | 0.93791397 |

## Adding/removing information

`factor_effects.csv` retains the effect of adding S/E/J in all four contexts; `paired_changes.csv` counts corrections/losses relative to DS.

## Permutation after fitting

Classification uses all 60,251 sources. Permutation uses 100 held-out sources per letter/fold: 13,000 shared IDs, five fixed shuffles per group/fold. Labels and models remain fixed. All columns in a group move together.
D/S/E/J blocks are tested where present. Individual E checks are tested in DSE/DE; five E families and five J families are tested in DSEJ/DEJ. CSVs retain precision/F1 losses. Figure values are percentage points. Fold/repeat spread is descriptive, not an independent confidence interval; negative losses are retained.

## Interpretation and limitations

Permutation measures fitted-model dependence, not marginal gains from refitting. Related features may substitute, and shuffling may create unusual combinations. The eight training variants answer the complementary adding/removing question.
D/S use image points; E/J use world points. Same-coordinate J controls are reported separately. Coordinate source, scaling and fixed profiles limit interpretation. No independent anatomical ground truth is implied.
Folds were inspected during development and are not verified people. Scores condition on accepted detection. There is no untouched final test or demonstrated webcam gain.

## Evidence

`protocol.json`, `input_verification.json`, `metrics.csv`, `per_letter.csv`, `paired_changes.csv`, `predictions.csv`, `confusions.csv`, `factor_effects.csv`, `same_coordinate_controls.csv`, `permutation_selection.csv`, `permutation.csv`, `permutation_summary.csv`, `verification.json`. Fold models/checkpoints: `runs/feature_factorial_01/`.
