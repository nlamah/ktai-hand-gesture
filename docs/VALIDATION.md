# Research validation and reproducibility

The audit checks technical evidence for stored landmarks → classification.
It preserves existing datasets, splits, models and reports, writes a new output
directory and performs no detector optimization or classifier search. For the
practical webcam guide, see [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## Read-only audit

Run from the project root in the normal environment:

```bash
python -m scripts.audit_experiments --experiments experiments/shin2021_kaggle_01 experiments/shin2021_finger_kaggle_01 --out exports/research_validation_new --require-shared-splits
```

Optionally add `--paper data/external/shin2021.pdf` with a locally saved full
article. Its source and hash are archived; a missing paper is never invented.
See [PAPER_SHIN2021.md](PAPER_SHIN2021.md) for the mapping, coordinate and
normalization assumptions and unresolved ambiguities. Exact scientific
reproduction is not automatically established.

`AUDIT.md` and `audit.json` contain method, evidence, expected/observed outcome,
status, limitation and next step per check. Statuses are `passed`, `failed` and
`not_checked`; missing evidence is never a pass. Every command requires a new
output directory outside source datasets/experiments.

Checks cover all eight representations, independent known distance/angle
examples, appropriate translation/scale expectations, axis-rotation sensitivity,
invalid values, feature count/order/composition, ID/label/role consistency,
train/validation/test and group separation, OOF multiplicities and shared actual
evaluated mappings. Declared training sets are compared semantically across
experiments, ignoring irrelevant timestamps/model metadata.

Stored RGB hashes are compared across splits; missing hashes trigger image
decoding. Existing hashes are not recomputed from every image. Near-duplicates
and true identities remain unexamined. Filename blocks do not prove verified
people/sessions. Saved folds describe intended membership; they cannot prove
unrecorded historical per-fold fit inputs. That check remains `not_checked`.

A targeted current-code probe records actual scaler fit/transform rows with
unique numeric examples and excludes a held-out marker. Metadata perturbation
checks all input variants. The audit streams all saved final-fit CSV IDs/columns,
checks trusted local joblib hashes, and reconstructs three numeric input rows
when the model's source guard permits it. It does not disable old source guards.

Independent stdlib integer counts recalculate TP/FP/FN, precision/recall/F1,
support, accuracy, confusion, pooled CV, unweighted fold means and sample SD
(`ddof=1`). No sklearn metrics or production reporting helpers are called.
Saved CSV values are compared with absolute tolerance `1e-9`. Unknown/rejected
outputs use `__unknown__`; zero denominators give zero. Development CV uses
all frozen classes. Archived incomplete final-test reports are compared under
their existing truth-present macro policy; the new webcam workflow instead
uses explicitly fixed predeclared stress classes, with zero for missing classes.

## Replay one selected variant

```bash
python -m scripts.reproduce_experiment --experiment experiments/shin2021_finger_kaggle_01/svm --variant finger_features_only --out exports/research_reproduction_new
```

This practical existing run has the required data. It repeats only one variant's
frozen CV and final development fit in a new directory, without predicting a
reserved test. Data/source/library guards remain active. Missing prerequisites
are recorded, and actual execution failures are failed checks. The trace observes
the real `estimator.fit` matrices/labels, compares them with declared training rows
and records scaler fitted-row counts. Identical numeric rows cannot by themselves
distinguish photographs.

Commit, dirty worktree/diff, current Python source hashes, package versions,
thread environment/pools, source data/split hashes, configuration and seeds are
recorded. OOF mappings and independently computed scores are compared; identical
joblib bytes are not required. Before/after source file checks cover sizes/mtimes
and additionally JSON/source hashes, not every large binary's bytes twice.

## Executed by Codex on 2026-10-28

| Local generated output | Observed result |
|---|---|
| `exports/research_validation_01/AUDIT.md` | 250 passed, 37 not checked, no failures; 14 saved variants in three plans |
| `exports/research_reproduction_01/AUDIT.md` | 3 passed: actual fit trace, matching predictions/scores, source files unchanged |
| `exports/research_software_test_02/test_result/report.html` | Synthetic workflow: selected augmentation, analysis, live export and sealed saved test including detection failure |

No saved score discrepancies were found at the configured precision. The shared
assignments and 60,251 usable Kaggle photos were checked. Only the existing
SVM/finger-only variant was replayed. The actual registered webcam baseline was
inspected as SVC with 820 Shin inputs and all 26 classes.

Missing historical fit traces, independent final-test observations, verified
collection identities, near-duplicate review and historical detector metadata
limit conclusions. Older input-code hashes intentionally fail current guards.
These findings do not justify silently replacing old results. Describe Kaggle
scores as exploratory development CV. If a future audit finds a mismatch,
preserve evidence, investigate class policy/averaging/rounding or the source
defect, and regenerate affected derived results under a new name.

New tests deliberately corrupt saved accuracy, duplicate OOF rows, change truth
labels and introduce cross-split pixel duplicates. The audit must detect these
without repairing original evidence. A small replay also checks file preservation.

```bash
python -m unittest discover -s tests -v
python tests/smoke_research.py --out exports/research_software_test_new
```

## Human review and AI use

Codex implemented the workflow/tests and ran the reported automated checks.
The complete suite passed: 126 tests. The 16 research/webcam commands in the
research guide were parsed against the real command adapters. Native capture and
A/B views were rendered with synthetic frames and visually inspected by Codex;
physical camera acquisition was not exercised.
Tests written by Codex are not independent human review. None of these researcher
actions has been confirmed:

- [ ] Recalculate a distance, a 90°/180° joint angle and a signed axis cosine.
- [ ] Compare paper point pairs, coordinates and normalization with the oracle.
- [ ] Trace ID → features → train-only fit → held-out transform → prediction.
- [ ] Compare one independent TP/FP/FN example with a saved report.
- [ ] Investigate failed/unknown audit items and document consequences.
- [ ] Review real group identities, near-duplicates and test independence.
- [ ] Validate physical webcam operation, gesture references/labels and nominal
  viewpoint setup, recording your own actions and dates.

Software correctness, reproducibility and scientific validity are distinct.
Matching predictions can reproduce an invalid evaluation. This audit provides
evidence for the components actually checked, with the stated limits.
