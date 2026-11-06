# Horizontal RGB mirror experiment

This experiment compares original-only training with original + horizontal RGB mirror training. Its primary question is the effect on macro-precision and macro-F1 in a fixed balanced presentation test. Results and generated figures are in [the experiment report](../results/mirror_01/report.md); the paper summarizes the experiment and Appendix L gives the full design.

## Correct reflection and source separation

Reflect the RGB columns, keep the A-Z label, and retain anatomical landmark indices. Do not rotate the image, exchange the thumb/little-finger indices, flip only a handedness label, or mirror only the on-screen preview. Rerun the same detector on the original and reflected pixels independently. Use the returned image/world coordinates and handedness in the same 835-input live encoder. Exact geometric reflection is tested as a control; it is not substituted for actual redetection.

Every original and its derivatives inherit one source filename-block group. Split source groups before selecting fitting examples or augmenting. The two learners use the same accepted original fitting sources; the mirror arm additionally uses accepted reflections of those sources. A failed mirror leaves its original in training. Sources accepted only after mirroring are not added, so the augmentation does not quietly expand the original source population.

Each source retains total fitting weight one. Angle preprocessing is fitted only on original fitting rows and reused; SVM gamma and balanced class weights are held equal between arms. Both SVM and LightGBM use fixed recipes, without outcome-driven tuning.

## Evaluation

The test sources are evaluated as originals, as mirrors, and as a fixed random mixture. The mixture selects exactly one view per source with seed 20261106 and half mirrored within each letter/fold. Both learners see the identical mixture. Detector failure never causes a source to be replaced by an easier view.

Report all-attempt accuracy, macro-precision, macro-recall and macro-F1, including no-prediction failures. Conditional accepted-only and paired-both-accepted results are additional views, not substitutes for the primary score. Letter-level scores, fold differences and paired corrected/lost counts are retained. Precision does not directly penalize a no-prediction failure; recall and F1 do.

The historical cache labels 57,424 of 60,251 accepted images Right (95.31%) and 2,827 Left (4.69%). These are MediaPipe estimates of the depicted hand, not verified handedness or signer dominance. A 50% reflected test mix does not guarantee exactly 50% true left hands; detected hand distributions are therefore reported separately. New original and mirrored detections both use the same recorded settings, rather than comparing a fresh mirror run with uncertain historical thresholds.

No independently collected left-handed signer cohort is available. This is a synthetic reflection experiment. J/Z movement, unseen-person reliability, new illumination and new viewpoints remain outside its evidence.

## Run from the repository root

```bash
conda activate ktai-hand-gesture
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.mirror_dataset --workers 4
python -m research.studies.mirror_augmentation --stage cv
python -m research.studies.mirror_augmentation --stage train
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.report_mirror
python -m research.studies.verify_mirror
python -m unittest discover -s tests
```

The extraction processes all 78,000 sources in two views. Matching chunk checkpoints under `runs/mirror_01/detections/` make interruptions resumable. The source dataset, historical landmarks, old predictions and existing model files remain intact. A different detector configuration needs fresh experiment paths. On macOS, MediaPipe needs access to the graphics runtime even for this IMAGE-mode workflow.

CV reuses matching fold models under the saved protocol. Full-data training creates two new live bundles and refuses to overwrite them on a second invocation. Raw landmarks/features and fold models remain under ignored `runs/`; compact protocols, test assignments, predictions, metrics and verification records are under `results/mirror_01/`. The reporter generates `paper/mirror_*rows.tex`, `paper/mirror_numbers.tex` and the figure from those results.

## Compare live

```bash
python -m research.live_camera --model models/asl_shin_angles_mirror_baseline.joblib
```

Select **Fresh original [7]** and **Mirror trained [8]** on the same camera stream, using the actual left or right hand. The old **World joints [2]** uses the historical fit and is not the matched fresh baseline. The screen preview remains mirrored for convenience; the detector input stays unmirrored. Snapshot exports retain the exact active model hash and features. Model switching stays locked during guided collection.

Selected live examples are diagnostics. To estimate real left-hand performance, collect all planned attempts with hidden predictions from both hands and reserve new participants/sessions before choosing further adjustments.
