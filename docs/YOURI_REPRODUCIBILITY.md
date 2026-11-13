# RQ1 — Reproducibility: ASL letter confusion (M/N and R/U)

## Scope and results

Three **26-class SVMs** with the same Shin 2021 combined 820-feature representation were compared using the same 60 stored, detector-usable webcam test records (15 each of M, N, R, U):

| Model | Development training data | Accuracy (60 test images) | M/N mutual confusion | R/U mutual confusion |
|---|---|---:|---:|---:|
| Baseline | Original Kaggle development set | 39/60 = 65.00% | 13/30 = 43.33% | 8/30 = 26.67% |
| Targeted | Original + 59 development recordings (M=14, N=15, R=15, U=15) | 47/60 = 78.33% | 7/30 = 23.33% | 6/30 = 20.00% |
| N-only ablation | Original + the **same 15 N development recordings** | 44/60 = 73.33% | 8/30 = 26.67% | 8/30 = 26.67% |

The initial targeted comparison was performed before its 60 test labels/results were viewed. The N-only comparison was designed **after** viewing the targeted results and used the same test samples: it is an *exploratory viewed-set retest*, not an independent validation. Both development and test recordings came from the same participant; exact pixel/ID overlap checks passed for the original evaluation, but this does not establish independence of participants, viewpoint, or near-duplicate poses. No general-data control experiment was run.

## 1. Code and environment

From the project root, use the pinned project commit and the project's environment:

```cmd
conda env create -f environment.yml
conda activate ktai-hand-gesture
python -m scripts.youri_reproduce
```

This **replays predictions** with the three saved models; it does **not** re-record the webcam test or refit any models. See `exports/youri_reproduced/summary.json` and `predictions.csv`.

Original RQ1 code: `scripts/youri_confusion.py` (initial Kaggle cross-validation error analysis), `scripts/youri_n_only.py` (N-only refit), and `scripts/youri_reproduce.py` (three-model replay). The original baseline's CSV is under `experiments/shin2021_kaggle_01/svm/shin2021_combined/cv_predictions.csv`.

## 2. External evidence bundle (not committed to public Git)

Keep these relative directories in the same positions when the research bundle is unpacked alongside the repository:

- `experiments/shin2021_kaggle_01/svm/` — original fitted baseline and experiment metadata.
- `experiments/youri_targeted_01/` — targeted model, `selection.json`, and training provenance.
- `experiments/youri_n_only_01/` — N-only fitted model and provenance.
- `data/youri_development/` — own development recordings; source of the selected 59 samples and subset of 15 N samples.
- `data/youri_test/` — the original 60 test captures, images, manifest, landmark records and recorded trial metadata.
- `exports/youri_evaluation/` — original baseline-versus-targeted result and `scores.csv`.
- `exports/youri_n_only_retest/` — exploratory baseline-versus-N-only retest.
- `exports/youri_baseline.json`, `exports/youri_test_protocol.json`, `exports/youri_test_seal.json` — provenance; some historic JSON files have machine-specific absolute paths and are for auditing, not portable directly.

Full model **retraining** additionally requires the original imported `data/asl_kaggle/` dataset, and original source experiment manifests with paths valid on the new computer. If these are not available, the three-model replay remains possible with the fitted models and stored test landmarks. The raw Kaggle image dataset may be obtained/imported separately according to `docs/KAGGLE_TRAINING.md`; exact replication of the original experiment requires the same dataset selection, split, source/feature code and software versions.

The webcam recordings may contain personal imagery. Do **not** push them to a public repository without informed permission; provide a restricted course attachment or approved private distribution instead. Also check large model sizes and redistribution rights before including files.

## 3. Check the original evidence

Review these recorded outputs (no training required):

```cmd
type exports\youri_evaluation\scores.csv
python -c "import json; d=json.load(open('experiments/youri_targeted_01/selection.json',encoding='utf-8')); print(d['added_counts']['label'])"
python -c "import json; d=json.load(open('exports/youri_evaluation/result.json',encoding='utf-8')); print(d['paired_categories'], d['overlap_checks'])"
```

Expected initial scores: baseline accuracy 0.65, macro-F1 0.6061384; targeted accuracy 0.7833333, macro-F1 0.7758333; 59 usable added images; originally **8 corrected, 0 introduced** errors. For the N-only retest inspect its own `result.json`, `scores.csv` and explicit viewed-set-retest flag.

## 4. Full model refitting (optional, more expensive)

**Targeted (59 images):** First register the local baseline from the original source experiment; ensure its `split.json` and imported Kaggle project's paths are valid on this machine. The original route in this study was:

```cmd
python -m scripts.research register --model experiments/shin2021_kaggle_01/svm/shin2021_combined/model.joblib --experiment experiments/shin2021_kaggle_01/svm --alphabet ASL --name "Youri Baseline" --require-baseline --out exports/repro_baseline.json
python -m scripts.research train --baseline exports/repro_baseline.json --additions data/youri_development --all-development --dataset-out data/repro_targeted --experiment-out experiments/repro_targeted --hypothesis "Additional M N R U training images reduce mutual confusion" --change "Original development data plus targeted development records" --expected "Lower M N R U mutual confusion"
```

**N-only (15 images):** From `experiments/youri_targeted_01/selection.json`, select the 15 IDs in `added_ids` whose `label` is `N` in `data/youri_development/manifest.json`, and pass them as arguments after `--ids` to the same `scripts.research train` command, with new dataset and experiment output directories. Alternatively use the committed `scripts/youri_n_only.py` if compatible with the supplied project version. The important control is that the N-only images must be **the same 15 N images from the 59-image targeted extension**, not newly selected samples.

Never overwrite the original training experiment directories. Refit models can be checked by invoking `python -m scripts.youri_reproduce` after placing them at the expected model paths, or adapt its `MODELS` paths to the new fitted files.

Note: retraining may not produce byte-identical `joblib` artifacts on different platforms or dependency versions. Report both the original frozen-model replay and any newly retrained results separately.

## 5. Files to include in the PR

- `scripts/youri_confusion.py`
- `scripts/youri_n_only.py`
- `scripts/youri_reproduce.py`
- `docs/YOURI_REPRODUCIBILITY.md`
- Optional: `experiments/youri/results/confusion_pairs.csv`, `confusion_matrix.png` and summary CSVs after reviewing any IDs/paths.

Do not commit full personal images or the large Kaggle data to a public PR. Deliver the evidence bundle through the course's authorized distribution channel.
