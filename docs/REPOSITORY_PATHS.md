# Working from any checkout

Open a terminal in the repository root, the directory containing `app.py`, `environment.yml`, `research/` and `experiments/`. All commands below start there. Activate the shared environment by name; Python is selected from that environment rather than from a user's installation folder.

```powershell
# Once, if the environment does not exist:
conda env create -f environment.yml

conda activate ktai-hand-gesture
python --version
python -m pip check
```

Use Python 3.12 and the versions in `requirements.txt`. Prepare the local dataset and trained model assets according to [REPRODUCE.md](../REPRODUCE.md). The dataset belongs under `data/asl-alphabet-download/`; the full A–Z experiment needs 3,000 training images per letter. The repository README includes an inventory check. These large local assets are separate from the source code.

## Validation commands

The portability tests create their own tiny image fixture and mock detection. They exercise copying a checkout, resuming an existing run, and opening relative gallery links without the real dataset or a webcam:

```powershell
python -m unittest tests.test_recovery_paths -v
python experiments/hand_detection_recovery/verify_results.py --run paper/evidence/rq2_test_110
```

Run the full software suite after preparing the local model and example assets described in the reproduction guide:

```powershell
python -m unittest discover -s tests -v
```

For an end-to-end RQ2 check with real images, use a new run name. The wrapper replaces a run with that name unless `--resume` is supplied:

```powershell
python experiments/hand_detection_recovery/run.py --name portable_check_110 --limit 110
python experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/portable_check_110
python experiments/hand_detection_recovery/plot_results.py --run experiments/hand_detection_recovery/runs/portable_check_110
python experiments/hand_detection_recovery/run.py --name portable_check_110 --limit 110 --resume
```

The analysis and gallery are created automatically by the run command. To regenerate an existing run's analysis:

```powershell
python experiments/hand_detection_recovery/analyze.py --run experiments/hand_detection_recovery/runs/portable_check_110
```

## Paths in shared results

Default RQ2 source and output paths are stored relative to the repository root, with `/` separators. Image URLs in `review_gallery.html` are relative to the gallery file. Copy the original dataset and run outputs into the same relative locations in a colleague's checkout. Its containing folder and drive may differ.

Explicitly selecting a source or output outside the repository still works locally, but those external files must be supplied separately. Keep source and run folders inside the repository layout for a self-contained handoff.

Resume verifies source inventory, settings and processing-code hashes. Copy source files with their modification times preserved when resuming a new run on another checkout; changes to files or code require a new run. Completed exports can be verified and analyzed independently of SQLite checkpoints.

The supplied historical text exports have been normalized to this layout. Measurements, image IDs, landmarks and recorded image/model/code hashes were preserved. The archived evidence manifest retains the original export hashes as `pre_path_normalization_sha256` and records the current export hashes under `sha256`. Figure and repeatability provenance similarly retain their earlier hashes. Original SQLite checkpoints retain their historical code/inventory contract and are not rewritten into new measurements.

## Validation recorded on 9 November 2026

- The full combined suite passed: **239 tests**.
- Four portability tests passed, including resuming a copied checkout, gallery links from a different working directory, URL escaping and N47 figure generation after relocation.
- The real-image `portable_check_110` run completed baseline, preprocessing, retry, analysis, verification, figures and resume. It contained 110 images, 46 original misses and 8 unique recoveries.
- An audit of 94 historical export files checked 876,076 rows: measurement fields, image hashes and landmark values were unchanged by path normalization.
- Six saved RQ2 evidence sets passed verification, including both 78,000-image result sets and the archived paper evidence.
- No personal Documents-directory references remain in the repository's documentation and text exports. Git history and binary checkpoint provenance are outside this text normalization.

The full dataset detector was not rerun, and no physical webcam test was performed. Prepare the dataset and model assets described above before running the same checks on another workstation. The commands in this document start from the repository root.
