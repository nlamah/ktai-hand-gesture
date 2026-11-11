# Working from any checkout

Open a terminal in the repository root, the directory containing `requirements.txt`, `research/` and `experiments/`. All commands below start there. Activate the shared environment by name; Python is selected from that environment rather than from a user's installation folder.

```bash
# Once, if the environment does not exist:
conda create -n ktai-hand-gesture python=3.12

conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
python --version
python -m pip check
```

Use Python 3.12 and the versions in `requirements.txt`. Prepare the local dataset as described in the [RQ2 guide](../experiments/hand_detection_recovery/README.md). The dataset belongs under `data/asl-alphabet-download/`; the full A–Z experiment needs 3,000 training images per letter. Check this inventory before a complete experiment; see the RQ2 experiment guide for the expected layout. These large local assets are separate from the source code.

## Validation commands

The portability tests create their own tiny image fixture and mock detection. They exercise copying a checkout, resuming an existing run, and opening relative gallery links without the real dataset or a webcam:

```bash
python -m unittest tests.test_recovery_paths -v
```

Run the full software suite:

```bash
python -m unittest discover -s tests -v
```

For an end-to-end RQ2 check with real images, use a new run name. The wrapper replaces a run with that name unless `--resume` is supplied:

```bash
python experiments/hand_detection_recovery/run.py --name portable_check_110 --limit 110
python experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/portable_check_110
python experiments/hand_detection_recovery/plot_results.py --run experiments/hand_detection_recovery/runs/portable_check_110
python experiments/hand_detection_recovery/run.py --name portable_check_110 --limit 110 --resume
```

The analysis and gallery are created automatically by the run command. To regenerate an existing run's analysis:

```bash
python experiments/hand_detection_recovery/analyze.py --run experiments/hand_detection_recovery/runs/portable_check_110
```

## Paths in shared results

Default RQ2 source and output paths are stored relative to the repository root, with `/` separators. Image URLs in `review_gallery.html` are relative to the gallery file. Copy the original dataset and run outputs into the same relative locations in a colleague's checkout. Its containing folder and drive may differ.

Explicitly selecting a source or output outside the repository still works locally, but those external files must be supplied separately. Keep source and run folders inside the repository layout for a self-contained handoff.

Resume verifies source inventory, settings and processing-code hashes. Copy source files with their modification times preserved when resuming a new run on another checkout; changes to files or code require a new run. Completed exports can be verified and analyzed independently of SQLite checkpoints.
