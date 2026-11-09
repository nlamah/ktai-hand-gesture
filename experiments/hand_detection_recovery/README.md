# Hand Detection Recovery — RQ2

This folder contains the experiment for recovering hand detection after image processing.
`run.py` sequentially calls the existing commands from
`research.recovery_experiment`. The image processing, detector, and counting
remain in the existing functions.

## Running the Experiment

Open an Anaconda PowerShell terminal in the repository root, containing `app.py` and `environment.yml`. All commands below start there. For a fresh environment, first run `conda env create -f environment.yml`.

```powershell
conda activate ktai-hand-gesture
$py = (Get-Command python -CommandType Application).Source
& $py --version
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $py experiments/hand_detection_recovery/run.py --name full_dataset_01
```

Place the source dataset under `data/asl-alphabet-download/`. Follow the complete dataset inventory and setup instructions in [the repository README](../../README.md#rq2-powershell-reproduction).

This performs three steps on the local A–Z training images:

1. **Baseline** measures detection on the original images.
2. **Preprocess** performs erosion, dilation, and a Canny overlay on each `no_hand` image.
3. **Retry** re-evaluates the detection and writes the comparison and totals.

For a quick technical check:

```powershell
& $py experiments/hand_detection_recovery/run.py --name check_01 --limit 110
```

`--limit` selects the first images in the dataset order. This is a technical
check, not a representative research sample. The included check
is located under `runs/chain_check_110`.

## Results per measurement run

Each name is assigned a folder under `experiments/hand_detection_recovery/runs/`.


| File or folder | Description |
| --- | --- |
| `settings.json` | Settings, software versions, and checksums |
| `baseline.csv` | Detection results for each original photo |
| `no_hand.csv` | Original photos without hand detection |
| `processed/` | PNG images per variant and letter |
| `processed.csv` | Mappings between original and processed photos |
| `per_image.csv` | Comparison per photo and successful variants |
| `summary.csv` | Counts and recovery rates per variant |
| `unrecovered.csv` | Photos for which no variant resulted in recovery |
| `retry_landmarks.jsonl` | Detection results and hand point coordinates |
| `progress.sqlite` | Progress data that allows the existing code to resume |

`recovered_variants` in `per_image.csv` contains all variants that resulted in `ok`.
`any_variant_unique` in `summary.csv` counts each recovered image once.
Recovery here means changing from `no_hand` to `ok`. You can then assess
visual plausibility using your predefined criteria.

## Clean Start and Resume

By default, the startup script deletes the entire existing results folder for
the specified `--name` before the baseline begins. This includes old CSV files,
processed images, and saved progress. All three steps are re-run.
Use a new name if you want to keep a previous run.

To resume after an interruption, explicitly add `--resume`, using the same
name and any `--limit`:

```powershell
& $py experiments/hand_detection_recovery/run.py --name check_01 --limit 110 --resume
```

If an error occurs, the pipeline stops. Resuming requires the same code and source data.
Two start commands with the same name cannot run at the same time.
The individual CLI steps continue to use their own progress tracking, so that the next
step can read the results of the previous step.
All original development images, including N47,
are included by default in the full A–Z measurement. If there are any exclusions,
define them separately in advance for an independent evaluation.

## Checking exported evidence

Recompute counts and check correspondence between image IDs, variant statuses,
landmark records and summary percentages without rerunning the detector:

```powershell
& $py experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/test_110
```

The evidence used in the RQ2 appendix is preserved under
`paper/evidence/rq2_test_110`. Use that directory with the same verification command
to check the archived results. `evidence_manifest.json` lists SHA256 checksums of
the archived exports. A clean run removes its own output folder, so retain a
separate evidence copy when reporting a run in the paper.

## Code locations

- `run.py`, `main()` — calls the three existing CLI steps.
- `research/recovery_experiment.py`, `baseline()` — baseline measurement.
- `research/recovery_experiment.py`, `preprocess()` — saves processed photos.
- `research/recovery_experiment.py`, `retry()` and `export_results()` — retest and comparison.
- `research/morphology_pilot.py`, `variants()` — filter operations.
- `research/data.py`, `Detector` — model settings and status determination.

## Sharing and testing from the repository root

New runs store original and processed image paths relative to the repository root, using `/` separators. The gallery uses links relative to its own HTML file. Keep the same `data/` and `experiments/` layout when sharing a checkout; the containing drive, user name and checkout folder can differ.

```powershell
python -m unittest discover -s tests -v
python experiments/hand_detection_recovery/verify_results.py --run paper/evidence/rq2_test_110
```

The archived evidence was normalized to repository-relative paths without changing image IDs, pixels, detector settings, landmarks or measurements. Its manifest records both the original export hashes and the hashes after path normalization. Historical SQLite checkpoints retain their recorded code/inventory contract; create a new run to use the current code. See [the portability notes](../../docs/REPOSITORY_PATHS.md) for the sharing layout and validation commands.
