# RQ2: recovering missed hand detections

This guide describes the RQ2 contribution: retrying failed MediaPipe hand detections after image processing. Numbering follows the group project: RQ1 targeted data collection, RQ2 detection recovery, RQ3 feature comparison. General project context is in the [main README](../README.md); the [RQ3 guide](../RQ3/README.md) describes the separate classification study.

**The RQ2 implementation is available** through `python -m RQ2`. This public interface calls the existing scripts in `research/recovery_experiment.py` and `experiments/hand_detection_recovery/`; it does not duplicate the detection algorithms.

## Public commands

From the repository root, after activating the Conda environment:

```bash
python -m RQ2 list
python -m RQ2 --status
python -m RQ2 recovery --name rq2_check_110 --limit 110 --dry-run
python -m RQ2 recovery --name rq2_check_110 --limit 110
python -m RQ2 recovery --name rq2_check_110 --action verify
python -m RQ2 recovery --name rq2_check_110 --action report
python -m RQ2 recovery --name rq2_check_110 --limit 110 --resume
```

The default `reproduce` action runs the three detection stages, analysis, verification and figures. `verify` checks existing exports; `report` regenerates analysis and figures. `--dry-run` only prints commands and does not create, delete or modify run files. All subprocesses use the active Python environment and the repository root as their working directory.

**Reproduction replaces an existing run with the same name unless `--resume` is supplied.** Reporting regenerates the empty review template; keep completed judgments in a separate file. `--limit` and `--resume` apply only to reproduction. Use the same limit when resuming.

For all available images, omit `--limit`. To compare two independently completed runs:

```bash
python -m RQ2 recovery --name rq2_full_01 --dry-run
python -m RQ2 compare --reference rq2_check_110 --name rq2_check_110_repeat
```

The repeat must already exist. Inspect its `repeatability.json`, especially `all_compared_outputs_equal`; command completion alone is not a claim that outputs match. The detailed script commands below remain supported.

## Question and main comparison

**To what extent does a 3 x 3 dilation filter recover visually plausible hand landmarks in ASL Alphabet training images for which MediaPipe initially detects no hand?**

Dilation is the central comparison. Erosion and a Canny overlay are additional processing comparisons. Each variant starts from the same original image; the filters are not chained.

| Condition | Processing before detection |
|---|---|
| Original baseline | Decode the original image and run the detector |
| `dilatatie` | Dilate the RGB image with a 3 x 3 all-ones kernel, one iteration |
| `erosie` | Erode the RGB image with the same kernel, one iteration |
| `canny_overlay` | Convert to grayscale, apply a 5 x 5 Gaussian blur, run Canny with thresholds 50/150, blend the edges into the original RGB image with weights 1.0/0.25 |

The original baseline covers every selected input. Only images with baseline status `no_hand` are processed and retried. Each of those images gets all three variants, using the same detector model and thresholds.

RQ2 measures **detection recovery**, not letter-classification precision or F1. It reuses `research.data.Detector`; it does not train the DS/DSJ classifiers from the baseline/RQ3 study. Classifier models and accepted-landmark caches do not replace the raw images needed here.

## Detector settings and outcome definitions

The detector uses MediaPipe Hand Landmarker in `IMAGE` mode, with a maximum of two hands, minimum detection confidence 0.5 and minimum presence confidence 0.5. The model is `models/hand_landmarker.task`. Its expected SHA256 is checked; if the file is absent, the loader downloads and checks it, requiring internet access on that first run.

| Status | Meaning in this implementation |
|---|---|
| `no_hand` | No hand detected |
| `multiple_hands` | More than one hand detected; not accepted as a single-hand result |
| `invalid_geometry` | One hand detected, but the geometric feature validation failed |
| `ok` | One hand detected and the geometric feature extraction accepted its landmarks |

A recovery is a change from original `no_hand` to retry `ok`. This is an automatic acceptance rule, **not proof of anatomically correct landmarks**. Visual review is a separate part of answering the research question.

For each variant, recovery percentage is `100 * recovered / original_misses`. `any_variant_unique` counts an original image once when at least one variant succeeds. Do not add variant counts to obtain unique recovery: variants can succeed on the same image. With zero original misses, the implementation writes a recovery percentage of zero.

## Installation

Run all commands from the repository root in Anaconda Prompt or a terminal with Conda available. Commands use relative paths and do not depend on a particular user's installation folder.

If the environment already exists:

```bash
conda activate ktai-hand-gesture
python --version
python -m pip check
```

For a new environment only:

```bash
conda create -n ktai-hand-gesture python=3.12
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
python -m pip check
```

If Conda asks to remove an existing environment, answer **no** and activate it instead. The current checkout has no root `environment.yml`; use `requirements.txt`. After a dependency change, install that file into the active environment again. Dependencies include version ranges, so keep each run's recorded versions when comparing results across machines.

## Inputs and dataset inventory

Extract the ASL Alphabet training images into this layout:

```text
data/asl-alphabet-download/
  asl_alphabet_train/
    A/A1.jpg, A2.jpg, ...
    B/B1.jpg, B2.jpg, ...
    ...
    Z/Z1.jpg, Z2.jpg, ...
```

The extra nesting `asl_alphabet_train/asl_alphabet_train/A/...` is also supported. The loader reads A-Z directories and positive numbered filenames such as `N47.jpg`, with `.jpg`, `.jpeg` or `.png` extensions. It ignores nonletter classes. RQ2 does not read the supplied `asl_alphabet_test` directory.

A complete A-Z training set has 3,000 images per letter, 78,000 in total. Missing letter directories or missing numbered images do not automatically stop a run: **a successful run may use a partial dataset**. Check the inventory before reporting full-dataset results:

```bash
python -c "from pathlib import Path; from collections import Counter; from research.kaggle import LETTERS, _split_folder, _training_entries; rows=list(_training_entries(_split_folder(Path('data/asl-alphabet-download'), 'asl_alphabet_train'), 500)); counts=Counter(r['label'] for r in rows); print('Total:', len(rows)); print({letter: counts[letter] for letter in LETTERS}); print('Expected 3000 per letter:', all(counts[letter] == 3000 for letter in LETTERS))"
```

This checks counts, not independent dataset authenticity. The latest local inventory check found all 78,000 entries: 3,000 per letter, with numbering 1-3000 complete and no duplicate numbers. This supersedes the earlier 71,945-entry inventory; it is a filename/count check, not a visual or full image-decoding audit.

The shared `baseline.bundle` archive contains cached classification inputs and explicitly excludes raw images. It is therefore insufficient to reproduce RQ2 detection. Supply the original image dataset separately. Share large datasets and generated runs outside source control, retaining the relative directory layout.

## Quick software and end-to-end checks

First run the tests for the RQ2 path handling and shared import helpers. They create small fixtures and mock detection; they do not require the full Kaggle dataset or a webcam:

```bash
python -m unittest tests.test_rq2_interface tests.test_recovery_paths tests.test_import -v
```

Then run the real detector on 110 inputs. Choose a fresh run name:

```bash
python experiments/hand_detection_recovery/run.py --name rq2_check_110 --limit 110
python experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/rq2_check_110
python experiments/hand_detection_recovery/plot_results.py --run experiments/hand_detection_recovery/runs/rq2_check_110
```

The runner performs baseline detection, preprocessing, retry detection and analysis in order. `verify_results.py` should print `"verified": true`. Plotting writes figures and their underlying tables. Open `analysis.md`, `review_gallery.html` and `figures/` inside the run directory.

`--limit` selects the first inputs in A-Z order, then numerical filename order. It is not random or letter-balanced. The 110-image check therefore cannot establish a recovery rate for the whole alphabet. `N47 is absent from this run; its comparison was skipped` is expected when N47 is outside the selection; other figures are still generated.

## Full run, interruption and resume

After checking the complete input inventory:

```bash
python experiments/hand_detection_recovery/run.py --name rq2_full_01
python experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/rq2_full_01
python experiments/hand_detection_recovery/plot_results.py --run experiments/hand_detection_recovery/runs/rq2_full_01
```

Without `--limit`, all available A-Z training entries are processed. This does not enforce 78,000 inputs. The experiment stores three processed PNGs per original miss, plus landmarks and a SQLite checkpoint, so it needs additional disk space beyond the original dataset.

**Without `--resume`, the runner deletes any existing output directory with the same run name before starting.** Use a new name to retain an earlier measurement. Run names accept letters, digits, underscores and hyphens and must start with a letter or digit.

Resume an interrupted quick check with the same name and limit:

```bash
python experiments/hand_detection_recovery/run.py --name rq2_check_110 --limit 110 --resume
```

Resume a full run without adding a limit:

```bash
python experiments/hand_detection_recovery/run.py --name rq2_full_01 --resume
```

The pipeline stops on a failed stage. It checks the recorded source inventory, configuration and processing-code hashes before resuming; a change requires a new run. Inventory checks include source file sizes and modification times. Preserve those timestamps when moving an unfinished run to another checkout. Two wrapper processes cannot use the same run name simultaneously.

For an alternative source directory, use the individual stages. Use exactly the same source, run directory and optional limit in all three commands:

```bash
python -m research.recovery_experiment baseline --source data/my-asl-copy --run experiments/hand_detection_recovery/runs/rq2_custom_01 --limit 110
python -m research.recovery_experiment preprocess --source data/my-asl-copy --run experiments/hand_detection_recovery/runs/rq2_custom_01 --limit 110
python -m research.recovery_experiment retry --source data/my-asl-copy --run experiments/hand_detection_recovery/runs/rq2_custom_01 --limit 110
python experiments/hand_detection_recovery/analyze.py --run experiments/hand_detection_recovery/runs/rq2_custom_01
```

Unlike the wrapper's clean start, these individual stages retain progress in the chosen directory. The custom source still needs the `asl_alphabet_train` layout described above.

## Results and visual assessment

All output belongs to `experiments/hand_detection_recovery/runs/<name>/`.

| File or directory | What to inspect |
|---|---|
| `settings.json` | Selected input count, limit, model/code/inventory hashes, library versions and detector/filter settings |
| `baseline.csv` | Original-image IDs, paths, hashes and detector outcomes |
| `no_hand.csv` | Baseline misses selected for processing |
| `processed.csv`, `processed/` | Mapping to each processed image and the generated PNGs |
| `per_image.csv` | Paired outcomes for all three variants and `recovered_variants` |
| `summary.csv` | Per-variant and unique recovery counts and percentages |
| `recovered.csv` | Original misses recovered by at least one variant |
| `unrecovered.csv` | Original misses with no accepted retry; this can include outcomes other than `no_hand` |
| `still_no_hand.csv` | Original misses for which all three retries remain `no_hand` |
| `retry_landmarks.jsonl` | Retry detection records, including available image/world landmarks |
| `progress.sqlite` | Checkpoints used for resuming; retain when an unfinished run must continue |
| `analysis.md`, `by_letter.csv` | Human-readable analysis and breakdown by letter |
| `review_gallery.html` | Original and processed images with retry landmark overlays |
| `visual_review_template.csv` | Empty fields for manual judgments per image and variant |
| `figures/` | Plots, underlying tables and `figure_sources.json` provenance |

Before entering judgments, copy `visual_review_template.csv` to a separate file, such as `visual_review_completed.csv`. Rerunning analysis regenerates the template. A clean start with the same run name also deletes manually added files in that run, so preserve completed assessments separately.

Define visual acceptance criteria before scoring: for example, whether landmarks follow the visible wrist/fingers and whether a purported recovery places points on the hand rather than the background. These are suggested criteria, not an automated or already validated scoring protocol. Record `judgement`, `visible_change`, `possible_explanation`, `alternative_explanation` and `reviewer`. Distinguish observed image changes from speculative explanations of detector behavior. Record exclusions and the reviewed sample size; inspecting selected successful examples alone does not validate every recovery.

Image paths are repository-relative for the normal dataset/output layout; gallery links are relative to the gallery file. Preserve the original dataset and processed images at their corresponding locations when sharing visual evidence. CSV/JSON exports alone support numerical verification but not a complete visual review.

## Verification, reports and repeatability

Verify existing exports without rerunning the detector:

```bash
python experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/rq2_check_110
```

The verifier checks input counts and IDs, original misses, three processed variants per miss, correspondence between retry records and paired outcomes, accepted landmark counts, summary percentages and unrecovered IDs. When an evidence manifest is present, it also checks the listed export hashes. Run it with ordinary Python, not `python -O`, because its checks use assertions.

`verified: true` establishes internal consistency of these exports. It does not rerun MediaPipe, independently authenticate the original dataset, certify visual plausibility or show improved letter classification. The `recovered.csv` and `still_no_hand.csv` convenience exports are not separately checked by the current verifier.

Regenerate analysis and figures from a completed run:

```bash
python experiments/hand_detection_recovery/analyze.py --run experiments/hand_detection_recovery/runs/rq2_check_110
python experiments/hand_detection_recovery/plot_results.py --run experiments/hand_detection_recovery/runs/rq2_check_110
```

To test repeatability, make a second independent run with the same input selection:

```bash
python experiments/hand_detection_recovery/run.py --name rq2_check_110_repeat --limit 110
python experiments/hand_detection_recovery/verify_results.py --run experiments/hand_detection_recovery/runs/rq2_check_110_repeat
python experiments/hand_detection_recovery/compare_runs.py --reference experiments/hand_detection_recovery/runs/rq2_check_110 --repeat experiments/hand_detection_recovery/runs/rq2_check_110_repeat
```

Inspect `repeatability.json` and `repeatability_differences.csv` in the repeat run. Check `all_compared_outputs_equal`; a successful comparison command alone does not mean the runs matched. This comparison covers the exported content and retry coordinates, not baseline coordinates that were never exported. A resume check tests checkpoint reuse; it is not an independent repeat measurement.

## Observed technical check and interpretation limits

The 110-image check verified on 10 November 2026 produced:

| Measure | Result |
|---|---:|
| Original images | 110 |
| Original `ok` detections | 64 |
| Original `no_hand` detections | 46 |
| Recovered by erosion | 2 |
| Recovered by dilation | 6 |
| Recovered by Canny overlay | 0 |
| Unique recovered images | 8 |
| Unique recovery among original misses | 17.391% |

The recovery filenames were A59, A74, A75, A85, A89, A90, A92 and A98. This is a technical check on the first available images, not a full-dataset research result. It demonstrates that baseline, preprocessing, retry, verification and figure generation can execute together. The 11 targeted tests also passed on repeat; an earlier attempt encountered a Windows temporary-file access error.

The design retries only original misses, so it does not estimate damage to originally accepted detections if preprocessing were applied universally. It does not establish recognition quality for new participants, webcam video, other datasets or the movements of J/Z. N47 is a development example and should not be presented as independent validation. Any final claim about *visually plausible* recovery needs the recorded visual assessments as well as the automatic counts.

## Code locations

| Responsibility | Implementation |
|---|---|
| Ordered stages, clean start and run locking | [run.py](../experiments/hand_detection_recovery/run.py) |
| Input selection, settings, checkpoints and paired results | [recovery_experiment.py](../research/recovery_experiment.py) |
| Image filters | [morphology_pilot.py](../research/morphology_pilot.py) |
| Shared detector, model verification and image decoding | [data.py](../research/data.py) |
| A-Z source enumeration | [kaggle.py](../research/kaggle.py) |
| Relative paths and gallery URLs | [path_utils.py](../experiments/hand_detection_recovery/path_utils.py) |
| Independent export consistency checks | [verify_results.py](../experiments/hand_detection_recovery/verify_results.py) |
| Tables, gallery and assessment template | [analyze.py](../experiments/hand_detection_recovery/analyze.py) |
| Figures and figure provenance | [plot_results.py](../experiments/hand_detection_recovery/plot_results.py) |
| Repeated-run comparison | [compare_runs.py](../experiments/hand_detection_recovery/compare_runs.py) |
| Relocation/resume tests and shared import tests | [test_recovery_paths.py](../tests/test_recovery_paths.py), [test_import.py](../tests/test_import.py) |

## Alignment with RQ3

The comparison below covers the tracked RQ3 package files and their responsibilities. RQ2 needs equivalent capabilities where relevant, not copies of experiments that answer a different question.

| RQ3 component | RQ2 equivalent/current state | Assessment |
|---|---|---|
| `README.md` | This guide describes the question, inputs, methods, commands, results, limitations and provenance | Present |
| `__init__.py` | `RQ2/__init__.py` exists | Present |
| `__main__.py` | RQ2 dispatches through `RQ2/experiment.py` to the existing scripts | Implemented; includes list, status and validated run names |
| `contribution.json` | RQ2 declares its implementation, documentation, detector and actual output directory | Updated |
| `workflows.py` | RQ2 registers recovery reproduce/verify/report and completed-run comparison | Implemented with list, action and dry-run support |
| `gh_rotation.py` | RQ2 uses `verify_results.py` for its own paired detection exports | Verification role covered; G/H classifier replay is not applicable to RQ2 |
| `live_examples.py` | RQ2 has a recorded-image gallery and optional N47 figures | Visual inspection supported; webcam replay is outside this experiment |
| Shared baseline/model use | RQ2 reuses the detector on raw images; RQ3 uses accepted landmarks and classifiers | Appropriate difference; do not add classification training merely for structural symmetry |
| Shared reproduction bundle | RQ3's cache bundle excludes original images | RQ2 requires separate raw-image input distribution/instructions |
| Generated output exclusions | RQ2 `runs/`, `.locks/`, local `paper/` and results are excluded by appended ignore rules | Existing ignore rules retained |
| Public-interface tests | `tests/test_rq2_interface.py` checks dispatch, error propagation, metadata and dry-run preservation; the RQ1 placeholder assertion remains | RQ2 tests are independently runnable without RQ3 imports |

The public interface preserves the original output layout and clean-start/resume behavior. Numerical implementations remain in their existing modules. RQ3-specific G/H and webcam replay code is not needed for RQ2.

The complete merged repository's test suite was not green in the 10 November audit: it also referenced missing modules such as `research.geometry`, `research.shin`, `research.workbook` and `gesturelab.validation.provenance`. Those broader integration issues are distinct from the successfully exercised RQ2 pipeline. The targeted RQ2 tests are not a substitute for the full suite:

```bash
python -m unittest discover -s tests -v
```

Paper artifacts are not required to run or verify RQ2; numerical exports, settings, source code and the separately supplied images are the relevant reproduction materials.
