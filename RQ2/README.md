# RQ2: recovering missed hand detections

This guide describes Danny's contribution. Installation and the shared baseline are covered in the main [README](../README.md). Numbering follows the group paper: RQ1 targeted data collection, RQ2 detection recovery, RQ3 feature comparison.

## Question and main comparison

**To what extent does a 3 x 3 dilation filter recover visually plausible hand landmarks in ASL Alphabet training images for which MediaPipe initially detects no hand?**

| Condition | Processing before detection |
|---|---|
| Original baseline | Decode the original image and run the detector |
| `dilatatie` (dilation) | Dilate the RGB image with a 3 x 3 all-ones kernel, one iteration |
| `erosie` (erosion) | Erode the RGB image with the same kernel, one iteration |
| `canny_overlay` | 5 x 5 Gaussian blur, Canny 50/150, edges blended into the original with weights 1.0/0.25 |

Dilation is the central comparison; erosion and the Canny overlay are additional comparisons. Every variant starts from the same original image; filters are not chained. Only images with baseline status `no_hand` are processed and retried, with the same detector model and thresholds. The variant identifiers keep their original names because they are column names in the recorded evidence.

The detector is `research.data.Detector`: MediaPipe Hand Landmarker in `IMAGE` mode, at most two hands, detection and presence confidence 0.5, model `models/hand_landmarker.task` (SHA256 checked). Outcomes are `ok` (one hand, accepted geometry), `no_hand`, `multiple_hands` and `invalid_geometry`. A **recovery** is a change from `no_hand` to `ok`. That is an automatic acceptance rule, not proof of anatomically correct landmarks; visual review answers the plausibility part. RQ2 trains no letter classifier and reports no precision or F1.

## Run

From the repository root, after installation and with the raw ASL Alphabet training images in `data/asl-alphabet-download/asl_alphabet_train/A-Z/`:

```bash
python -m RQ2 list
python -m RQ2 recovery --name rq2_check_110 --limit 110 --dry-run
python -m RQ2 recovery --name rq2_check_110 --limit 110
python -m RQ2 recovery --name rq2_full_02
```

`recovery` runs baseline detection, preprocessing, retry detection, analysis, verification and figures. `--limit` takes the first N images in A-Z and filename order; it is a technical check, not a balanced sample. **Reproduction replaces an existing run with the same name** unless `--resume` is given; resume with the same limit. Checkpoints refuse to resume after source, settings or code changes. To verify or regenerate existing results:

```bash
python -m RQ2 recovery --name rq2_check_110 --action verify
python -m RQ2 recovery --name rq2_check_110 --action report
python -m RQ2 compare --reference rq2_check_110 --name rq2_check_110_repeat
```

`compare` needs two completed runs and writes `repeatability.json`; check `all_compared_outputs_equal`. [workflows.py](workflows.py) defines the commands and output directories. The cached classifier bundle from the main README excludes raw images and cannot reproduce RQ2.

## Findings

Full run on all 78,000 training images (`runs/test_all`), repeated independently with identical outputs (`runs/test_all_repeat/repeatability.json`):

| Measure | Result |
|---|---:|
| Baseline accepted (`ok`) | 60,251 (77.24%) |
| Baseline `no_hand` | 17,598 |
| Recovered by dilation | 3,560 (20.23%) |
| Recovered by erosion | 2,018 (11.47%) |
| Recovered by Canny overlay | 1,391 (7.90%) |
| Recovered by any variant, unique images | 4,799 (27.27%) |
| Still `no_hand` in all three variants | 12,779 |
| Coverage after accepting any variant | 83.40% |

- Dilation recovers the most baseline misses, but most misses (about 80% for dilation) remain undetected.
- Variants overlap: 6,969 successful image-variant pairs correspond to 4,799 distinct images, so percentages must not be added.
- Retries rarely produce `multiple_hands` (dilation 31, erosion 17, Canny 2) and never `invalid_geometry`.
- Letters differ strongly; N (53.5% coverage after recovery) and M (70.3%) remain the weakest. See `by_letter.csv`.

The 60,251 accepted baseline detections match the input count of the RQ3 classification study. Recovery is technical: the visual review of landmark plausibility has not yet been recorded, so the full research question is not yet answered. The design retries only misses and does not estimate damage to originally accepted images. N47 is a development example, not independent validation. Results do not cover new users, webcam video, other datasets or the movements of J/Z.

Evidence is in `experiments/hand_detection_recovery/runs/test_all/` (local; run folders are excluded from git): `summary.csv`, `per_image.csv`, `by_letter.csv`, `retry_outcomes.csv`, `settings.json` and `figures/`.

## Reproduction details

### Inputs

A complete A-Z training set has 3,000 images per letter (78,000). The extra nesting `asl_alphabet_train/asl_alphabet_train/A/...` is supported; nonletter classes and `asl_alphabet_test` are ignored. Missing images do not stop a run, so check `images` in `settings.json` before reporting full-dataset results. For a different source folder, run the stages directly with identical `--source`, `--run` and `--limit`:

```bash
python -m research.recovery_experiment baseline --source data/my-asl-copy --run experiments/hand_detection_recovery/runs/rq2_custom_01
python -m research.recovery_experiment preprocess --source data/my-asl-copy --run experiments/hand_detection_recovery/runs/rq2_custom_01
python -m research.recovery_experiment retry --source data/my-asl-copy --run experiments/hand_detection_recovery/runs/rq2_custom_01
python -m RQ2 recovery --name rq2_custom_01 --action report
```

### Outputs

Each run is written to `experiments/hand_detection_recovery/runs/<name>/`.

| File | Contents |
|---|---|
| `settings.json` | Input count, limit, model/code/inventory hashes, library versions, detector and filter settings |
| `baseline.csv`, `no_hand.csv` | Original outcomes and the misses selected for retry |
| `processed.csv`, `processed/` | Processed PNGs and their hashes |
| `per_image.csv`, `summary.csv` | Paired outcomes per image and recovery per variant |
| `recovered.csv`, `unrecovered.csv`, `still_no_hand.csv` | Recovered misses, misses without accepted retry, misses with zero detections in every variant |
| `retry_landmarks.jsonl` | Retry detections with image/world landmarks |
| `analysis.md`, `by_letter.csv`, `retry_outcomes.csv` | Readable analysis and breakdowns |
| `review_gallery.html`, `visual_review_template.csv` | Visual comparison and empty review fields |
| `figures/` | Plots, underlying tables and `figure_sources.json` |
| `progress.sqlite` | Resume checkpoints |

### Visual review

Copy `visual_review_template.csv` to a separate file such as `visual_review_completed.csv` before entering judgments; reporting regenerates the template and a clean restart deletes the run folder. Define criteria first, for example whether points follow the visible wrist and fingers. Use `plausible`, `implausible`, `uncertain` or `not_applicable`, describe visible changes before explanations, and record the reviewer and sample size.

### Verification

`verify` checks counts and IDs, the three processed variants per miss, agreement between retry records and paired outcomes, landmark counts, summary percentages and the recovered/unrecovered/still-no-hand lists. Run it with normal Python, not `python -O`. `verified: true` means the exports are internally consistent; it does not rerun MediaPipe or certify visual plausibility.

## Code locations

| Responsibility | Implementation |
|---|---|
| Public commands | [__main__.py](__main__.py), [workflows.py](workflows.py) |
| Ordered stages, clean start and run lock | [run.py](../experiments/hand_detection_recovery/run.py) |
| Input selection, checkpoints and paired results | [recovery_experiment.py](../research/recovery_experiment.py) |
| Image filters | [morphology_pilot.py](../research/morphology_pilot.py) |
| Detector, model check and image decoding | [data.py](../research/data.py) |
| Export verification | [verify_results.py](../experiments/hand_detection_recovery/verify_results.py) |
| Tables, gallery and review template | [analyze.py](../experiments/hand_detection_recovery/analyze.py) |
| Figures | [plot_results.py](../experiments/hand_detection_recovery/plot_results.py) |
| Repeated-run comparison | [compare_runs.py](../experiments/hand_detection_recovery/compare_runs.py) |
| Tests | [test_rq2_interface.py](../tests/test_rq2_interface.py), [test_recovery_paths.py](../tests/test_recovery_paths.py) |

## Additional paper analysis and unchanged-PNG control

These modules write to a **separate, new output directory**. They never replace the source run or its checkpoints. Existing analysis output is rejected; the PNG control requires an explicit `--resume` to reuse its own checkpoint.

Recompute the rejection categories, recoveries without successful dilation, and per-letter results:

```bash
python -m RQ2.paper_analysis --run experiments/hand_detection_recovery/runs/test_all --output experiments/hand_detection_recovery/runs/paper_analysis_01
```

Outputs are `summary.json`, `by_letter.csv`, `rejection_combinations.csv` and `other_rejected_images.csv`. The analysis verifies the source exports first and records their hashes. Per-letter miss percentage uses all originals of that letter; recovery percentage uses only that letter's original misses. For a letter without original misses, recovery percentage is left empty rather than reported as measured zero. Status counts distinguish unique images from retries, and an image may appear in more than one rejection-status group.

For the saved `test_all` run, 1,239 unique images were recovered without a successful dilation retry. The 20 unrecovered images with at least one status other than `no_hand` all have `multiple_hands` (21 retries); none has `invalid_geometry`. The remaining 39 attempts within those 20 images have `no_hand`.

The optional null control decodes each original miss, saves the unmodified pixels as PNG, reads it back and checks exact pixel equality. It then runs the detector on both the decoded original and the reloaded PNG, recording both outcomes and landmarks. A nonzero result is retained, never treated as an expected-zero assertion.

For a small technical check:

```bash
python -m RQ2.png_control --run experiments/hand_detection_recovery/runs/test_all --output experiments/hand_detection_recovery/runs/png_control_check_01 --limit 20
```

For all original misses, use a different output name and omit the limit:

```bash
python -m RQ2.png_control --run experiments/hand_detection_recovery/runs/test_all --output experiments/hand_detection_recovery/runs/png_control_full_01
```

Resume with the same source and limit (if used):

```bash
python -m RQ2.png_control --run experiments/hand_detection_recovery/runs/test_all --output experiments/hand_detection_recovery/runs/png_control_full_01 --resume
```

The control writes `settings.json`, `per_image.csv`, `detections.jsonl`, `summary.json`, unchanged PNGs under `images/`, and `progress.sqlite`. Each completed pair is checkpointed. A source/model/version change is rejected; resumed controls also reject changed code or saved PNGs. A forced process termination can leave `control.lock`: confirm that no control process is active before removing a stale lock.

`png_recoveries_vs_historical` counts PNG outcomes that are now `ok` among historical misses. `original_recoveries_vs_historical` identifies changes already present when replaying the original. `paired_png_recoveries` counts current-original `no_hand` to current-PNG `ok`. Pixel equality does not guarantee deterministic detector outcomes. The comparison uses sequential calls on the same detector; it does not by itself distinguish nondeterminism from an ordering effect. Model and MediaPipe/OpenCV versions must match the source settings, while current code hashes and differences from historical code are recorded explicitly.

The first 20-image technical check returned zero original or PNG recoveries, no paired status changes and identical pixels for every PNG round trip. This is not a result for all 17,598 original misses; run the full control before making that claim in the paper. The limit selects a deterministic letter/filename ordering, not a representative sample.

Run the additional tests with:

```bash
python -m unittest tests.test_rq2_paper_checks tests.test_rq2_interface tests.test_recovery_paths tests.test_import -v
```
