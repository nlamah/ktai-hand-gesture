# Tuning MediaPipe hand-detection thresholds

MediaPipe's `min_hand_detection_confidence` and `min_hand_presence_confidence`
are separate from the finger-angle thresholds in `config/finger_thresholds.json`.
The default for both detector thresholds is `0.5`. This workflow compares them
on the same saved development images; it does not edit the dataset manifest or
reuse previously stored detection results.

## 1. Run a coarse development sweep

```bash
python -m scripts.tune_detector_thresholds \
  --project data/asl_kaggle \
  --out experiments/detector_thresholds_coarse_01
```

By default this checks all 25 combinations of `0.1`, `0.2`, `0.3`, `0.4` and
`0.5`, plus the `(0.5, 0.5)` baseline if it was not already in the grid. It
selects up to 40 included development images per label using a stable sample
order. Test-role images and excluded images are never analyzed. To use all
development images, pass `--per-label 0`.

The output directory must be new. It contains:

- `summary.csv`: usable-detection coverage, recovered/lost counts relative to
  the baseline and detection failure categories for each candidate.
- `sample_results.csv`: per-image result and the image path for manual review.
- `run.json`: sample/source manifest hash, model hash and all run settings.

## 2. Compare finalists on all development images

Inspect `sample_results.csv` for images recovered by lower thresholds, especially
cases with multiple hands or invalid geometry. Check that the landmarks belong
to the intended hand and that the label is correct. Select a few candidates
that improve coverage without an unacceptable increase in questionable
detections, then run only those across the complete development set. For example:

```bash
python -m scripts.tune_detector_thresholds \
  --project data/asl_kaggle \
  --out experiments/detector_thresholds_full_01 \
  --detection-confidences 0.2,0.3,0.5 \
  --presence-confidences 0.2,0.3,0.5 \
  --per-label 0
```

The baseline is always measured first, even if it is not in the supplied lists.
This tool reports detection coverage only; higher coverage is not proof of
correct landmarks or improved gesture recognition. Retrain and compare
classifiers using identical development splits before deciding whether the
recovered images help.

## 3. Reanalyze with selected thresholds

After selecting thresholds using development results only, reanalyze the
dataset with those settings:

```bash
python -m scripts.analyze_dataset \
  --project data/asl_kaggle \
  --min-hand-detection-confidence 0.2 \
  --min-hand-presence-confidence 0.3 \
  --reprocess
```

`--reprocess` deliberately replaces stored detection results for every image
in the project, including test-role images. Do this only after locking the
thresholds; preserve a copy of the original project first if you need to keep
the original detections. Without `--reprocess`, the command analyzes pending
images only, preserving existing behavior.

## 4. Freeze and validate

Record the selected detector thresholds and model checksum with the experiment.
Do not use the reserved final test set to choose thresholds. After settings and
the model are fixed, evaluate the complete recognition pipeline on the
independent test set and count missing detections as failures.
