# ASL landmark feature review

This local Streamlit tool reads per-letter expectations from an Excel workbook,
calculates MediaPipe hand landmarks for the COCO `train` split, and compares
landmark-derived features with the expected values. It does not modify the
source images or annotations and does not train a letter classifier.

Start it from the repository root in the project environment:

```bash
conda activate ktai-hand-gesture
python -m streamlit run apps/asl_landmark_feature_review_app.py
```

By default, the app reads:

- Excel references from `~/Downloads/Untitled spreadsheet.xlsx`
- COCO training images from `data/external/asl_letters_v1/train`
- A landmark cache at `data/reports/asl_letters_v1_train_landmarks.json`

Override those paths when starting the app:

```bash
python -m streamlit run apps/asl_landmark_feature_review_app.py -- \
  --dataset data/external/asl_letters_v1/train \
  --reference "/path/to/ASL feature references.xlsx" \
  --cache data/reports/asl_train_landmarks.json
```

If the workbook path is not available, upload the `.xlsx` file in the app.
Press **Landmarks berekenen voor de train-split** once to run the hand
detector. Subsequent threshold changes reuse those saved landmarks and update
the feature comparisons without rerunning MediaPipe. The app can filter
images that match all scored features, images with mismatches/unknown values,
and images where hand detection needs review. Download the current filtered
image summary as CSV.

Workbook columns are mapped to separate MCP/IP/PIP/DIP angles, index-middle
and middle-ring spread, index-middle crossing, and thumb contact with the
index/middle fingertip. Slash-separated categories are accepted alternatives.
`kan` is omitted from scoring. A `nee` thumb-contact expectation means either
`near` or `far` is acceptable; only `contact` is a mismatch.

The workbook's thumb-MCP values (`B-vormig`, `C-vormig`, `L-vormig`, and
`neutraal`) do not directly correspond to joint-angle categories. For
exploration only, the app approximates B with palmward bend, L with bend away
from the palm, and C with a near index fingertip and a bent thumb IP. These
thumb-shape outputs must be checked against the displayed image and
measurements; they are not validated anatomical rules. J and Z are dynamic ASL
letters, so one still image cannot fully verify their motion.

The cache is tied to the COCO annotation file and the train image filenames,
sizes, and modification times. If the source data changes, calculate landmarks
again to refresh the cache.

## Train-only threshold calibration

The workbook review can also be used to select categorical thresholds on the
COCO `train` images. The tuner groups Roboflow augmentations by the filename
prefix before `.rf.` and reserves a deterministic 20% of source groups as an
internal holdout. It selects settings using only the other groups, then reports
the untouched holdout score. It never reads COCO `valid` or `test`.

Run it from the project environment:

```bash
python -m gesturelab.features.asl_threshold_tuning
```

By default it uses the workbook, train split, and landmark cache listed above.
The output is written to
`experiments/asl_feature_threshold_tuning_01/categorical_thresholds.json`,
alongside `calibration_report.json`. To use another output path or workbook:

```bash
python -m gesturelab.features.asl_threshold_tuning \
  --reference "/path/to/ASL feature references.xlsx" \
  --output-dir experiments/asl_feature_threshold_tuning_02
```

The selection metric maximizes mean per-feature accuracy; exact-image match
rate and per-feature, per-category macro-F1 break ties. Unknown feature
predictions count as incorrect.
Images where MediaPipe found no hand or multiple hands are reported separately
and cannot be fixed by changing categorical thresholds. The reported agreement
is with the current workbook hypotheses, not a human-verified ground truth or
letter-recognition accuracy. The report also flags selected thresholds at the
edge of the searched range. Review those, the train and holdout feature scores,
and the workbook assumptions before using the selected JSON in the separate
Shin model comparison. The all-train score is descriptive; use the disjoint
holdout score as the less-biased estimate of threshold generalization.
