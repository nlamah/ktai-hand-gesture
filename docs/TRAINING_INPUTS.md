# Where model inputs are created and used

The recorded dataset contains photos, labels, sessions and MediaPipe landmarks.
A training input variant transforms those same landmarks into numeric model
inputs. Changing input variants does not create new photos or a different split.

## Responsibilities

| File | Responsibility |
|---|---|
| `gesturelab/features/finger_features.py` | `compute_finger_features()` calculates the 81 derived geometric measurements. It does not train a classifier or read letter labels. |
| `gesturelab/features/landmark_features.py` | Normalizes image/world coordinates without calculating finger measurements. |
| `gesturelab/features/inspection.py` | Provides the complete feature view used for recording and inspection. This does not decide which input a model receives. |
| `gesturelab/features/training_inputs.py` | Owns extractor composition, column selection, source dependencies and model-input reconstruction. |
| `gesturelab/training/experiment_plan.py` | Freezes sample IDs, groups, folds, thresholds and source hashes. Splitting uses metadata, not derived features. |
| `gesturelab/training/fitting.py` | Builds the chosen input matrix, fits fresh classifiers per CV fold, fits the final model and saves artifacts. It contains no finger formulas. |
| `gesturelab/training/comparison.py` | Evaluates a selected pair on the same photo IDs and writes reports. |
| `gesturelab/camera/runtime.py` | Asks `model_features()` for inputs; it has no feature-variant branches. |

The feature inspection app also calls the finger calculations to compare them
with your manual assessments. Capture stores calculated features for inspection;
training recomputes its selected inputs from landmarks using frozen settings,
rather than trusting previously cached feature values.

## Available variants

| Variant | Inputs | Values |
|---|---|---:|
| `without_features` | Normalized image and world landmarks | 126 |
| `with_features` | Same landmarks plus geometric finger features | 207 |
| `finger_features_only` | Geometric finger features, without landmark coordinates | 81 |

The existing `scripts/train_without_features.py` and `scripts/train_with_features.py` commands
remain shortcuts. The default comparison remains those original two variants.
For any registered variant, use:

```bash
python -m scripts.train_model --experiment experiments/ngt_02 --variant finger_features_only
```

Prepare `ngt_02` first using the workflow's split command. You can train multiple
variants in that same frozen experiment directory. Every variant uses the same
development IDs, folds, classifier, seed and reserved test IDs.

After training both selected variants, compare development CV:

```bash
python -m scripts.compare_results --experiment experiments/ngt_02 --variants without_features finger_features_only
```

The command prints the HTML report path. Custom pairs use a separate directory,
such as `comparison_cv_without_features_vs_finger_features_only`. In result JSON,
`first` and `second` follow the order passed to `--variants`; for example,
`macro_f1_difference_second_minus_first` is the second score minus the first.
Only after all choices are fixed, add `--final-test`. Comparing many variants
on the final test and selecting the best would turn that test into tuning data.

## Add another variant

For a different selection of existing features, add one entry to
`TRAINING_INPUTS` in `gesturelab/features/training_inputs.py`. For example, normalized
image coordinates alone:

```python
'image_coordinates_only': TrainingInput(
    title='Normalized image coordinates only',
    extractors=(landmarks_for_sample,),
    prefixes=('norm_',),
    expected_features=63,
),
```

No training loop, CLI parser, reporting function or camera loop needs another
conditional branch. `scripts/train_model.py --help` and `scripts/compare_results.py --help`
automatically show registered variants.

For genuinely new measurements, implement a deterministic extractor with this
interface in a clearly named module inside `gesturelab/`:

```python
def custom_features_for_sample(sample, thresholds):
    # Read landmarks and frozen settings, never the letter label or other samples.
    return {'custom_measurement': 0.0}  # Replace with the actual calculation.
```

Import the new extractor into `training_inputs.py` and register it in the
`extractors` tuple. An existing extractor can be combined with others:

```python
extractors=(landmarks_for_sample, custom_features_for_sample),
prefixes=('norm_', 'world_', 'custom_'),
expected_features=127,
```

Only `training_inputs.py` decides which extractors run. Prefix order defines
column order. Extractors must use unique column names; duplicates raise an error
rather than silently replacing measurements.

Extractor source files are automatically included in experiment snapshots and
model input-code hashes. If an extractor calls code in another local helper
module, declare that helper in the same definition, for example
`dependencies=('features/custom_helper.py',)`. No edit to training, comparison, camera,
or experiment hash lists is needed. Dependency paths are relative to `gesturelab/`. Extractor and dependency source
files may be in any of its subpackages, such as `features/`.

The existing `features_for_sample()` function serves recording/inspection and
older saved bundles. New training inputs are composed directly from independent
extractors; they do not need to use this complete feature view.

Do not learn thresholds, normalization statistics or feature selections from
the complete dataset in an extractor. Any learned preprocessing belongs inside
the classifier's training pipeline, fitted separately for each fold.

## Verify the experiment

- `training_features.csv` contains exactly the numeric inputs used by that variant.
- `training.json` lists feature columns and their count.
- `cv_predictions.csv` contains held-out development predictions only.
- `model.joblib` records the input variant and columns for inference.
- `split.json` freezes the shared experiment and includes source-code hashes.

```bash
python -m unittest discover -s tests -p test_training_inputs.py -v
python -m unittest discover -s tests -p test_training.py -v
python -m unittest discover -s tests -v
```

The tests check that baseline extraction never invokes finger geometry, augmented
inputs preserve baseline values, additional variants can train and compare,
and reserved test photos remain separate.

Changes to feature definitions or training code require a new experiment
rather than editing the hashes in an existing split. Existing saved models
continue to check their feature-code hashes; models trained before these feature
code changes need retraining for live recognition with the updated code.
