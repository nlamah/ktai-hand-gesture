# Where model inputs are created and used

The recorded dataset contains photos, labels, sessions and MediaPipe landmarks.
A training input variant transforms those same landmarks into numeric model
inputs. Changing input variants does not create new photos or a different split.

## Responsibilities

| File | Responsibility |
|---|---|
| `gesturelab/features/finger_features.py` | `compute_finger_features()` calculates the 81 derived geometric measurements. It does not train a classifier or read letter labels. |
| `gesturelab/features/landmark_features.py` | Normalizes image/world coordinates without calculating finger measurements. |
| `gesturelab/features/paper_shin2021.py` | Computes the separate paper distances, axis cosines and their combination from stored image landmarks. |
| `gesturelab/features/inspection.py` | Provides the complete feature view used for recording and inspection. This does not decide which input a model receives. |
| `gesturelab/features/training_inputs.py` | Owns extractor composition, column selection, source dependencies and model-input reconstruction. |
| `gesturelab/training/experiment_plan.py` | Freezes sample IDs, groups, folds, thresholds and source hashes. Splitting uses metadata, not derived features. |
| `gesturelab/training/fitting.py` | Builds the chosen input matrix, fits fresh classifiers per CV fold, fits the final model and saves artifacts. It contains no finger formulas. |
| `gesturelab/training/classifiers.py` | Constructs the classifier and its training-only preprocessing from the frozen profile. |
| `gesturelab/training/distance_weighting.py` | Applies optional index/middle fingertip weighting after preprocessing; feature extraction stays unchanged. |
| `gesturelab/training/mirror_augmentation.py` | Creates reflected development landmarks and selects original/mirror fit rows after splitting; validation/test stay original. |
| `gesturelab/training/neural_network.py` | Defines the fixed MLP preset and balances classes from each fit's training labels only. |
| `gesturelab/training/classifier_comparison.py` | Compares saved classifier runs only after checking matching numeric inputs, policies, original IDs and folds. |
| `gesturelab/training/comparison.py` | Evaluates two or more inputs on the same photo IDs and writes reports. |
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
| `shin2021_distance` | Table 2 distances with per-photo z-score | 190 |
| `shin2021_angle` | Directed-pair x/y/z axis cosines | 630 |
| `shin2021_combined` | Paper distances followed by axis cosines | 820 |
| `shin2021_with_finger_features` | Shin combined values plus 81 finger features | 901 |
| `all_features` | Shin combined values plus 126 landmark and 81 finger values | 1,027 |

See [PAPER_SHIN2021.md](PAPER_SHIN2021.md) for the paper's exact pairs, coordinate
choice and documented ambiguities. All inputs share the selected photo IDs;
the paper variants do not automatically exclude letters or drop invalid samples.

The existing `scripts/train_without_features.py` and `scripts/train_with_features.py` commands
remain shortcuts. The default comparison remains those original two variants.
For any registered variant, use:

```bash
python -m scripts.train_model --experiment experiments/ngt_02 --variant finger_features_only
```

Prepare `ngt_02` first using the workflow's split command. You can train multiple
variants in that same frozen experiment directory. Every variant uses the same
development IDs, folds, classifier, seed and reserved test IDs.
The default `project` profile shares classifier hyperparameters. The optional
`shin2021_asl` profile has published feature-dependent parameters; it is a separate
comparison setting rather than evidence of a feature-only effect.

After training both selected variants, compare development CV:

```bash
python -m scripts.compare_results --experiment experiments/ngt_02 --variants without_features finger_features_only
```

The command prints the HTML report path. Custom pairs use a separate directory,
such as `comparison_cv_without_features_vs_finger_features_only`. In result JSON,
`first` and `second` follow the order passed to `--variants`; for example,
`macro_f1_difference_second_minus_first` is the second score minus the first.
For three or more inputs, pass every desired name to `--variants`. The report
adds a wide per-photo comparison and `pairwise_comparisons.csv` for every pair;
the default two-input output retains its established field names. The
`scripts.run_shin2021` command runs both classifiers with all eight comparison
inputs and verifies identical assignments across classifiers.
Use its `--variants` option to choose a focused comparison, for example
`finger_features_only shin2021_combined shin2021_with_finger_features all_features`.
The 901-value input versus the 820-value input isolates the addition of finger
features to the Shin representation; the 1,027-value input additionally includes
the baseline landmark coordinates. These are project extensions, not features
or results claimed by the paper.

Only after all choices are fixed, add `--final-test`. Comparing many variants
on the final test and selecting the best would turn that test into tuning data.

## Weight the index/middle fingertip distance

For SVM, set `--index-middle-tip-weight` when preparing a new experiment.
The default `1` preserves existing behavior. The weight applies after learned
scaling and is stored in the model pipeline for identical live inference.
Shin inputs weight `shin_dist_08_12`; finger inputs weight
`geom_tip_distance_index_middle`; hybrids with both weight both columns.
See [FINGERTIP_WEIGHTING.md](FINGERTIP_WEIGHTING.md) for exact commands, supported
variants, interpretation and the existing U/V/R confusion analysis.

## Train on reflected landmarks

Set `--mirror-training` when preparing a new experiment to fit each original
training example and its geometric reflection. Validation and reserved test
photos remain original. Feature extraction stays unchanged, and no mirrored
photo files are added to the source dataset. See
[MIRROR_TRAINING.md](MIRROR_TRAINING.md) for the definition, limitations, counts
and commands to compare the resulting model live.

## Compare MLP with the SVM baseline

Prepare a new experiment with `--model mlp` and reuse the original SVM split.
For the controlled 820-value Shin comparison, retain weight 1, no mirror training
and the same preprocessing. The preset has hidden layers `(64, 32)` and no
random internal validation holdout. See [MLP_TRAINING.md](MLP_TRAINING.md) for
training, numeric/OOF comparison and live A/B commands.

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
`TrainingInput.standardize` defaults to true for the existing scaler-based
classifiers. Paper inputs set it to false because their input definition already
specifies per-photo distance normalization and unscaled axis cosines. Their
`definition` string records that convention in the model; it does not change
the baseline's preprocessing. The fitted pipeline is saved with the classifier.
Hybrid inputs use `scaled_prefixes` to scale only their added columns inside the
training fold: `geom_` for the 901-value input; `norm_`, `world_` and `geom_` for
the 1,027-value input. The first 820 Shin values pass through unchanged. LightGBM
does not use this scaler. Both composition and scaling selection belong to the
input definition; the camera continues to call the same reconstruction interface.

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
