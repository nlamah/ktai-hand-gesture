# Compare an MLP with the original SVM baseline

This experiment changes the classifier while keeping the original baseline's
data, 820 Shin input values, preprocessing policy and exact grouped folds. It
does not combine the MLP change with fingertip weighting or mirror augmentation.
All 26 original classes, including J and Z, remain available.

## Fixed configuration

| Setting | SVM baseline | MLP candidate |
|---|---|---|
| Input | `shin2021_combined`, 820 values | Identical values and column order |
| Normalization | Existing per-photo distance z-score and axis cosines | Identical; no additional scaler on this variant |
| Distance weight | 1 | 1 |
| Mirror training | Off | Off |
| Development/test photos and folds | Frozen original assignment | Reused exactly with `--split-from` |
| Class balancing | `class_weight="balanced"` | Equivalent inverse-frequency sample weights computed in each training fit |
| Classifier | SVC, RBF, existing project profile | MLP with hidden layers `(64, 32)`, ReLU, Adam |
| Random seed | 42 | 42 |

The MLP preset uses `alpha=0.0001`, learning rate `0.001`, `max_iter=300`,
`tol=0.0001`, `n_iter_no_change=20` and automatic batch size. `early_stopping=False`
avoids introducing a random internal validation split that ignores source
groups. Training may stop when its training loss no longer improves. No outer
validation or reserved-test labels are used for stopping or class weighting.

Implementation: `gesturelab/training/neural_network.py` subclasses sklearn's
`MLPClassifier` only to supply balanced fit weights. Prediction uses sklearn's
normal learned network. The existing scikit-learn 1.8.0 dependency supports
weighted MLP fitting; no PyTorch, TensorFlow or GPU dependency is added.
See the official [MLPClassifier documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPClassifier.html).

MLPs are sensitive to input scale. For this controlled experiment, the existing
Shin normalization is retained for both classifiers. The 190 distance values
already have a per-photo z-score; axis cosines lie in [-1, 1]. An additional
dataset-fitted scaler would be a separate preprocessing change. Other registered
variants use the same existing scaler policy for SVM and MLP, fitted inside
each training fold. See sklearn's [scaling guidance](https://scikit-learn.org/1.8/modules/neural_networks_supervised.html#tips-on-practical-use).

This compares two specified classifier configurations, not the best achievable
SVM versus the best achievable MLP. Network connection weights are not a single
interpretable importance value for each joint distance.

## 1. Prepare and train only the MLP

Run from the project root in the `ktai-hand-gesture` environment:

```bash
python -m scripts.prepare_split --project data/asl_kaggle --out experiments/asl_mlp_01 --model mlp --grouping source --folds 3 --seed 42 --allow-no-test --split-from experiments/shin2021_finger_kaggle_01/svm --index-middle-tip-weight 1
python -m scripts.train_model --experiment experiments/asl_mlp_01 --variant shin2021_combined
```

Do not add `--mirror-training` for this comparison. The original SVM is preserved
and does not need to be trained again. Use a new output name for a subsequent
run. The new experiment freezes current training code, preset and dependencies;
its template imports only the verified original photo/fold assignments.

The command trains one fresh MLP per fold, then a final MLP on all usable
development photos. It performs no parameter sweep or final-test evaluation.
Training the complete real dataset can take time. The implementation tests use
synthetic data; no real-data MLP score is claimed until this command finishes.

Outputs are under `experiments/asl_mlp_01/shin2021_combined/`:

- `model.joblib`: final model for webcam use.
- `cv_predictions.csv`: one OOF prediction per original development photo.
- `cv_folds.csv`: fold scores, timings, optimizer iterations and iteration-limit flags.
- `training.json`: pooled/mean-fold metrics, fixed parameters, final training loss,
  class balancing and iteration diagnostics.
- `training_features.csv`: unchanged raw model inputs for the numeric comparison.

If sklearn reports that the iteration limit was reached, inspect the recorded
diagnostics. Reaching a limit is not a claim of convergence. Any later change
to the preset needs a separately recorded development run; do not tune using a
reserved final test.

## 2. Compare saved development predictions

```bash
python -m scripts.compare_classifiers --baseline experiments/shin2021_finger_kaggle_01/svm --candidate experiments/asl_mlp_01 --variant shin2021_combined --out exports/mlp_vs_svm_01
```

Open `exports/mlp_vs_svm_01/report.html`. This command fits no model and generates
no new test predictions. Before reporting it checks:

- Original train/validation/test IDs, classes and all grouped folds.
- Dataset fingerprint, feature settings, seed and relevant numeric library versions.
- Model/split integrity, actual model dimensions, input definition and column order.
- Matching preprocessing, class-balancing policy, distance weight and augmentation.
- Every saved numeric training row, matched by ID regardless of CSV row order.
- Exact OOF ID/fold multiplicities and equal true labels between runs.

The report includes independently recomputed pooled accuracy/macro-F1, fold
means/sample SD, per-letter precision/recall/F1/support, count/normalized confusion
matrices and paired predictions with corrected/new/shared errors. Source hashes
and configurations are in `result.json`. Numeric matrix verification reads both
large CSVs and can take a while.

Existing artifacts describe historical execution; they do not prove an
unrecorded fit trace. Filename blocks are not verified participants, and this
is development CV rather than independent final performance. Repeated model
selection on these results can be optimistic. Use new final-test recordings
after fixing the model choice.

## 3. Compare the actual models live

```bash
python -m scripts.research register --model experiments/asl_mlp_01/shin2021_combined/model.joblib --experiment experiments/asl_mlp_01 --alphabet ASL --name "MLP 64-32, original Shin inputs" --out exports/research_setup/mlp_01.json
python -m apps.webcam ab --model-a exports/research_setup/baseline.json --model-b exports/research_setup/mlp_01.json --out exports/live_svm_vs_mlp_01 --intended-label R --countdown 2
```

A is the original SVM; B is the MLP. Both get the same frame and detector output,
each using its stored classifier pipeline. Try both hands and restart with
`--intended-label U` or `V`. The label is for assessment only; all 26 outputs
remain possible. Default predictions are raw, without smoothing or confidence
percentages. Live exploration is development evidence, not the independent
final test. See [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).
