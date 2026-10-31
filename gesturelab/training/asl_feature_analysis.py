"""Permutation importance and grouped-feature ablations for saved ASL runs."""
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from gesturelab.datasets.data import atomic_json
from gesturelab.features.asl_models import input_columns, input_feature_groups
from gesturelab.features.training_inputs import validate_model_input
from gesturelab.training.asl_categorical import _classifier


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_analysis_run(run, require_validation_model):
    run = Path(run).resolve()
    report_path = run / "training.json"
    split_path = run / "split.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if report.get("route") not in ("svm", "mlp"):
        raise ValueError("Feature importance and ablations require an SVM or MLP run.")
    train_path = run / "training_features.csv.gz"
    validation_path = run / "validation_features.csv.gz"
    for path, key in (
        (train_path, "training_features_sha256"),
        (validation_path, "validation_features_sha256"),
    ):
        if _sha256(path) != report.get(key):
            raise ValueError(f"Saved feature rows changed: {path.name}.")
    training = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    columns = report.get("feature_columns")
    groups = input_feature_groups(report.get("input_variant", ""))
    if columns != list(input_columns(report.get("input_variant", ""))):
        raise ValueError("Saved feature-column order differs from the selected ASL input schema.")
    if set(training.columns) != {"id", "label", *columns} or set(validation.columns) != {"id", "label", *columns}:
        raise ValueError("Saved feature table columns are incomplete or unexpected.")
    if training.id.duplicated().any() or validation.id.duplicated().any():
        raise ValueError("Saved feature tables contain duplicate sample IDs.")
    if set(training.id) != set(split["training_ids"]) or set(validation.id) != set(split["validation_ids"]):
        raise ValueError("Saved feature rows differ from the frozen development split.")
    if set(training.label) != set(split["labels"]) or set(validation.label) != set(split["labels"]):
        raise ValueError("Saved feature rows do not contain the frozen label set.")

    validation_bundle = None
    validation_model_path = run / "validation_model.joblib"
    if require_validation_model:
        if not validation_model_path.is_file():
            raise ValueError("This run has no training-only validation model.")
        if _sha256(validation_model_path) != report.get("validation_model_sha256"):
            raise ValueError("Saved validation model changed.")
        validation_bundle = joblib.load(validation_model_path)
        validate_model_input(validation_bundle)
        if validation_bundle["split_sha256"] != _sha256(split_path):
            raise ValueError("Validation model and frozen split do not match.")
    return run, report, split, training, validation, validation_bundle


def _macro_f1(labels, predicted, classes):
    return float(f1_score(labels, predicted, labels=classes, average="macro", zero_division=0))


def permutation_importance(run, output, repeats=20, seed=42):
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Permutation repeats must be a positive integer.")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Analysis output already exists. Choose a new directory.")
    run, report, split, _, validation, bundle = _load_analysis_run(run, True)
    columns = report["feature_columns"]
    groups = input_feature_groups(report["input_variant"])
    classes = split["labels"]
    truth = validation.label.astype(str).to_numpy()
    matrix = validation[columns].to_numpy(dtype=float)
    estimator = bundle["estimator"]
    baseline = _macro_f1(truth, estimator.predict(matrix), classes)
    rng = np.random.default_rng(seed)
    rows = []
    for feature in input_feature_groups(report["input_variant"]):
        indices = [columns.index(column) for column in groups[feature]]
        scores = []
        for _ in range(repeats):
            order = rng.permutation(len(matrix))
            permuted = matrix.copy()
            permuted[:, indices] = matrix[order][:, indices]
            scores.append(_macro_f1(truth, estimator.predict(permuted), classes))
        rows.append({
            "feature": feature,
            "baseline_macro_f1": baseline,
            "permuted_macro_f1_mean": float(np.mean(scores)),
            "permuted_macro_f1_std": float(np.std(scores, ddof=1)) if repeats > 1 else 0.0,
            "importance_macro_f1_drop": baseline - float(np.mean(scores)),
            "repeats": repeats,
        })
    output.mkdir(parents=True)
    table = pd.DataFrame(rows).sort_values(
        "importance_macro_f1_drop", ascending=False, kind="stable")
    table.to_csv(output / "permutation_importance.csv", index=False)
    atomic_json(output / "analysis.json", {
        "method": "grouped one-hot permutation on frozen validation rows",
        "source_run": str(run),
        "dataset_fingerprint": split["dataset_fingerprint"],
        "validation_ids": split["validation_ids"],
        "seed": seed,
        "repeats": repeats,
        "baseline_macro_f1": baseline,
        "note": "Importance is predictive association on this validation split, not causal contribution.",
    })
    return output


def feature_ablation(run, output, seed=42):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Ablation output already exists. Choose a new directory.")
    run, report, split, training, validation, _ = _load_analysis_run(run, False)
    route = report["route"]
    columns = report["feature_columns"]
    classes = split["labels"]
    train_y = training.label.astype(str).to_numpy()
    validation_y = validation.label.astype(str).to_numpy()
    rows = []
    groups = input_feature_groups(report["input_variant"])
    for omitted_feature in (None, *groups):
        selected_columns = [
            column for column in columns
            if omitted_feature is None or column not in groups[omitted_feature]
        ]
        estimator = _classifier(route, len(training), seed)
        started = time.perf_counter()
        estimator.fit(training[selected_columns].to_numpy(dtype=float), train_y)
        fit_seconds = time.perf_counter() - started
        predicted = estimator.predict(validation[selected_columns].to_numpy(dtype=float))
        score = _macro_f1(validation_y, predicted, classes)
        rows.append({
            "omitted_feature": omitted_feature or "none_baseline",
            "macro_f1": score,
            "macro_f1_delta_vs_baseline": None,
            "n_features": len(selected_columns),
            "fit_seconds": fit_seconds,
        })
    baseline = rows[0]["macro_f1"]
    for row in rows:
        row["macro_f1_delta_vs_baseline"] = row["macro_f1"] - baseline
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "feature_ablation.csv", index=False)
    atomic_json(output / "analysis.json", {
        "method": "one feature group removed; refit on frozen training rows",
        "source_run": str(run),
        "route": route,
        "dataset_fingerprint": split["dataset_fingerprint"],
        "training_ids": split["training_ids"],
        "validation_ids": split["validation_ids"],
        "seed": seed,
        "note": "Ablation effect is split-specific and does not establish a causal or final-test result.",
    })
    return output
