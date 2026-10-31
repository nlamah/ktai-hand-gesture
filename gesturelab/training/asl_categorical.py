"""Frozen validation and model fitting for ASL categorical inputs."""
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.svm import SVC

from gesturelab.datasets.data import (
    atomic_json, fingerprint, fingerprint_manifest, read_manifest,
)
from gesturelab.features.asl_models import (
    ASLReferenceRuleClassifier, INPUT_MODE, INPUT_VARIANTS, input_columns,
    assert_one_hot_rows, encode_categories,
)
from gesturelab.features.asl_references import load_reference_matrix, validate_reference_matrix
from gesturelab.features.asl_schema import FEATURE_SCHEMA_VERSION, UNKNOWN
from gesturelab.features.categorical_hand import extract_categorical_features, validate_parameters
from gesturelab.features.inspection import FEATURE_VERSION
from gesturelab.features.paper_shin2021 import VERSION as SHIN_FEATURE_VERSION, combined_features
from gesturelab.features.asl_roles import ROLE_SCHEMA_VERSION
from gesturelab.features.asl_calibration import _validate_role_assignments
from gesturelab.training.experiments import metrics
from gesturelab.training.neural_network import BalancedMLPClassifier, mlp_parameters, training_diagnostics

MODEL_ROUTES = ("svm", "mlp", "rules")


def _read_role_inputs(project, role_map_path, model_name):
    project = Path(project).resolve()
    role_map_path = Path(role_map_path).resolve()
    manifest = read_manifest(project)
    dataset_fingerprint = fingerprint_manifest(manifest)
    role_map = json.loads(role_map_path.read_text(encoding="utf-8"))
    if role_map.get("schema_version") != ROLE_SCHEMA_VERSION:
        raise ValueError("Unsupported ASL research role map.")
    if role_map.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Role map and dataset do not match.")
    _validate_role_assignments(project, role_map, manifest)
    if not isinstance(model_name, str) or model_name not in MODEL_ROUTES:
        raise ValueError(f"Choose one of: {', '.join(MODEL_ROUTES)}.")
    records = {
        sample["id"]: sample
        for sample in manifest["samples"]
        if sample.get("included") and sample.get("status") == "ok"
        and sample.get("role") == "development"
    }
    training_ids = [
        sample_id for sample_id, role in role_map["roles"].items()
        if role == "calibration_train"
    ]
    validation_ids = [
        sample_id for sample_id, role in role_map["roles"].items()
        if role == "calibration_validation"
    ]
    if not training_ids or not validation_ids:
        raise ValueError("Both development training and validation roles are required.")
    if set(training_ids) | set(validation_ids) != set(records):
        raise ValueError("Research roles do not cover the eligible development photos.")
    training = [records[sample_id] for sample_id in training_ids]
    validation = [records[sample_id] for sample_id in validation_ids]
    train_labels = {sample["label"] for sample in training}
    validation_labels = {sample["label"] for sample in validation}
    if len(train_labels) < 2 or train_labels != validation_labels:
        raise ValueError("Training and validation must each contain the same two or more labels.")
    return project, role_map_path, role_map, dataset_fingerprint, training, validation


def _encoded_rows(samples, parameters, input_variant):
    columns = input_columns(input_variant)
    rows = []
    for sample in samples:
        values = {}
        if input_variant in ("categorical", "categorical_plus_shin2021"):
            categories = extract_categorical_features(
                sample["world_landmarks"], parameters, sample.get("handedness"),
            )
            values.update(encode_categories(categories))
        if input_variant in ("shin2021_combined", "categorical_plus_shin2021"):
            values.update(combined_features(sample["image_landmarks"]))
        if tuple(values) != columns:
            raise RuntimeError("ASL input columns do not match the selected variant.")
        rows.append(values)
    if input_variant != "shin2021_combined":
        assert_one_hot_rows(rows)
    frame = pd.DataFrame(rows, columns=columns, dtype=float)
    return frame.to_numpy(dtype=float)


def _classifier(model_name, n_samples, seed):
    if model_name == "svm":
        return SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    if model_name == "mlp":
        return BalancedMLPClassifier(**mlp_parameters(seed))
    raise ValueError(f"No trainable classifier for route {model_name!r}.")


def _label_scores(samples, predicted, labels):
    truth = [sample["label"] for sample in samples]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0,
    )
    return {
        **metrics(truth, predicted, labels),
        "labels": list(labels),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(labels) + [UNKNOWN]).tolist(),
    }


def _feature_source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = (
        "features/asl_schema.py", "features/categorical_hand.py",
        "features/asl_models.py", "features/paper_shin2021.py",
        "training/asl_categorical.py",
        "training/neural_network.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in files
    }


def train_asl_categorical(project, role_map_path, output, model="svm", thresholds_path=None,
                          references_path=None, seed=42, input_variant="categorical"):
    """Compare one fixed model route on validation and save a development-only model."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Model output already exists. Use a new directory.")
    project, role_map_path, role_map, dataset_fingerprint, training, validation = _read_role_inputs(
        project, role_map_path, model)
    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"Choose one of: {', '.join(INPUT_VARIANTS)}.")
    if model == "rules" and input_variant != "categorical":
        raise ValueError("Reference rules use categorical inputs only.")
    if thresholds_path:
        parameters = validate_parameters(json.loads(Path(thresholds_path).read_text(encoding="utf-8")))
    else:
        parameters = validate_parameters()
    labels = sorted({sample["label"] for sample in training})
    columns = input_columns(input_variant)
    train_x = _encoded_rows(training, parameters, input_variant)
    validation_x = _encoded_rows(validation, parameters, input_variant)
    train_y = np.asarray([sample["label"] for sample in training], dtype=str)
    validation_y = np.asarray([sample["label"] for sample in validation], dtype=str)
    split = {
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint,
        "role_map_sha256": hashlib.sha256(role_map_path.read_bytes()).hexdigest(),
        "labels": labels,
        "grouping": role_map["grouping"],
        "seed": seed,
        "development_ids": sorted([sample["id"] for sample in training + validation]),
        "training_ids": sorted(sample["id"] for sample in training),
        "validation_ids": sorted(sample["id"] for sample in validation),
        "test_ids": sorted(
            sample_id for sample_id, role in role_map["roles"].items()
            if role == "final_test"
        ),
        "test_evaluated": False,
        "synthetic": any(
            sample.get("provenance", {}).get("type") == "synthetic"
            for sample in training + validation
        ),
    }
    estimator = None
    validation_estimator = None
    warnings_seen = []
    if model == "rules":
        if references_path is None:
            raise ValueError("The rules route requires a reviewed reference matrix.")
        matrix = load_reference_matrix(references_path)
        if set(labels) - set(matrix["labels"]):
            raise ValueError("Reference matrix does not contain every training label.")
        matrix = {
            **matrix,
            "labels": labels,
            "letters": {label: matrix["letters"][label] for label in labels},
        }
        validate_reference_matrix(matrix)
        if not any(
            pattern["status"] == "verified"
            for label in labels
            for pattern in matrix["letters"][label]["patterns"]
        ):
            raise ValueError("At least one verified reference pattern is required for the rules route.")
        estimator = ASLReferenceRuleClassifier(matrix)
        validation_predicted = estimator.predict(validation_x)
        selected_reference_matrix = matrix
    else:
        estimator = _classifier(model, len(training), seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(train_x, train_y)
        warnings_seen = [str(warning.message) for warning in caught]
        validation_estimator = estimator
        validation_predicted = estimator.predict(validation_x)
        selected_reference_matrix = None

    validation_scores = _label_scores(validation, validation_predicted, labels)
    if model != "rules":
        full_development = training + validation
        full_x = np.vstack((train_x, validation_x))
        full_y = np.asarray([sample["label"] for sample in full_development], dtype=str)
        estimator = _classifier(model, len(full_development), seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(full_x, full_y)
        warnings_seen.extend(str(warning.message) for warning in caught)
    if fingerprint(project) != split["dataset_fingerprint"]:
        raise ValueError("Dataset changed during model fitting. The result is not a valid frozen run.")
    output.mkdir(parents=True)
    atomic_json(output / "split.json", split)
    split_hash = hashlib.sha256((output / "split.json").read_bytes()).hexdigest()
    bundle = {
        "estimator": estimator,
        "config": {
            "model": "asl_reference_rules" if model == "rules" else model,
            "feature_set": input_variant,
            "input_variant": input_variant,
            "input_mode": INPUT_MODE,
            "columns": list(columns),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "shin_feature_version": SHIN_FEATURE_VERSION if input_variant != "categorical" else None,
            "thresholds": parameters,
        },
        "input_mode": INPUT_MODE,
        "feature_config": parameters,
        "feature_version": FEATURE_VERSION,
        "input_definition": (
            FEATURE_SCHEMA_VERSION if input_variant == "categorical"
            else f"{FEATURE_SCHEMA_VERSION}+{SHIN_FEATURE_VERSION}"
            if input_variant == "categorical_plus_shin2021" else SHIN_FEATURE_VERSION
        ),
        "input_variant": input_variant,
        "input_preprocessing": {
            "categorical_encoding": "fixed-schema one-hot" if input_variant != "shin2021_combined" else None,
            "scaling": "none",
            "shin_distances": "per-image z-score",
            "shin_axes": "direction cosines",
        },
        "feature_code_hashes": _feature_source_hashes(),
        "labels": labels,
        "candidate": f"{model}__asl_categorical",
        "synthetic": split["synthetic"],
        "split_sha256": split_hash,
    }
    if selected_reference_matrix is not None:
        bundle["references"] = selected_reference_matrix
    joblib.dump(bundle, output / "model.joblib")
    if validation_estimator is not None:
        joblib.dump({**bundle, "estimator": validation_estimator},
                    output / "validation_model.joblib")
    training_features = pd.DataFrame(train_x, columns=columns)
    training_features.insert(0, "label", train_y)
    training_features.insert(0, "id", [sample["id"] for sample in training])
    validation_features = pd.DataFrame(validation_x, columns=columns)
    validation_features.insert(0, "label", validation_y)
    validation_features.insert(0, "id", [sample["id"] for sample in validation])
    training_features.to_csv(output / "training_features.csv.gz", index=False, compression="gzip")
    validation_features.to_csv(output / "validation_features.csv.gz", index=False, compression="gzip")
    predictions = pd.DataFrame([
        {
            "id": sample["id"], "label": sample["label"],
            "prediction": str(prediction),
            "condition": sample.get("condition", ""),
            "session": sample.get("session", ""),
            "participant": sample.get("participant", ""),
        }
        for sample, prediction in zip(validation, validation_predicted)
    ])
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    report = {
        "status": "complete",
        "route": model,
        "input_definition": bundle["input_definition"],
        "input_variant": input_variant,
        "n_features": len(columns),
        "feature_columns": list(columns),
        "thresholds": parameters,
        "validation": validation_scores,
        "fit_diagnostics": training_diagnostics(estimator) if model == "mlp" else {},
        "warnings": warnings_seen,
        "training_ids": split["training_ids"],
        "validation_ids": split["validation_ids"],
        "test_evaluated": False,
        "dataset_fingerprint": split["dataset_fingerprint"],
        "feature_code_hashes": bundle["feature_code_hashes"],
        "training_features_sha256": hashlib.sha256(
            (output / "training_features.csv.gz").read_bytes()).hexdigest(),
        "validation_features_sha256": hashlib.sha256(
            (output / "validation_features.csv.gz").read_bytes()).hexdigest(),
        "validation_model_sha256": (
            hashlib.sha256((output / "validation_model.joblib").read_bytes()).hexdigest()
            if validation_estimator is not None else None
        ),
        "resolved_classifier_parameters": {
            key: value if isinstance(value, (str, int, float, bool, list, dict, type(None)))
            else str(value)
            for key, value in estimator.get_params(deep=False).items()
        },
        "note": "Validation is development evidence. The saved classifier is refit on all development images; reserved test images were not scored.",
    }
    atomic_json(output / "training.json", report)
    if fingerprint(project) != split["dataset_fingerprint"]:
        raise ValueError("Dataset changed during model fitting. The result is not a valid frozen run.")
    return output
