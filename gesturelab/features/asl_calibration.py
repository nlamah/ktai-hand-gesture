"""Choose categorical feature thresholds using blind human development annotations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from gesturelab.datasets.data import atomic_json, fingerprint_manifest, read_manifest
from gesturelab.features.asl_annotations import (
    NOT_ASSESSABLE, load_annotations, validate_annotation_file,
)
from gesturelab.features.asl_schema import (
    FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION,
)
from gesturelab.features.categorical_hand import extract_categorical_features, validate_parameters
from gesturelab.features.asl_roles import ROLE_SCHEMA_VERSION
from gesturelab.training.experiments import groups_for


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_configurations(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate file must contain a nonempty 'candidates' list.")
    normalized = [validate_parameters(candidate) for candidate in candidates]
    unique = []
    seen = set()
    for candidate in normalized:
        signature = json.dumps(candidate, sort_keys=True)
        if signature not in seen:
            unique.append(candidate)
            seen.add(signature)
    baseline = validate_parameters()
    if json.dumps(baseline, sort_keys=True) not in seen:
        unique.insert(0, baseline)
    return unique


def _validate_role_assignments(project, role_map, manifest=None):
    manifest = manifest if manifest is not None else read_manifest(project)
    samples = manifest["samples"]
    samples_by_id = {sample["id"]: sample for sample in samples}
    eligible = [
        sample for sample in samples
        if sample.get("included") and sample.get("status") == "ok"
        and sample.get("role") == "development"
    ]
    expected_ids = {sample["id"] for sample in eligible}
    roles = role_map.get("roles")
    groups = role_map.get("groups")
    if not isinstance(roles, dict) or not isinstance(groups, dict):
        raise ValueError("Role map must contain role and group assignments.")
    actual_ids = {
        sample_id for sample_id, role in roles.items()
        if role in ("calibration_train", "calibration_validation")
    }
    if actual_ids != expected_ids:
        raise ValueError("Calibration roles must cover exactly the eligible development images.")
    if role_map.get("grouping") not in ("session", "participant", "source"):
        raise ValueError("Role map has an unsupported grouping policy.")
    frame = pd.DataFrame([{
        "id": sample["id"], "label": sample["label"],
        "participant": sample["participant"], "session": sample["session"],
        "provenance": sample.get("provenance", {}),
    } for sample in eligible])
    actual_groups = dict(zip(frame["id"], groups_for(frame, role_map["grouping"]).astype(str)))
    if any(groups.get(sample_id) != group
           for sample_id, group in actual_groups.items()):
        raise ValueError("Role map source groups do not match the dataset.")
    group_roles = {}
    for sample_id, role in roles.items():
        if role == "final_test" and (
            sample_id not in samples_by_id or samples_by_id[sample_id].get("role") != "test"
        ):
            raise ValueError("Final-test assignment does not refer to a reserved test image.")
        if role not in ("calibration_train", "calibration_validation", "final_test"):
            raise ValueError(f"Unknown research role {role!r} for {sample_id}.")
        if role in ("calibration_train", "calibration_validation"):
            group = groups[sample_id]
            previous_role = group_roles.setdefault(group, role)
            if previous_role != role:
                raise ValueError("Calibration roles split a source group between train and validation.")


def _consensus_samples(manifest, roles, annotation_file, partition):
    samples = {
        sample["id"]: sample
        for sample in manifest["samples"]
        if sample.get("included") and sample.get("status") == "ok"
        and sample.get("role") == "development"
    }
    annotations_by_sample = {}
    for annotation in annotation_file["annotations"]:
        sample_id = annotation["sample_id"]
        assigned_role = roles.get(sample_id)
        if assigned_role not in ("calibration_train", "calibration_validation"):
            raise ValueError(f"Annotation is not assigned to a calibration partition: {sample_id}.")
        if sample_id not in samples:
            raise ValueError(f"Calibration annotation does not refer to an eligible development image: {sample_id}.")
        if assigned_role == partition:
            annotations_by_sample.setdefault(sample_id, []).append(annotation["features"])

    result = []
    for sample_id, reviewer_values in annotations_by_sample.items():
        consensus = {}
        disagreements = {}
        not_assessable = []
        for feature in FEATURE_SCHEMA:
            key = feature["key"]
            reported = [values[key] for values in reviewer_values if key in values]
            values = {value for value in reported if value != NOT_ASSESSABLE}
            if NOT_ASSESSABLE in reported:
                not_assessable.append(key)
            elif len(values) == 1:
                consensus[key] = next(iter(values))
            elif len(values) > 1:
                disagreements[key] = sorted(values)
        if consensus:
            result.append({
                "sample": samples[sample_id],
                "truth": consensus,
                "disagreements": disagreements,
                "not_assessable": not_assessable,
            })
    if not result:
        raise ValueError(f"No agreed, assessable feature annotations in {partition}.")
    return result


def _score(samples, parameters):
    per_feature = {}
    all_scores = []
    for sample in samples:
        record = sample["sample"]
        predicted = extract_categorical_features(
            record["world_landmarks"], parameters, record.get("handedness"),
        )
        for key, truth in sample["truth"].items():
            per_feature.setdefault(key, {"truth": [], "predicted": []})
            per_feature[key]["truth"].append(truth)
            per_feature[key]["predicted"].append(predicted[key])

    feature_rows = []
    for feature in FEATURE_SCHEMA:
        key = feature["key"]
        values = per_feature.get(key)
        if not values:
            continue
        labels = sorted(set(values["truth"]) | set(values["predicted"]))
        macro_f1 = float(f1_score(
            values["truth"], values["predicted"], labels=labels,
            average="macro", zero_division=0,
        ))
        coverage = float(np.mean([value != "unknown" for value in values["predicted"]]))
        feature_rows.append({
            "feature": key,
            "n": len(values["truth"]),
            "macro_f1": macro_f1,
            "accuracy": float(np.mean(np.asarray(values["truth"]) == np.asarray(values["predicted"]))),
            "coverage": coverage,
        })
        all_scores.append(macro_f1)
    if not all_scores:
        raise ValueError("No annotated features could be scored.")
    return {
        "n_samples": len(samples),
        "n_feature_annotations": sum(row["n"] for row in feature_rows),
        "n_conflicting_features_excluded": sum(
            len(sample["disagreements"]) for sample in samples),
        "n_not_assessable_features_excluded": sum(
            len(sample["not_assessable"]) for sample in samples),
        "macro_f1_mean_per_feature": float(np.mean(all_scores)),
        "coverage_mean_per_feature": float(np.mean([row["coverage"] for row in feature_rows])),
        "features": feature_rows,
    }


def calibrate_asl_thresholds(project, role_map_path, candidate_path, output):
    """Select thresholds on calibration-train labels and report untouched validation labels."""
    project = Path(project).resolve()
    role_map_path = Path(role_map_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Calibration output already exists. Choose a new path.")
    manifest = read_manifest(project)
    dataset_fingerprint = fingerprint_manifest(manifest)
    role_map = json.loads(role_map_path.read_text(encoding="utf-8"))
    if (role_map.get("schema_version") != ROLE_SCHEMA_VERSION
            or role_map.get("dataset_fingerprint") != dataset_fingerprint):
        raise ValueError("Role map is incompatible with the current dataset.")
    if not isinstance(role_map.get("roles"), dict):
        raise ValueError("Role map has no sample assignments.")
    groups = role_map.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("Role map has no group assignments.")
    train_groups = {
        groups[sample_id] for sample_id, role in role_map["roles"].items()
        if role == "calibration_train" and sample_id in groups
    }
    validation_groups = {
        groups[sample_id] for sample_id, role in role_map["roles"].items()
        if role == "calibration_validation" and sample_id in groups
    }
    if not train_groups or not validation_groups or train_groups & validation_groups:
        raise ValueError("Calibration roles must contain separate, nonempty source groups.")
    _validate_role_assignments(project, role_map, manifest)
    annotations = load_annotations(project, manifest)
    validate_annotation_file(annotations)
    candidates = _candidate_configurations(candidate_path)
    training = _consensus_samples(
        manifest, role_map["roles"], annotations, "calibration_train")
    validation = _consensus_samples(
        manifest, role_map["roles"], annotations, "calibration_validation")

    scored = []
    for index, parameters in enumerate(candidates):
        scored.append({
            "candidate": index,
            "parameters": parameters,
            "training": _score(training, parameters),
        })
    selected = max(
        scored,
        key=lambda row: (
            row["training"]["macro_f1_mean_per_feature"],
            row["training"]["coverage_mean_per_feature"],
            -row["candidate"],
        ),
    )
    baseline = validate_parameters()
    report = {
        "status": "complete",
        "schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "role_map_sha256": _file_sha256(role_map_path),
        "candidate_file_sha256": _file_sha256(candidate_path),
        "annotation_file_sha256": _file_sha256(
            Path(project) / "annotations" / "asl_feature_annotations.json"),
        "selected_candidate": selected["candidate"],
        "selected_parameters": selected["parameters"],
        "training_scores": scored,
        "baseline_validation": _score(validation, baseline),
        "selected_validation": _score(validation, selected["parameters"]),
        "selection_policy": "Highest mean per-feature macro-F1 on calibration_train; coverage then input order break ties.",
        "limitations": [
            "Validation data is not used to select the thresholds.",
            "Only unanimous human assessments are scored; disagreements and not-assessable values are excluded.",
            "This is feature-threshold calibration, not letter classification or a final test evaluation.",
            "No model was trained and final_test samples were not read for scoring.",
        ],
    }
    output.mkdir(parents=True)
    atomic_json(output / "calibration.json", report)
    atomic_json(output / "categorical_thresholds.json", selected["parameters"])
    return output
