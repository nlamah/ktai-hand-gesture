"""Freeze feature-calibration roles without changing dataset metadata."""
from pathlib import Path

import pandas as pd

from gesturelab.datasets.data import atomic_json, fingerprint_manifest, read_manifest
from gesturelab.training.experiments import grouped_folds, groups_for

ROLE_SCHEMA_VERSION = 1


def create_asl_role_map(project, output, grouping="source", folds=5, seed=42):
    project = Path(project).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Role-map output already exists. Use a new path.")
    manifest = read_manifest(project)
    development = [
        sample for sample in manifest["samples"]
        if sample.get("included") and sample.get("status") == "ok"
        and sample.get("role") == "development"
    ]
    if not development:
        raise ValueError("No included, analyzed development images are available.")
    frame = pd.DataFrame([{
        "id": sample["id"], "label": sample["label"],
        "participant": sample["participant"], "session": sample["session"],
        "provenance": sample.get("provenance", {}),
    } for sample in development])
    split, groups = grouped_folds(frame, folds, grouping, seed)
    train_indices, validation_indices = split[0]
    train_groups = set(groups[train_indices])
    validation_groups = set(groups[validation_indices])
    if train_groups & validation_groups:
        raise RuntimeError("Grouped split unexpectedly shares calibration groups.")
    roles = {
        sample_id: "calibration_train"
        for sample_id in frame.iloc[train_indices]["id"]
    }
    roles.update({
        sample_id: "calibration_validation"
        for sample_id in frame.iloc[validation_indices]["id"]
    })
    roles.update({
        sample["id"]: "final_test"
        for sample in manifest["samples"]
        if sample.get("included") and sample.get("role") == "test"
    })
    group_map = dict(zip(frame["id"], groups.astype(str)))
    role_map = {
        "schema_version": ROLE_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint_manifest(manifest),
        "grouping": grouping,
        "n_folds": folds,
        "seed": seed,
        "calibration_fold": 0,
        "roles": roles,
        "groups": group_map,
        "counts": {
            "calibration_train": len(train_indices),
            "calibration_validation": len(validation_indices),
            "final_test": sum(role == "final_test" for role in roles.values()),
        },
        "note": "Role assignment only. No feature calibration, model fitting, or test evaluation was run.",
    }
    atomic_json(output, role_map)
    return output
