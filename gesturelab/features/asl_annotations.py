"""Blind, per-image human annotations for the versioned ASL feature schema."""
from datetime import datetime, timezone
import json
from pathlib import Path

from gesturelab.datasets.data import atomic_json, read_manifest
from gesturelab.features.asl_schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION

ANNOTATION_FILENAME = "asl_feature_annotations.json"
ANNOTATION_SCHEMA_VERSION = 1
NOT_ASSESSABLE = "not_assessable"


def annotation_path(project):
    return Path(project) / "annotations" / ANNOTATION_FILENAME


def annotatable_samples(project, manifest=None):
    manifest = manifest if manifest is not None else read_manifest(project)
    return [
        sample for sample in manifest["samples"]
        if sample.get("included") and sample.get("status") == "ok"
        and sample.get("role") == "development"
    ]


def empty_annotation_file():
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "annotations": [],
    }


def load_annotations(project, manifest=None):
    path = annotation_path(project)
    if not path.is_file():
        return empty_annotation_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_annotation_file(data)
    sample_by_id = {
        sample["id"]: sample for sample in annotatable_samples(project, manifest)
    }
    for annotation in data["annotations"]:
        sample = sample_by_id.get(annotation["sample_id"])
        if sample is None:
            raise ValueError(f"Annotation refers to a missing, excluded, test, or unanalyzed sample: {annotation['sample_id']}.")
        if annotation["image_sha256"] != sample.get("original_sha256", ""):
            raise ValueError(f"Source image changed since annotation: {annotation['sample_id']}.")
    return data


def validate_annotation_file(data):
    if (not isinstance(data, dict)
            or data.get("schema_version") != ANNOTATION_SCHEMA_VERSION
            or data.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or not isinstance(data.get("annotations"), list)):
        raise ValueError("Unsupported ASL feature annotation file.")
    feature_categories = {
        feature["key"]: {category["id"] for category in feature["categories"]
                         if category["id"] != "unknown"}
        for feature in FEATURE_SCHEMA
    }
    seen = set()
    for annotation in data["annotations"]:
        if not isinstance(annotation, dict):
            raise ValueError("Each annotation must be an object.")
        sample_id = annotation.get("sample_id")
        annotator = annotation.get("annotator")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Each annotation needs a sample ID.")
        if not isinstance(annotator, str) or not annotator.strip():
            raise ValueError("Each annotation needs an annotator.")
        identity = sample_id, annotator
        if identity in seen:
            raise ValueError("A sample may have at most one annotation per annotator.")
        seen.add(identity)
        values = annotation.get("features")
        if not isinstance(values, dict) or set(values) - set(feature_categories):
            raise ValueError(f"Annotation for {sample_id} has unknown feature keys.")
        for key, value in values.items():
            if not isinstance(value, str) or (
                value != NOT_ASSESSABLE and value not in feature_categories[key]
            ):
                raise ValueError(f"Invalid human category for {sample_id}/{key}.")
        for field in ("annotated_at", "image_sha256"):
            if not isinstance(annotation.get(field, ""), str):
                raise ValueError(f"{field} for {sample_id} must be text.")
    return data


def save_annotation(project, sample, annotator, features):
    if sample.get("role") != "development" or sample.get("status") != "ok" or not sample.get("included"):
        raise ValueError("Only included development images with successful hand detection can be annotated.")
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("Enter an annotator identifier.")
    candidate = {
        "sample_id": sample["id"],
        "image_sha256": sample.get("original_sha256", ""),
        "annotator": annotator.strip(),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "features": dict(features),
    }
    data = load_annotations(project)
    data["annotations"] = [
        item for item in data["annotations"]
        if (item["sample_id"], item["annotator"]) != (candidate["sample_id"], candidate["annotator"])
    ]
    data["annotations"].append(candidate)
    validate_annotation_file(data)
    atomic_json(annotation_path(project), data)
    return candidate


def agreement_report(data):
    validate_annotation_file(data)
    by_sample = {}
    for annotation in data["annotations"]:
        by_sample.setdefault(annotation["sample_id"], []).append(annotation)
    rows = []
    for feature in FEATURE_SCHEMA:
        compared = 0
        agreements = 0
        for annotations in by_sample.values():
            values = [
                annotation["features"].get(feature["key"])
                for annotation in annotations
                if annotation["features"].get(feature["key"]) != NOT_ASSESSABLE
                and annotation["features"].get(feature["key"]) is not None
            ]
            if len(values) < 2:
                continue
            compared += 1
            agreements += len(set(values)) == 1
        rows.append({
            "feature": feature["key"],
            "double_annotated_samples": compared,
            "exact_agreements": agreements,
            "exact_agreement": agreements / compared if compared else None,
        })
    return rows
