"""Compare landmark-derived hand features with an ASL reference workbook."""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import unicodedata

from openpyxl import load_workbook
from PIL import Image

from gesturelab.datasets.data import Detector, atomic_json
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS
from gesturelab.features.categorical_hand import (
    categorize_hand_measurements,
    measure_hand_features,
)

CACHE_VERSION = 1
REFERENCE_COLUMNS = {
    "duim_mcp": "thumb_mcp_shape",
    "duim_ip": "flexion_thumb_ip",
    "wijsvinger_mcp": "flexion_index_mcp",
    "wijs_pip": "flexion_index_pip",
    "wijs_dip": "flexion_index_dip",
    "middel_mcp": "flexion_middle_mcp",
    "middel_pip": "flexion_middle_pip",
    "middel_dip": "flexion_middle_dip",
    "ring_mcp": "flexion_ring_mcp",
    "ring_pip": "flexion_ring_pip",
    "ring_dip": "flexion_ring_dip",
    "pink_mcp": "flexion_pinky_mcp",
    "pink_pip": "flexion_pinky_pip",
    "pink_dip": "flexion_pinky_dip",
    "spreiding_wijs_middel": "spread_index_middle",
    "spreiding_middel_ring": "spread_middle_ring",
    "kruising_wijs_middel": "crossing_index_middle",
    "duim_contact_wijs": "thumb_tip_index_proximity",
    "duim_contact_middel": "thumb_tip_middle_proximity",
}

_FLEXION_VALUES = {
    "gestrekt": "extended",
    "licht_gebogen": "partly_bent",
    "gebogen": "bent",
}
_THUMB_SHAPES = {
    "neutraal": "not_B-vormig",
    "b_vormig": "B-vormig",
    "niet_b_vormig": "not_B-vormig",
    "c_vormig": "not_B-vormig",
    "l_vormig": "not_B-vormig",
}


def _normalize(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.strip().lower().replace("lich gebogen", "licht gebogen")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _expected_values(feature, cell):
    values = [_normalize(part) for part in str(cell).split("/")]
    if not values or any(not value for value in values):
        raise ValueError(f"Empty reference value for {feature}.")
    if "kan" in values:
        if len(values) != 1:
            raise ValueError(f"'kan' cannot be combined with another value for {feature}.")
        return None

    if feature.startswith("flexion_"):
        mapping = dict(_FLEXION_VALUES)

    elif feature == "thumb_mcp_shape":
        mapping = _THUMB_SHAPES
    elif feature.startswith("spread_"):
        mapping = {"ja": "spread", "nee": "not_spread"}
    elif feature == "crossing_index_middle":
        mapping = {"ja": "suspected_crossed", "nee": "not_crossed"}
    elif feature.startswith("thumb_tip_"):
        mapping = {"ja": "contact", "nee": "no_contact"}
    else:
        raise ValueError(f"No reference mapping exists for {feature}.")

    try:
        allowed = {mapping[value] for value in values}
        return sorted(allowed)
    except KeyError as error:
        raise ValueError(
            f"Unsupported workbook value {error.args[0]!r} for {feature}."
        ) from error


def load_reference_workbook(contents):
    """Parse the user's A–Z feature sheet and map its categories to extractor IDs."""
    workbook = load_workbook(BytesIO(contents), data_only=True, read_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows, None)
    if not header:
        raise ValueError("The reference workbook is empty.")

    feature_columns = {}
    for index, value in enumerate(header[1:], start=1):
        if value is None or not str(value).strip():
            continue
        key = _normalize(value)
        if key not in REFERENCE_COLUMNS:
            raise ValueError(f"Unknown feature column in workbook: {value!r}.")
        feature = REFERENCE_COLUMNS[key]
        if feature in feature_columns.values():
            raise ValueError(f"Duplicate feature column in workbook: {value!r}.")
        feature_columns[index] = feature
    if set(feature_columns.values()) != set(REFERENCE_COLUMNS.values()):
        missing = sorted(set(REFERENCE_COLUMNS.values()) - set(feature_columns.values()))
        raise ValueError(f"Workbook is missing feature columns: {', '.join(missing)}.")

    references = {}
    for row_number, row in enumerate(rows, start=2):
        if not row or row[0] is None or not str(row[0]).strip():
            continue
        label = str(row[0]).strip().upper()
        if label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(label) != 1:
            raise ValueError(f"Invalid letter in workbook row {row_number}: {label!r}.")
        if label in references:
            raise ValueError(f"Duplicate letter {label} in workbook.")
        values = {}
        for column, feature in feature_columns.items():
            raw_value = row[column] if column < len(row) else None
            if raw_value is None or not str(raw_value).strip():
                raise ValueError(f"Missing {feature} value for letter {label}.")
            values[feature] = {
                "raw": (
                    "/".join(dict.fromkeys(
                        "B-vormig" if _normalize(part) == "b_vormig" else "niet B-vormig"
                        for part in str(raw_value).split("/")
                    )) if feature == "thumb_mcp_shape" and _normalize(raw_value) != "kan"
                    else str(raw_value).strip()
                ),
                "allowed": _expected_values(feature, raw_value),
            }
        references[label] = values

    if set(references) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        missing = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(references))
        raise ValueError(f"Workbook is missing letters: {', '.join(missing)}.")
    workbook.close()
    return references


def read_coco_split(split_directory):
    """Validate a single-letter COCO split and resolve contained image paths."""
    split_directory = Path(split_directory).expanduser().resolve()
    annotation_path = split_directory / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise ValueError(f"COCO annotations are missing: {annotation_path}")
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {
        category["id"]: category["name"].strip().upper()
        for category in document.get("categories", [])
    }
    annotations_by_image = defaultdict(list)
    for annotation in document.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    samples = []
    seen_names = set()
    for image in document.get("images", []):
        image_id = image["id"]
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) != 1:
            raise ValueError(
                f"Image {image.get('file_name')!r} must have exactly one annotation."
            )
        label = category_names.get(annotations[0]["category_id"], "")
        if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise ValueError(f"Image has an invalid letter label: {label!r}.")
        filename = image.get("file_name", "")
        image_path = (split_directory / filename).resolve()
        if not image_path.is_relative_to(split_directory) or not image_path.is_file():
            raise ValueError(f"Image path is missing or escapes the split: {filename!r}.")
        if filename in seen_names:
            raise ValueError(f"Duplicate COCO image filename: {filename!r}.")
        seen_names.add(filename)
        samples.append({
            "id": str(image_id),
            "filename": filename,
            "label": label,
            "image_path": str(image_path),
        })
    if not samples:
        raise ValueError(f"No COCO images found in {split_directory}.")
    return samples


def split_fingerprint(split_directory, samples=None):
    split_directory = Path(split_directory).expanduser().resolve()
    annotation_path = split_directory / "_annotations.coco.json"
    samples = samples if samples is not None else read_coco_split(split_directory)
    digest = sha256(annotation_path.read_bytes())
    for sample in sorted(samples, key=lambda item: item["filename"]):
        image_path = Path(sample["image_path"])
        stat = image_path.stat()
        digest.update(
            f"{sample['filename']}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def analyze_split(split_directory, progress=None):
    """Compute and retain world landmarks for a COCO split, without modifying it."""
    split_directory = Path(split_directory).expanduser().resolve()
    samples = read_coco_split(split_directory)
    detector = Detector()
    try:
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample["image_path"]) as image:
                detection = detector.analyze(image)
            sample.update({
                key: detection[key]
                for key in ("status", "n_hands", "world_landmarks", "handedness", "error")
                if key in detection
            })
            if progress:
                progress(index, len(samples))
    finally:
        detector.close()
    return {
        "cache_version": CACHE_VERSION,
        "dataset_path": str(split_directory),
        "dataset_fingerprint": split_fingerprint(split_directory, samples),
        "samples": samples,
    }


def save_landmark_cache(cache, path):
    if cache.get("cache_version") != CACHE_VERSION:
        raise ValueError("Unsupported landmark cache version.")
    atomic_json(path, cache, compact=True)


def load_landmark_cache(path, split_directory):
    path = Path(path)
    if not path.is_file():
        return None
    cache = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cache, dict) or not isinstance(cache.get("samples"), list):
        raise ValueError(f"Invalid landmark cache structure: {path}")
    samples = read_coco_split(split_directory)
    if (cache.get("cache_version") != CACHE_VERSION
            or cache.get("dataset_fingerprint") != split_fingerprint(split_directory, samples)):
        return None
    return cache


def _thumb_mcp_shape(categories, measurements, parameters):
    deviation = measurements["thumb_mcp_signed_deviation_deg"]
    if deviation is None:
        return "unknown"
    return ("B-vormig" if deviation > parameters["workbook_thumb"]["b_min_deviation_deg"]
            else "not_B-vormig")


def categorize_workbook_features(measurements, parameters=None):
    """Map measurements to the feature categories used by workbook comparison."""
    from gesturelab.features.categorical_hand import validate_parameters

    params = validate_parameters(parameters)
    categories = categorize_hand_measurements(measurements, params)
    for feature in categories:
        if feature.startswith("flexion_") and feature != "flexion_thumb_ip":
            angle = measurements[f"{feature.removeprefix('flexion_')}_angle_deg"]
            categories[feature] = (
                "unknown" if angle is None else
                "extended" if angle >= params["finger_flexion"]["extended_min_deg"] else
                "bent" if angle <= params["finger_flexion"]["bent_max_deg"] else "partly_bent"
            )
    for left, right in (("index", "middle"), ("middle", "ring")):
        feature = f"spread_{left}_{right}"
        gap = measurements[feature + "_gap_ratio"]
        depth = measurements[feature + "_depth_ratio"]
        direction = measurements[feature + "_direction_cosine"]
        angles = [measurements[f"{finger}_{joint}_angle_deg"]
                  for finger in (left, right) for joint in ("pip", "dip")]
        if any(value is None for value in (gap, depth, direction, *angles)):
            state = "unknown"
        else:
            state = "spread" if (
                gap >= params["workbook_spread"]["minimum_gap_ratio"]
                and depth <= params["workbook_spread"]["maximum_depth_ratio"]
                and direction > 0
                and min(angles) >= params["workbook_spread"]["minimum_joint_angle_deg"]
            ) else "not_spread"
        categories[feature] = state
    crossing = params["workbook_crossing"]
    gap = measurements["crossing_profile_gap_ratio"]
    depth = measurements["crossing_profile_depth_ratio"]
    joint_angles = [measurements[f"{finger}_{joint}_angle_deg"]
                    for finger in ("index", "middle") for joint in ("pip", "dip")]
    if any(value is None for value in joint_angles):
        categories["crossing_index_middle"] = "unknown"
    elif min(joint_angles) < crossing["minimum_joint_angle_deg"]:
        categories["crossing_index_middle"] = "not_crossed"
    elif gap is None or depth is None:
        categories["crossing_index_middle"] = "unknown"
    else:
        categories["crossing_index_middle"] = (
            "suspected_crossed" if gap < -max(crossing["minimum_swap_ratio"], 1e-6)
            and depth <= crossing["maximum_depth_ratio"] else "not_crossed"
        )
    return {
        **categories,
        "thumb_mcp_shape": _thumb_mcp_shape(categories, measurements, params),
    }


def compare_sample(sample, references, parameters=None):
    """Compare one landmark record to its letter's workbook expectations."""
    params = parameters or PARAMETER_DEFAULTS
    expected = references[sample["label"]]
    if sample.get("status") != "ok" or sample.get("world_landmarks") is None:
        comparisons = {
            key: {
                "expected": value["raw"],
                "actual": "niet beschikbaar",
                "status": "unknown" if value["allowed"] is not None else "not_scored",
            }
            for key, value in expected.items()
        }
        return _sample_result(sample, comparisons)

    measurements = measure_hand_features(sample["world_landmarks"], sample.get("handedness"))
    actual = categorize_workbook_features(measurements, params)
    comparisons = {}
    for feature, cell in expected.items():
        category = actual[feature]
        allowed = cell["allowed"]
        if allowed is None:
            status = "not_scored"
        elif category == "unknown":
            status = "unknown"
        elif feature.startswith("thumb_tip_") and "no_contact" in allowed:
            status = (
                "match"
                if "contact" in allowed or category != "contact"
                else "mismatch"
            )
        else:
            status = "match" if category in allowed else "mismatch"
        comparisons[feature] = {
            "expected": cell["raw"],
            "actual": category,
            "status": status,
            "measurement": _measurement_for(feature, measurements),
        }
    return _sample_result(sample, comparisons)


def _measurement_for(feature, measurements):
    if feature == "thumb_mcp_shape":
        return {
            "thumb_mcp_signed_deviation_deg":
                measurements["thumb_mcp_signed_deviation_deg"],
            "thumb_mcp_angle_deg": measurements["thumb_mcp_angle_deg"],
            "thumb_ip_angle_deg": measurements["thumb_ip_angle_deg"],
            "thumb_tip_index_distance_ratio":
                measurements["thumb_tip_index_distance_ratio"],
        }
    if feature.startswith("flexion_"):
        return {feature.removeprefix("flexion_") + "_angle_deg":
                measurements[feature.removeprefix("flexion_") + "_angle_deg"]}
    if feature.startswith("spread_"):
        return {key: value for key, value in measurements.items()
                if key.startswith(feature + "_")}
    if feature == "crossing_index_middle":
        return {
            "crossing_profile_gap_ratio": measurements["crossing_profile_gap_ratio"],
            "crossing_profile_depth_ratio": measurements["crossing_profile_depth_ratio"],
            "crossing_base_lateral_ratio": measurements["crossing_base_lateral_ratio"],
            "crossing_tip_lateral_ratio": measurements["crossing_tip_lateral_ratio"],
        }
    finger = "index" if feature.endswith("_index_proximity") else "middle"
    key = f"thumb_tip_{finger}_distance_ratio"
    return {key: measurements[key]}


def _sample_result(sample, comparisons):
    statuses = [item["status"] for item in comparisons.values()]
    matches = statuses.count("match")
    mismatches = statuses.count("mismatch")
    unknown = statuses.count("unknown")
    not_scored = statuses.count("not_scored")
    scored = matches + mismatches
    status = (
        "needs_detection_review"
        if sample.get("status") != "ok"
        else "needs_tuning"
        if mismatches or unknown
        else "match"
    )
    return {
        **sample,
        "image_status": status,
        "match_rate": matches / scored if scored else None,
        "matches": matches,
        "mismatches": mismatches,
        "unknown": unknown,
        "not_scored": not_scored,
        "comparisons": comparisons,
    }


def evaluate_samples(samples, references, parameters=None):
    results = [compare_sample(sample, references, parameters) for sample in samples]
    feature_stats = {}
    for feature in references[next(iter(references))]:
        statuses = [
            result["comparisons"][feature]["status"]
            for result in results
        ]
        matches = statuses.count("match")
        mismatches = statuses.count("mismatch")
        unknown = statuses.count("unknown")
        scored = matches + mismatches
        feature_stats[feature] = {
            "feature": feature,
            "matches": matches,
            "mismatches": mismatches,
            "unknown": unknown,
            "not_scored": statuses.count("not_scored"),
            "scored": scored,
            "match_rate": matches / scored if scored else None,
        }
    return results, feature_stats
