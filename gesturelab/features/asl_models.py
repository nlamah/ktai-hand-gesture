"""Categorical model-input encoding and an abstaining reference-rule baseline."""
import numpy as np

from gesturelab.features.asl_references import reference_cell, validate_reference_matrix
from gesturelab.features.asl_schema import FEATURE_SCHEMA, UNKNOWN
from gesturelab.features.categorical_hand import extract_categorical_features
from gesturelab.features.paper_shin2021 import ALL_PAIRS, DISTANCE_PAIRS, combined_features

INPUT_MODE = "asl_onehot_and_shin2021_v4"
MIN_RULE_EVIDENCE = 4
MIN_RULE_MATCH_RATIO = 0.8
INPUT_VARIANTS = (
    "categorical",
    "shin2021_combined",
    "categorical_plus_shin2021",
)

FEATURE_COLUMNS = tuple(
    f"{feature['key']}__{category['id']}"
    for feature in FEATURE_SCHEMA
    for category in feature["categories"]
)
SHIN_DISTANCE_COLUMNS = tuple(
    f"shin_dist_{left:02d}_{right:02d}" for left, right in DISTANCE_PAIRS
)
SHIN_AXIS_COLUMNS = tuple(
    f"shin_angle_{left:02d}_{right:02d}_cos_{axis}"
    for left, right in ALL_PAIRS for axis in "xyz"
)
SHIN_COLUMNS = SHIN_DISTANCE_COLUMNS + SHIN_AXIS_COLUMNS
FEATURE_COLUMN_GROUPS = {
    feature["key"]: tuple(
        f"{feature['key']}__{category['id']}"
        for category in feature["categories"]
    )
    for feature in FEATURE_SCHEMA
}
SHIN_FEATURE_GROUPS = {
    "shin_distances": SHIN_DISTANCE_COLUMNS,
    "shin_axis_cosines": SHIN_AXIS_COLUMNS,
}
_FEATURE_CATEGORY_IDS = {
    feature["key"]: tuple(category["id"] for category in feature["categories"])
    for feature in FEATURE_SCHEMA
}


def encode_categories(categories):
    expected = {feature["key"]: {item["id"] for item in feature["categories"]}
                for feature in FEATURE_SCHEMA}
    if not isinstance(categories, dict) or set(categories) != set(expected):
        raise ValueError("Categorical model input does not match the ASL feature schema.")
    encoded = {}
    for feature in FEATURE_SCHEMA:
        key = feature["key"]
        category = categories[key]
        if not isinstance(category, str) or category not in expected[key]:
            raise ValueError(f"Unknown category {category!r} for {key}.")
        for item in feature["categories"]:
            encoded[f"{key}__{item['id']}"] = float(category == item["id"])
    return encoded


def input_columns(variant):
    if variant == "categorical":
        return FEATURE_COLUMNS
    if variant == "shin2021_combined":
        return SHIN_COLUMNS
    if variant == "categorical_plus_shin2021":
        return FEATURE_COLUMNS + SHIN_COLUMNS
    raise ValueError(f"Unknown ASL input variant: {variant!r}.")


def input_feature_groups(variant):
    groups = dict(FEATURE_COLUMN_GROUPS) if variant != "shin2021_combined" else {}
    if variant in ("shin2021_combined", "categorical_plus_shin2021"):
        groups.update(SHIN_FEATURE_GROUPS)
    if variant not in INPUT_VARIANTS:
        raise ValueError(f"Unknown ASL input variant: {variant!r}.")
    return groups


def model_features_for_asl_bundle(bundle, sample):
    if bundle.get("input_mode") != INPUT_MODE:
        raise ValueError("The saved model does not use an ASL categorical/Shin input.")
    variant = bundle.get("config", {}).get("input_variant", "categorical")
    values = {}
    if variant in ("categorical", "categorical_plus_shin2021"):
        categories = extract_categorical_features(
            sample["world_landmarks"], bundle.get("feature_config"),
            sample.get("handedness"),
        )
        values.update(encode_categories(categories))
    if variant in ("shin2021_combined", "categorical_plus_shin2021"):
        values.update(combined_features(sample["image_landmarks"]))
    expected = input_columns(variant)
    if tuple(values) != expected:
        raise RuntimeError("ASL input columns do not match the fixed model schema.")
    return values


def predict_reference_rule(categories, matrix, min_evidence=MIN_RULE_EVIDENCE,
                           min_match_ratio=MIN_RULE_MATCH_RATIO):
    validate_reference_matrix(matrix)
    if isinstance(min_evidence, bool) or not isinstance(min_evidence, int) or min_evidence < 1:
        raise ValueError("Minimum rule evidence must be a positive integer.")
    if (isinstance(min_match_ratio, bool) or not isinstance(min_match_ratio, (int, float))
            or not 0 < min_match_ratio <= 1):
        raise ValueError("Minimum rule match ratio must be in (0, 1].")
    predictions = []
    for label in matrix["labels"]:
        for pattern in matrix["letters"][label]["patterns"]:
            if pattern["status"] != "verified":
                continue
            expected = [
                (feature["key"], reference_cell(pattern, feature["key"])["allowed"])
                for feature in FEATURE_SCHEMA
                if reference_cell(pattern, feature["key"])["state"] == "specified"
            ]
            evidence = [
                (key, allowed)
                for key, allowed in expected
                if categories.get(key, UNKNOWN) != UNKNOWN
            ]
            if len(evidence) < min_evidence:
                continue
            matches = sum(categories[key] in allowed for key, allowed in evidence)
            ratio = matches / len(evidence)
            if ratio >= min_match_ratio:
                predictions.append({
                    "label": label,
                    "pattern": pattern["name"],
                    "evidence": len(evidence),
                    "expected_features": len(expected),
                    "match_ratio": ratio,
                })
    labels = sorted({item["label"] for item in predictions})
    if len(labels) != 1:
        return UNKNOWN, {
            "reason": "no_matching_pattern" if not labels else "ambiguous_patterns",
            "matches": predictions,
        }
    return labels[0], {"reason": "matched", "matches": predictions}


class ASLReferenceRuleClassifier:
    """Predict letters from verified matrix patterns and abstain on uncertainty."""

    def __init__(self, matrix, columns=FEATURE_COLUMNS,
                 min_evidence=MIN_RULE_EVIDENCE,
                 min_match_ratio=MIN_RULE_MATCH_RATIO):
        validate_reference_matrix(matrix)
        if tuple(columns) != FEATURE_COLUMNS:
            raise ValueError("Reference rules require the fixed ASL one-hot feature columns.")
        self.matrix = matrix
        self.columns = tuple(columns)
        self.min_evidence = min_evidence
        self.min_match_ratio = min_match_ratio
        self.classes_ = np.asarray(matrix["labels"], dtype=str)
        self.n_features_in_ = len(self.columns)

    def get_params(self, deep=True):
        return {
            "min_evidence": self.min_evidence,
            "min_match_ratio": self.min_match_ratio,
            "reference_letter_count": len(self.matrix["labels"]),
        }

    def predict(self, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.columns):
            raise ValueError("Reference-rule input has an incompatible feature shape.")
        predictions = []
        index = 0
        for row in matrix:
            categories = {}
            for feature in FEATURE_SCHEMA:
                options = _FEATURE_CATEGORY_IDS[feature["key"]]
                selected = np.flatnonzero(row[index:index + len(options)] > .5)
                if len(selected) != 1:
                    raise ValueError(f"Invalid one-hot values for {feature['key']}.")
                categories[feature["key"]] = options[int(selected[0])]
                index += len(options)
            predictions.append(predict_reference_rule(
                categories, self.matrix, self.min_evidence, self.min_match_ratio)[0])
            index = 0
        return np.asarray(predictions, dtype=str)


def assert_one_hot_rows(encoded_rows):
    """Validate the semantic one-hot invariant before fitting a classifier."""
    for row in encoded_rows:
        for feature in FEATURE_SCHEMA:
            values = [row[f"{feature['key']}__{item['id']}"] for item in feature["categories"]]
            if not np.isclose(sum(values), 1.0):
                raise ValueError(f"Feature {feature['key']} must have exactly one active category.")
