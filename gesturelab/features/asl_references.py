"""Editable ASL letter references kept separate from detector predictions."""
from copy import deepcopy
from html import escape
import csv
from io import StringIO
import json
from pathlib import Path

from gesturelab.datasets.data import atomic_json
from gesturelab.features.asl_schema import (
    FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, LETTERS, schema_document,
)

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "config" / "asl_letter_references.json"
CELL_STATES = ("unreviewed", "specified", "not_relevant")
PATTERN_STATES = ("concept", "verified")


def default_reference_matrix(labels=LETTERS):
    return {
        "schema_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "labels": list(labels),
        "letters": {
            label: {
                "example": {"path": "", "source": "", "reviewer": ""},
                "patterns": [{
                    "name": "default",
                    "status": "concept",
                    "source": "",
                    "reviewer": "",
                    "note": "",
                    "features": {},
                }],
            }
            for label in labels
        },
    }


def _validate_cell(feature, value):
    if not isinstance(value, dict) or value.get("state") not in CELL_STATES:
        raise ValueError(f"Invalid reference state for {feature}.")
    allowed = value.get("allowed", [])
    category_ids = {item["id"] for item in next(
        spec for spec in FEATURE_SCHEMA if spec["key"] == feature)["categories"]}
    if not isinstance(allowed, list) or len(allowed) != len(set(allowed)):
        raise ValueError(f"Allowed categories for {feature} must be a unique list.")
    if any(item not in category_ids or item == "unknown" for item in allowed):
        raise ValueError(f"Unknown or automatic-only category in {feature}.")
    if value["state"] == "specified" and not allowed:
        raise ValueError(f"Specified reference for {feature} needs one or more expected categories.")
    if value["state"] != "specified" and allowed:
        raise ValueError(f"Only specified references may contain expected categories for {feature}.")
    if not isinstance(value.get("note", ""), str):
        raise ValueError(f"Reference note for {feature} must be text.")


def validate_reference_matrix(matrix):
    if not isinstance(matrix, dict) or matrix.get("schema_version") != 1:
        raise ValueError("Unsupported ASL reference matrix format.")
    if matrix.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Reference matrix uses an incompatible feature schema version.")
    labels = matrix.get("labels")
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise ValueError("Reference labels must be a nonempty unique list.")
    letters = matrix.get("letters")
    if not isinstance(letters, dict) or set(letters) != set(labels):
        raise ValueError("Every configured label needs one reference entry.")
    feature_keys = {spec["key"] for spec in FEATURE_SCHEMA}
    for label in labels:
        entry = letters[label]
        if not isinstance(entry, dict):
            raise ValueError(f"ASL reference entry for {label} must be an object.")
        example = entry.get("example")
        if example is None:
            example = {"path": "", "source": "", "reviewer": ""}
        if not isinstance(example, dict) or any(
            not isinstance(example.get(field, ""), str)
            for field in ("path", "source", "reviewer")
        ):
            raise ValueError(f"ASL example metadata for {label} must contain text fields.")
        if example.get("path", "").strip() and not all(
            example.get(field, "").strip() for field in ("source", "reviewer")
        ):
            raise ValueError(f"ASL example for {label} needs source and reviewer.")
        patterns = entry.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Label {label} needs at least one allowed pattern.")
        names = set()
        for pattern in patterns:
            if not isinstance(pattern, dict) or not isinstance(pattern.get("name"), str) or not pattern["name"].strip():
                raise ValueError(f"Each pattern for {label} needs a name.")
            if pattern["name"] in names:
                raise ValueError(f"Duplicate pattern name for {label}: {pattern['name']}.")
            names.add(pattern["name"])
            if pattern.get("status") not in PATTERN_STATES:
                raise ValueError(f"Invalid pattern status for {label}/{pattern['name']}.")
            for field in ("source", "reviewer", "note"):
                if not isinstance(pattern.get(field, ""), str):
                    raise ValueError(f"{field} for {label}/{pattern['name']} must be text.")
            if pattern["status"] == "verified" and not all(
                pattern.get(field, "").strip() for field in ("source", "reviewer", "note")
            ):
                raise ValueError(f"Verified pattern {label}/{pattern['name']} needs source, reviewer, and note.")
            features = pattern.get("features", {})
            if not isinstance(features, dict) or set(features) - feature_keys:
                raise ValueError(f"Unknown reference feature for {label}/{pattern['name']}.")
            for feature, cell in features.items():
                _validate_cell(feature, cell)
    return matrix


def load_reference_matrix(path=REFERENCE_PATH):
    path = Path(path)
    if not path.is_file():
        return default_reference_matrix()
    matrix = json.loads(path.read_text(encoding="utf-8"))
    return validate_reference_matrix(matrix)


def save_reference_matrix(matrix, path=REFERENCE_PATH):
    validate_reference_matrix(matrix)
    atomic_json(path, matrix)


def reference_cell(pattern, feature):
    return pattern.get("features", {}).get(feature, {
        "state": "unreviewed", "allowed": [], "note": "",
    })


def _cell_label(pattern, feature, labels):
    cells = [reference_cell(pattern, feature)]
    state = cells[0]["state"]
    if state == "unreviewed":
        return "?"
    if state == "not_relevant":
        return "n.v.t."
    return " / ".join(labels.get(category, category) for category in cells[0]["allowed"])


def matrix_table(matrix):
    validate_reference_matrix(matrix)
    category_labels = {
        spec["key"]: {item["id"]: item["label"] for item in spec["categories"]}
        for spec in FEATURE_SCHEMA
    }
    rows = []
    for label in matrix["labels"]:
        entry = matrix["letters"][label]
        for pattern in entry["patterns"]:
            row = {
                "letter": label,
                "pattern": pattern["name"],
                "reference status": pattern["status"],
            }
            row.update({
                spec["key"]: _cell_label(pattern, spec["key"], category_labels[spec["key"]])
                for spec in FEATURE_SCHEMA
            })
            rows.append(row)
    return rows


def quality_report(matrix):
    validate_reference_matrix(matrix)
    issues = []
    pair_checks = []
    for label in matrix["labels"]:
        patterns = matrix["letters"][label]["patterns"]
        for pattern in patterns:
            cells = [reference_cell(pattern, feature["key"]) for feature in FEATURE_SCHEMA]
            missing = sum(cell["state"] == "unreviewed" for cell in cells)
            specified = sum(cell["state"] == "specified" for cell in cells)
            not_relevant = sum(cell["state"] == "not_relevant" for cell in cells)
            if missing:
                issues.append({
                    "type": "missing_references",
                    "label": label,
                    "pattern": pattern["name"],
                    "count": missing,
                })
            if pattern["status"] == "concept":
                issues.append({
                    "type": "concept_pattern",
                    "label": label,
                    "pattern": pattern["name"],
                    "count": 1,
                })
            if any(len(cell.get("allowed", [])) > 1 for cell in cells):
                issues.append({
                    "type": "multiple_allowed_values",
                    "label": label,
                    "pattern": pattern["name"],
                    "count": sum(len(cell.get("allowed", [])) > 1 for cell in cells),
                })
            if specified == 0:
                issues.append({
                    "type": "no_specified_features",
                    "label": label,
                    "pattern": pattern["name"],
                    "count": not_relevant,
                })

    for index, left in enumerate(matrix["labels"]):
        for right in matrix["labels"][index + 1:]:
            pattern_pairs = []
            for left_pattern in matrix["letters"][left]["patterns"]:
                for right_pattern in matrix["letters"][right]["patterns"]:
                    distinguishing = set()
                    for feature in FEATURE_SCHEMA:
                        key = feature["key"]
                        left_cell = reference_cell(left_pattern, key)
                        right_cell = reference_cell(right_pattern, key)
                        if left_cell["state"] == right_cell["state"] == "specified" and not (
                            set(left_cell["allowed"]) & set(right_cell["allowed"])
                        ):
                            distinguishing.add(key)
                    pattern_pairs.append({
                        "left_pattern": left_pattern["name"],
                        "right_pattern": right_pattern["name"],
                        "distinguishing_features": sorted(distinguishing),
                        "distinguishable": bool(distinguishing),
                    })
            identical_patterns = []
            for left_pattern in matrix["letters"][left]["patterns"]:
                left_signature = _complete_signature(left_pattern)
                if left_signature is None:
                    continue
                for right_pattern in matrix["letters"][right]["patterns"]:
                    if left_signature == _complete_signature(right_pattern):
                        identical_patterns.append({
                            "left_pattern": left_pattern["name"],
                            "right_pattern": right_pattern["name"],
                        })
            distinguishing = sorted({
                key
                for pair in pattern_pairs
                for key in pair["distinguishing_features"]
            })
            pair_checks.append({
                "left": left,
                "right": right,
                "distinguishing_features": distinguishing,
                "pattern_pairs": pattern_pairs,
                "distinguishable": all(pair["distinguishable"] for pair in pattern_pairs),
                "identical_reference_patterns": identical_patterns,
            })
    unresolved_pairs = [pair for pair in pair_checks if not pair["distinguishable"]]
    return {
        "feature_count": len(FEATURE_SCHEMA),
        "letter_count": len(matrix["labels"]),
        "issues": issues,
        "letter_pairs": pair_checks,
        "insufficiently_distinguished_pairs": unresolved_pairs,
        "identical_reference_patterns": [
            {"left": pair["left"], "right": pair["right"],
             "patterns": pair["identical_reference_patterns"]}
            for pair in pair_checks if pair["identical_reference_patterns"]
        ],
        "complete_letters": [
            label for label in matrix["labels"]
            if all(
                all(reference_cell(pattern, spec["key"])["state"] != "unreviewed" for spec in FEATURE_SCHEMA)
                for pattern in matrix["letters"][label]["patterns"]
            )
        ],
    }


def _complete_signature(pattern):
    cells = [reference_cell(pattern, feature["key"]) for feature in FEATURE_SCHEMA]
    if any(cell["state"] == "unreviewed" for cell in cells):
        return None
    return tuple(
        (cell["state"], tuple(sorted(cell["allowed"])))
        for cell in cells
    )


def export_reference_bundle(matrix):
    validate_reference_matrix(matrix)
    return {
        "schema": schema_document(),
        "references": matrix,
    }


def export_matrix_csv(matrix):
    rows = matrix_table(matrix)
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_matrix_html(matrix):
    rows = matrix_table(matrix)
    headers = list(rows[0])
    thead = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row[header]))}</td>" for header in headers) + "</tr>"
        for row in rows
    )
    report = quality_report(matrix)
    missing_count = sum(issue["count"] for issue in report["issues"] if issue["type"] == "missing_references")
    return (
        "<!doctype html><html lang=\"nl\"><meta charset=\"utf-8\">"
        "<title>ASL-letterfeaturematrix</title><h1>ASL-letterfeaturematrix</h1>"
        f"<p>Featureschema {escape(FEATURE_SCHEMA_VERSION)}. "
        f"Oningevulde referentiecellen: {missing_count}. "
        "Conceptuele referenties zijn niet als gevalideerde regels aangemerkt.</p>"
        f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></html>"
    )


def clone_matrix(matrix):
    return deepcopy(matrix)
