"""Group-aware calibration of categorical thresholds against an ASL workbook."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from gesturelab.datasets.data import atomic_json
from gesturelab.features.asl_excel_review import (
    categorize_workbook_features,
    load_landmark_cache,
    load_reference_workbook,
    measure_hand_features,
)
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS
from gesturelab.features.categorical_hand import validate_parameters


_GROUPING_VERSION = "roboflow-source-prefix-sha256-mod-5-v1"
_PARAMETER_GRIDS = {
    "workbook_thumb": [
        {"b_min_deviation_deg": value} for value in range(-90, 91, 2)
    ],
    "workbook_spread": [
        {"minimum_gap_ratio": round(gap, 3), "maximum_depth_ratio": round(depth, 3),
         "minimum_joint_angle_deg": angle}
        for gap in np.arange(0.05, 0.601, 0.025)
        for depth in (0.2, 0.3, 0.4, 0.5, 0.6)
        for angle in (110, 120, 130, 140, 150)
    ],
    "finger_flexion": [
        {"extended_min_deg": extended, "bent_max_deg": bent}
        for extended in range(140, 181, 5)
        for bent in range(70, min(151, extended), 5)
    ],
    "workbook_crossing": [
        {"minimum_swap_ratio": round(swap, 3), "maximum_depth_ratio": depth,
         "minimum_joint_angle_deg": angle}
        for swap in np.arange(0, 0.151, 0.025)
        for depth in (0.3, 0.5, 0.7)
        for angle in (80, 100, 120)
    ],
    "flexion": [
        {"extended_min_deg": extended, "bent_max_deg": bent}
        for extended in range(140, 181, 5)
        for bent in range(70, min(151, extended), 5)
    ],
    "spread": [
        {"minimum_spread_ratio": round(value, 3)}
        for value in np.arange(0.10, 0.901, 0.025)
    ],
    "thumb_position": [
        {"contact_max_ratio": round(contact, 3), "near_max_ratio": round(near, 3)}
        for contact in np.arange(0.05, 0.601, 0.025)
        for near in np.arange(0.10, 1.201, 0.025)
        if round(contact, 3) < round(near, 3)
    ],
    "thumb_mcp_position": [
        {"neutral_max_deviation_deg": value}
        for value in range(0, 61, 5)
    ],
    "crossing": [
        {"minimum_lateral_swap_ratio": round(value, 3)}
        for value in np.arange(0.0, 0.251, 0.005)
    ],
}


def _source_group(sample):
    return sample["filename"].split(".rf.", 1)[0]


def _grouped_split(samples):
    groups = sorted({_source_group(sample) for sample in samples})
    if len(groups) < 5:
        raise ValueError("At least five independent source groups are required.")
    holdout = {
        group for group in groups
        if int(sha256(f"{_GROUPING_VERSION}:{group}".encode()).hexdigest()[:8], 16) % 5 == 0
    }
    training = [sample for sample in samples if _source_group(sample) not in holdout]
    validation = [sample for sample in samples if _source_group(sample) in holdout]
    if not training or not validation:
        raise ValueError("The deterministic source-group split produced an empty partition.")
    train_labels = {sample["label"] for sample in training}
    validation_labels = {sample["label"] for sample in validation}
    if train_labels != validation_labels:
        raise ValueError("The grouped split must contain all labels in both partitions.")
    return training, validation, len(groups), len(holdout), len(groups) - len(holdout)


def _prepare_samples(samples, references):
    prepared = []
    for sample in samples:
        values = None
        if sample.get("status") == "ok" and sample.get("world_landmarks") is not None:
            values = measure_hand_features(
                sample["world_landmarks"], sample.get("handedness"),
            )
        prepared.append((sample, references[sample["label"]], values))
    return prepared


def _score(prepared, parameters):
    feature_matches = {}
    feature_totals = {}
    confusion = {}
    exact_images = 0
    evaluated_images = 0
    for sample, expected, measurements in prepared:
        if measurements is None:
            continue
        actual = categorize_workbook_features(measurements, parameters)
        image_matches = []
        for feature, cell in expected.items():
            allowed = cell["allowed"]
            if allowed is None:
                continue
            category = actual[feature]
            if feature.startswith("thumb_tip_") and category != "contact":
                category = "no_contact"
            matched = category in allowed
            feature_matches[feature] = feature_matches.get(feature, 0) + int(matched)
            feature_totals[feature] = feature_totals.get(feature, 0) + 1
            counts = confusion.setdefault(feature, {"tp": {}, "fp": {}, "fn": {}})
            if matched:
                counts["tp"][category] = counts["tp"].get(category, 0) + 1
            else:
                counts["fp"][category] = counts["fp"].get(category, 0) + 1
                for expected_category in allowed:
                    counts["fn"][expected_category] = (
                        counts["fn"].get(expected_category, 0) + 1
                    )
            image_matches.append(matched)
        if image_matches:
            evaluated_images += 1
            exact_images += int(all(image_matches))

    feature_accuracy = {
        feature: feature_matches.get(feature, 0) / count
        for feature, count in sorted(feature_totals.items())
    }
    feature_f1 = {}
    for feature, counts in confusion.items():
        labels = set(counts["tp"]) | set(counts["fp"]) | set(counts["fn"])
        class_f1 = []
        for label in labels:
            true_positive = counts["tp"].get(label, 0)
            false_positive = counts["fp"].get(label, 0)
            false_negative = counts["fn"].get(label, 0)
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            precision = true_positive / precision_denominator if precision_denominator else 0
            recall = true_positive / recall_denominator if recall_denominator else 0
            class_f1.append(
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )
        feature_f1[feature] = float(np.mean(class_f1)) if class_f1 else 0.0
    macro_accuracy = (
        float(np.mean(list(feature_accuracy.values()))) if feature_accuracy else 0.0
    )
    return {
        "detected_images": evaluated_images,
        "exact_match_images": exact_images,
        "exact_match_rate": exact_images / evaluated_images if evaluated_images else 0.0,
        "scored_feature_values": sum(feature_totals.values()),
        "feature_accuracy": feature_accuracy,
        "feature_macro_f1": feature_f1,
        "macro_feature_accuracy": macro_accuracy,
        "macro_feature_f1": (
            float(np.mean(list(feature_f1.values()))) if feature_f1 else 0.0
        ),
    }


def _objective(score):
    return (
        score["macro_feature_accuracy"],
        score["exact_match_rate"],
        score["macro_feature_f1"],
    )


def _selected_search_boundaries(parameters):
    boundaries = []
    for group, candidates in _PARAMETER_GRIDS.items():
        for key in candidates[0]:
            values = [candidate[key] for candidate in candidates]
            selected = parameters[group][key]
            minimum, maximum = min(values), max(values)
            if selected == minimum or selected == maximum:
                boundaries.append({
                    "parameter": f"{group}.{key}",
                    "selected": selected,
                    "minimum_searched": minimum,
                    "maximum_searched": maximum,
                })
    return boundaries


def _tune(training, initial_parameters, passes=3, groups=None):
    best = validate_parameters(initial_parameters)
    baseline = _score(training, best)
    for _ in range(passes):
        changed = False
        for group, candidates in _PARAMETER_GRIDS.items():
            if groups is not None and group not in groups:
                continue
            selected = best
            if group in ("finger_flexion", "workbook_crossing", "workbook_thumb"):
                selected_features = (
                    [key for key in training[0][1]
                     if key.startswith("flexion_") and key != "flexion_thumb_ip"]
                    if group == "finger_flexion" else
                    ["thumb_mcp_shape"] if group == "workbook_thumb" else ["crossing_index_middle"]
                )
                def objective(score):
                    return (
                        float(np.mean([score["feature_macro_f1"].get(key, 0)
                                       for key in selected_features])),
                        float(np.mean([score["feature_accuracy"].get(key, 0)
                                       for key in selected_features])),
                    )
            else:
                objective = _objective
            selected_score = _score(training, selected)
            for candidate in candidates:
                proposal = {
                    key: dict(value) for key, value in best.items()
                }
                proposal[group].update(candidate)
                proposal = validate_parameters(proposal)
                score = _score(training, proposal)
                if objective(score) > objective(selected_score):
                    selected, selected_score = proposal, score
            if selected[group] != best[group]:
                best = selected
                changed = True
        if not changed:
            break
    return best, baseline, _score(training, best)


def tune_landmark_cache(cache, references, workbook_sha256, passes=3,
                        initial_parameters=None, groups=None):
    """Tune on source groups and report once on a disjoint train-group holdout."""
    samples = cache.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Landmark cache does not contain samples.")
    (
        training_samples, validation_samples, group_count, holdout_count,
        training_group_count,
    ) = _grouped_split(samples)
    training = _prepare_samples(training_samples, references)
    validation = _prepare_samples(validation_samples, references)
    started = time.monotonic()
    initial_parameters = validate_parameters(initial_parameters)
    parameters, baseline_train, tuned_train = _tune(
        training, initial_parameters, passes=passes, groups=groups,
    )
    elapsed = time.monotonic() - started
    baseline_validation = _score(validation, initial_parameters)
    tuned_validation = _score(validation, parameters)
    all_train = training + validation
    return parameters, {
        "schema_version": 3,
        "thumb_mcp_categories": ["B-vormig", "not_B-vormig"],
        "slash_policy": "Only explicitly listed categories are accepted.",
        "finger_flexion_categories": ["extended", "partly_bent", "bent"],
        "initial_parameters": initial_parameters,
        "tuned_parameter_groups": list(groups) if groups is not None else list(_PARAMETER_GRIDS),
        "dataset_fingerprint": cache.get("dataset_fingerprint"),
        "workbook_sha256": workbook_sha256,
        "grouping": _GROUPING_VERSION,
        "source_groups": group_count,
        "training_source_groups": training_group_count,
        "holdout_source_groups": holdout_count,
        "source_groups_disjoint": True,
        "training_images": len(training_samples),
        "holdout_images": len(validation_samples),
        "detector_status_counts": {
            "training": _status_counts(training_samples),
            "holdout": _status_counts(validation_samples),
        },
        "selection_metric": (
            "For finger flexion and workbook crossing, maximize mean per-feature "
            "per-category macro-F1, breaking ties by feature accuracy. For other groups, "
            "maximize mean per-feature accuracy on training groups; break ties "
            "by exact-image match rate and per-feature, per-category macro-F1. Unknown "
            "predictions count as incorrect."
        ),
        "baseline": {
            "training": baseline_train,
            "holdout": baseline_validation,
            "all_train_descriptive": _score(all_train, initial_parameters),
        },
        "tuned": {
            "training": tuned_train,
            "holdout": tuned_validation,
            "all_train_descriptive": _score(all_train, parameters),
        },
        "parameters": parameters,
        "optimization": {
            "coordinate_descent_passes": passes,
            "elapsed_seconds": round(elapsed, 2),
            "selected_search_boundaries": _selected_search_boundaries(parameters),
        },
        "test_split_used": False,
    }


def _status_counts(samples):
    counts = {}
    for sample in samples:
        status = sample.get("status", "missing")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/external/asl_letters_v1/train"),
        help="COCO train split used to validate the landmark cache.",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/reports/asl_letters_v1_train_landmarks.json"),
    )
    parser.add_argument(
        "--reference", type=Path,
        default=Path(__file__).resolve().parents[2] / "assets/asl_feature_references.xlsx",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/asl_feature_threshold_tuning_01"),
    )
    parser.add_argument("--initial-thresholds", type=Path)
    parser.add_argument("--fingers-only", action="store_true")
    parser.add_argument("--spread-only", action="store_true")
    parser.add_argument("--fingers-and-crossing", action="store_true")
    parser.add_argument("--workbook-all", action="store_true")
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args(argv)
    if sum((args.fingers_only, args.spread_only, args.fingers_and_crossing, args.workbook_all)) > 1:
        parser.error("Choose only one tuning scope.")
    if args.passes < 1:
        parser.error("--passes must be at least one.")
    thresholds_path = args.output_dir / "categorical_thresholds.json"
    report_path = args.output_dir / "calibration_report.json"
    if thresholds_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Tuning output already exists in {args.output_dir}; choose a new --output-dir."
        )
    workbook_contents = args.reference.read_bytes()
    cache = load_landmark_cache(args.cache, args.dataset)
    if cache is None:
        raise ValueError("Landmark cache is stale or does not match the selected dataset.")
    references = load_reference_workbook(workbook_contents)
    parameters, report = tune_landmark_cache(
        cache, references, sha256(workbook_contents).hexdigest(), args.passes,
        initial_parameters=(json.loads(args.initial_thresholds.read_text())
                            if args.initial_thresholds else None),
        groups=(["workbook_thumb", "flexion", "finger_flexion", "workbook_spread",
                 "thumb_position", "workbook_crossing"] if args.workbook_all else
                ["finger_flexion", "workbook_crossing"] if args.fingers_and_crossing else
                ["workbook_spread"] if args.spread_only else
                ["finger_flexion"] if args.fingers_only else None),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(thresholds_path, parameters)
    atomic_json(report_path, report)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "baseline_holdout_macro_accuracy":
            report["baseline"]["holdout"]["macro_feature_accuracy"],
        "tuned_holdout_macro_accuracy":
            report["tuned"]["holdout"]["macro_feature_accuracy"],
        "baseline_holdout_macro_f1":
            report["baseline"]["holdout"]["macro_feature_f1"],
        "tuned_holdout_macro_f1":
            report["tuned"]["holdout"]["macro_feature_f1"],
        "baseline_holdout_exact_matches":
            report["baseline"]["holdout"]["exact_match_images"],
        "tuned_holdout_exact_matches":
            report["tuned"]["holdout"]["exact_match_images"],
        "thresholds": parameters,
    }, indent=2))


if __name__ == "__main__":
    main()
