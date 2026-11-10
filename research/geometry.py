"""Interpretable, label-independent ASL categories from MediaPipe landmarks."""
from math import degrees

import numpy as np

from research.schema import (
    FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, PARAMETER_DEFAULTS, UNKNOWN,
)

FINGER_JOINTS = {
    "thumb_mcp": (1, 2, 3),
    "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6),
    "index_pip": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10),
    "middle_pip": (9, 10, 11),
    "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14),
    "ring_pip": (13, 14, 15),
    "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18),
    "pinky_pip": (17, 18, 19),
    "pinky_dip": (18, 19, 20),
}
FINGER_ENDPOINTS = {
    "thumb": (1, 4),
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}
SPREAD_PAIRS = (("thumb", "index"), ("index", "middle"),
                ("middle", "ring"), ("ring", "pinky"))
_ALLOWED_PARAMETERS = {
    group: frozenset(values)
    for group, values in {
        "flexion": ("extended_min_deg", "bent_max_deg"),
        "finger_flexion": ("extended_min_deg", "bent_max_deg"),
        "workbook_crossing": ("minimum_swap_ratio", "maximum_depth_ratio", "minimum_joint_angle_deg"),
        "workbook_spread": ("minimum_gap_ratio", "maximum_depth_ratio", "minimum_joint_angle_deg"),
        "spread": ("minimum_spread_ratio",),
        "thumb_position": ("contact_max_ratio", "near_max_ratio"),
        "workbook_thumb": ("b_min_deviation_deg",),
        "thumb_mcp_position": ("neutral_max_deviation_deg",),
        "crossing": ("minimum_lateral_swap_ratio",),
    }.items()
}


def validate_parameters(parameters=None):
    result = {key: dict(value) for key, value in PARAMETER_DEFAULTS.items()}
    if parameters is not None:
        if not isinstance(parameters, dict) or set(parameters) - set(result):
            raise ValueError("Unknown categorical feature parameter group.")
        for group, values in parameters.items():
            if not isinstance(values, dict) or set(values) - _ALLOWED_PARAMETERS[group]:
                raise ValueError(f"Unknown parameter in group {group}.")
            result[group].update(values)
    for group, values in result.items():
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{group}.{key} must be a finite number.")
    if not 0 <= result["flexion"]["bent_max_deg"] < result["flexion"]["extended_min_deg"] <= 180:
        raise ValueError("Flexion thresholds must satisfy 0 ≤ bent < extended ≤ 180.")
    if not 0 <= result["finger_flexion"]["bent_max_deg"] < result["finger_flexion"]["extended_min_deg"] <= 180:
        raise ValueError("Finger flexion threshold must be in [0, 180] degrees.")
    if any(value < 0 for value in result["workbook_spread"].values()):
        raise ValueError("Workbook spread bounds must be nonnegative.")
    if result["workbook_spread"]["minimum_joint_angle_deg"] > 180:
        raise ValueError("Workbook spread joint angle must not exceed 180 degrees.")
    if (any(value < 0 for value in result["workbook_crossing"].values())
            or result["workbook_crossing"]["minimum_joint_angle_deg"] > 180):
        raise ValueError("Invalid workbook crossing bounds.")
    if result["spread"]["minimum_spread_ratio"] < 0:
        raise ValueError("spread.minimum_spread_ratio must be nonnegative.")
    if not 0 <= result["thumb_position"]["contact_max_ratio"] < result["thumb_position"]["near_max_ratio"]:
        raise ValueError("Thumb-position thresholds must satisfy 0 ≤ contact < near.")
    if not -180 <= result["workbook_thumb"]["b_min_deviation_deg"] <= 180:
        raise ValueError("Thumb B threshold must be in [-180, 180] degrees.")
    if not 0 <= result["thumb_mcp_position"]["neutral_max_deviation_deg"] < 90:
        raise ValueError("Thumb MCP neutral threshold must be in [0, 90) degrees.")
    if result["crossing"]["minimum_lateral_swap_ratio"] < 0:
        raise ValueError("crossing.minimum_lateral_swap_ratio must be nonnegative.")
    return result


def _points(values, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (21, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain 21 finite 3D landmarks.")
    return result


def _angle(a, b, c):
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return None
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(degrees(np.arccos(cosine)))


def _hand_basis(points):
    scale = float(np.linalg.norm(points[9] - points[0]))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Palm scale is zero or invalid.")
    up = (points[9] - points[0]) / scale
    lateral = points[5] - points[17]
    lateral = lateral - np.dot(lateral, up) * up
    lateral_length = float(np.linalg.norm(lateral))
    if not np.isfinite(lateral_length) or lateral_length <= 1e-8:
        lateral = None
    else:
        lateral /= lateral_length
    return scale, lateral, up


def _signed_thumb_mcp_deviation(points, handedness):
    if handedness not in ("Left", "Right"):
        return None

    longitudinal = points[9] - points[0]
    lateral = points[5] - points[17]
    palm_normal = np.cross(longitudinal, lateral)
    normal_length = float(np.linalg.norm(palm_normal))
    if normal_length <= 1e-8:
        return None
    palm_normal /= normal_length
    if handedness == "Right":
        palm_normal *= -1

    base = points[2] - points[1]
    distal = points[3] - points[2]
    if np.linalg.norm(base) <= 1e-8 or np.linalg.norm(distal) <= 1e-8:
        return None
    bend = np.arctan2(
        np.dot(np.cross(base, distal), palm_normal),
        np.dot(base, distal),
    )
    handedness_sign = 1 if handedness == "Right" else -1
    return float(degrees(bend) * handedness_sign)


def _crossing_profile(world, scale, lateral, up):
    """Compare distal finger polylines at equal palm-relative heights.

    Fingertips can uncross after a crossing at DIP level. Comparing at equal
    heights also avoids mistaking unequal finger lengths for lateral separation.
    """
    if lateral is None:
        return None, None
    normal = np.cross(lateral, up)
    projected = [np.stack([(world[i] - world[0]) @ lateral,
                          (world[i] - world[0]) @ up,
                          (world[i] - world[0]) @ normal]) / scale
                 for i in (6, 7, 8, 10, 11, 12)]
    order = np.sign(np.dot(world[5] - world[9], lateral))
    if not order:
        return None, None
    candidates = []
    for a, b in zip(projected[:2], projected[1:3]):
        for c, d in zip(projected[3:5], projected[4:6]):
            lower = max(min(a[1], b[1]), min(c[1], d[1]))
            upper = min(max(a[1], b[1]), max(c[1], d[1]))
            if upper - lower <= 1e-8 or abs(b[1] - a[1]) <= 1e-8 or abs(d[1] - c[1]) <= 1e-8:
                continue
            for height in (lower, upper):
                first = a + (b - a) * ((height - a[1]) / (b[1] - a[1]))
                second = c + (d - c) * ((height - c[1]) / (d[1] - c[1]))
                candidates.append((float(order * (first[0] - second[0])),
                                   float(abs(first[2] - second[2]))))
    return min(candidates) if candidates else (None, None)


def measure_hand_features(world_landmarks, handedness=None):
    """Return continuous measurements used by the categorical extractor and inspectors."""
    world = _points(world_landmarks, "World landmarks")
    scale, lateral, up = _hand_basis(world)
    measurements = {"palm_scale_world": scale}
    for name, triplet in FINGER_JOINTS.items():
        measurements[f"{name}_angle_deg"] = _angle(*(world[index] for index in triplet))
    measurements["thumb_mcp_signed_deviation_deg"] = _signed_thumb_mcp_deviation(
        world, handedness,
    )
    for left, right in SPREAD_PAIRS:
        a, b = FINGER_ENDPOINTS[left][1], FINGER_ENDPOINTS[right][1]
        measurements[f"spread_{left}_{right}_ratio"] = float(np.linalg.norm(world[a] - world[b]) / scale)
    for left, right in (("index", "middle"), ("middle", "ring")):
        left_base, left_tip = FINGER_ENDPOINTS[left]
        right_base, right_tip = FINGER_ENDPOINTS[right]
        prefix = f"spread_{left}_{right}"
        direction_left = world[left_tip] - world[left_base]
        direction_right = world[right_tip] - world[right_base]
        denominator = np.linalg.norm(direction_left) * np.linalg.norm(direction_right)
        measurements[prefix + "_direction_cosine"] = (
            float(np.dot(direction_left, direction_right) / denominator)
            if denominator > 1e-12 else None
        )
        if lateral is None:
            measurements[prefix + "_gap_ratio"] = None
            measurements[prefix + "_depth_ratio"] = None
        else:
            normal = np.cross(lateral, (world[9] - world[0]) / scale)
            deltas = [world[left_base + offset] - world[right_base + offset]
                      for offset in (1, 2, 3)]
            base_order = np.dot(world[left_base] - world[right_base], lateral)
            order_sign = np.sign(base_order)
            measurements[prefix + "_gap_ratio"] = float(min(
                order_sign * np.dot(delta, lateral) / scale for delta in deltas
            ))
            measurements[prefix + "_depth_ratio"] = float(max(
                abs(np.dot(delta, normal)) / scale for delta in deltas
            ))
    for finger in ("index", "middle", "ring", "pinky"):
        distance = np.linalg.norm(world[4] - world[FINGER_ENDPOINTS[finger][1]])
        measurements[f"thumb_tip_{finger}_distance_ratio"] = float(distance / scale)
    index_tip, middle_tip = world[8], world[12]
    index_base, middle_base = world[5], world[9]
    if lateral is None:
        measurements["crossing_base_lateral_ratio"] = None
        measurements["crossing_tip_lateral_ratio"] = None
    else:
        measurements["crossing_base_lateral_ratio"] = float(
            np.dot(index_base - middle_base, lateral) / scale)
        measurements["crossing_tip_lateral_ratio"] = float(
            np.dot(index_tip - middle_tip, lateral) / scale)

    gap, depth = _crossing_profile(world, scale, lateral, up)
    measurements["crossing_profile_gap_ratio"] = gap
    measurements["crossing_profile_depth_ratio"] = depth
    return measurements


def categorize_hand_measurements(values, parameters=None):
    """Map measured hand geometry to categories without recomputing landmarks."""
    params = validate_parameters(parameters)
    categories = {}
    flexion = params["flexion"]
    for joint in FINGER_JOINTS:
        if joint == "thumb_mcp":
            continue
        angle = values[f"{joint}_angle_deg"]
        if angle is None:
            state = UNKNOWN
        elif angle >= flexion["extended_min_deg"]:
            state = "extended"
        elif angle <= flexion["bent_max_deg"]:
            state = "bent"
        else:
            state = "partly_bent"
        categories[f"flexion_{joint}"] = state

    thumb_mcp = values["thumb_mcp_signed_deviation_deg"]
    neutral_threshold = params["thumb_mcp_position"]["neutral_max_deviation_deg"]
    if thumb_mcp is None:
        thumb_mcp_state = UNKNOWN
    elif abs(thumb_mcp) <= neutral_threshold:
        thumb_mcp_state = "neutral"
    elif thumb_mcp > 0:
        thumb_mcp_state = "toward_palm"
    else:
        thumb_mcp_state = "away_from_palm"
    categories["thumb_mcp_position"] = thumb_mcp_state

    spread_threshold = params["spread"]["minimum_spread_ratio"]
    for left, right in SPREAD_PAIRS:
        ratio = values[f"spread_{left}_{right}_ratio"]
        categories[f"spread_{left}_{right}"] = (
            "spread" if ratio >= spread_threshold else "not_spread"
        )

    margin = params["crossing"]["minimum_lateral_swap_ratio"]
    base_side = values["crossing_base_lateral_ratio"]
    tip_side = values["crossing_tip_lateral_ratio"]
    if (base_side is None or tip_side is None
            or abs(base_side) < margin or abs(tip_side) < margin):
        crossing = UNKNOWN
    else:
        crossing = "suspected_crossed" if base_side * tip_side < 0 else "not_crossed"
    categories["crossing_index_middle"] = crossing

    thumb = params["thumb_position"]
    for finger in ("index", "middle", "ring", "pinky"):
        ratio = values[f"thumb_tip_{finger}_distance_ratio"]
        if ratio <= thumb["contact_max_ratio"]:
            state = "contact"
        elif ratio <= thumb["near_max_ratio"]:
            state = "near"
        else:
            state = "far"
        categories[f"thumb_tip_{finger}_proximity"] = state

    expected_keys = {feature["key"] for feature in FEATURE_SCHEMA}
    if set(categories) != expected_keys:
        raise RuntimeError("Categorical feature schema does not match extractor outputs.")
    return categories


def extract_categorical_features(world_landmarks, parameters=None, handedness=None):
    """Return only categories; labels, letter references, and human notes are not inputs."""
    values = measure_hand_features(world_landmarks, handedness)
    return categorize_hand_measurements(values, parameters)


def hand_representation(world_landmarks, parameters=None, handedness=None):
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "categories": extract_categorical_features(world_landmarks, parameters, handedness),
        "measurements": measure_hand_features(world_landmarks, handedness),
    }
