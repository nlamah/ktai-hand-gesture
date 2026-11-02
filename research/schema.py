"""Versioned categorical ASL hand-feature definitions and fixed initial bounds."""

FEATURE_SCHEMA_VERSION = "asl-categorical-v4"
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
UNKNOWN = "unknown"

PARAMETER_DEFAULTS = {
    "flexion": {"extended_min_deg": 160.0, "bent_max_deg": 115.0},
    "finger_flexion": {"extended_min_deg": 160.0, "bent_max_deg": 115.0},
    "workbook_spread": {"minimum_gap_ratio": 0.25, "maximum_depth_ratio": 0.5, "minimum_joint_angle_deg": 120.0},
    "spread": {"minimum_spread_ratio": 0.375},
    "thumb_position": {"contact_max_ratio": 0.35, "near_max_ratio": 0.75},
    "workbook_thumb": {"b_min_deviation_deg": 20.0},
    "thumb_mcp_position": {"neutral_max_deviation_deg": 20.0},
    "workbook_crossing": {"minimum_swap_ratio": 0.02, "maximum_depth_ratio": 0.6, "minimum_joint_angle_deg": 100.0},
    "crossing": {"minimum_lateral_swap_ratio": 0.04},
}

_FLEXION_CATEGORIES = [
    {"id": "extended", "label": "Extended"},
    {"id": "partly_bent", "label": "Partly bent"},
    {"id": "bent", "label": "Bent"},
    {"id": UNKNOWN, "label": "Unknown"},
]
_SPREAD_CATEGORIES = [
    {"id": "not_spread", "label": "Not spread"},
    {"id": "spread", "label": "Spread"},
    {"id": UNKNOWN, "label": "Unknown"},
]
_THUMB_MCP_POSITION_CATEGORIES = [
    {"id": "toward_palm", "label": "Bent toward palm"},
    {"id": "neutral", "label": "Neutral"},
    {"id": "away_from_palm", "label": "Bent away from palm"},
    {"id": UNKNOWN, "label": "Unknown"},
]
_FINGER_DEFINITIONS = (
    ("thumb", "Thumb", (1, 2, 3, 4), (("mcp", "MCP joint", (1, 2, 3)),
                                     ("ip", "IP joint", (2, 3, 4)))),
    ("index", "Index finger", (5, 6, 7, 8), (("mcp", "MCP joint", (0, 5, 6)),
                                           ("pip", "PIP joint", (5, 6, 7)),
                                           ("dip", "DIP joint", (6, 7, 8)))),
    ("middle", "Middle finger", (9, 10, 11, 12), (("mcp", "MCP joint", (0, 9, 10)),
                                                  ("pip", "PIP joint", (9, 10, 11)),
                                                  ("dip", "DIP joint", (10, 11, 12)))),
    ("ring", "Ring finger", (13, 14, 15, 16), (("mcp", "MCP joint", (0, 13, 14)),
                                               ("pip", "PIP joint", (13, 14, 15)),
                                               ("dip", "DIP joint", (14, 15, 16)))),
    ("pinky", "Little finger", (17, 18, 19, 20), (("mcp", "MCP joint", (0, 17, 18)),
                                         ("pip", "PIP joint", (17, 18, 19)),
                                         ("dip", "DIP joint", (18, 19, 20)))),
)
_FINGER_POINTS = {name: points for name, _, points, _ in _FINGER_DEFINITIONS}
_FINGER_LABELS = {name: label.lower() for name, label, _, _ in _FINGER_DEFINITIONS}


def _feature(key, title, group, landmarks, method, parameter, limitations, categories):
    return {
        "key": key,
        "title": title,
        "group": group,
        "landmarks": list(landmarks),
        "measurement": method,
        "parameters": parameter,
        "validity": "Requires 21 finite MediaPipe landmarks and a nonzero palm scale.",
        "limitations": limitations,
        "categories": categories,
    }


FEATURE_SCHEMA = (
    tuple(
        _feature(
            (
                "thumb_mcp_position"
                if finger == "thumb" and joint_key == "mcp"
                else f"flexion_{finger}_{joint_key}"
            ),
            (
                f"{finger_label}: MCP position"
                if finger == "thumb" and joint_key == "mcp"
                else f"{finger_label}: {joint_label}"
            ),
            (
                "thumb_mcp_position"
                if finger == "thumb" and joint_key == "mcp"
                else "finger_flexion"
            ),
            (
                (*joint_points, 0, 5, 9, 17)
                if finger == "thumb" and joint_key == "mcp"
                else joint_points
            ),
            (
                "Signed MCP bend toward or away from the palm, relative to the local palm plane."
                if finger == "thumb" and joint_key == "mcp"
                else "Interior joint angle; wrist-to-MCP approximates the metacarpal direction for finger MCP joints."
            ),
            (
                "thumb_mcp_position"
                if finger == "thumb" and joint_key == "mcp"
                else "flexion"
            ),
            (
                "Direction depends on handedness classification and estimated 3D landmarks; an unknown hand side gives an unknown result."
                if finger == "thumb" and joint_key == "mcp"
                else "Each joint angle is classified separately; angles are not averaged across a finger. Finger MCP angles use the wrist as a geometric approximation."
            ),
            (
                _THUMB_MCP_POSITION_CATEGORIES
                if finger == "thumb" and joint_key == "mcp"
                else _FLEXION_CATEGORIES
            ),
        )
        for finger, finger_label, _, joints in _FINGER_DEFINITIONS
        for joint_key, joint_label, joint_points in joints
    )
    + tuple(
        _feature(
            f"spread_{left}_{right}",
            f"Spread: {_FINGER_LABELS[left]}–{_FINGER_LABELS[right]}",
            "finger_spread",
            (*_FINGER_POINTS[left], *_FINGER_POINTS[right]),
            "Euclidean distance between adjacent fingertips divided by wrist-to-middle-MCP distance.",
            "spread",
            "The ratio is scale invariant but depends on depth and estimated 3D landmarks.",
            _SPREAD_CATEGORIES,
        )
        for left, right in (("thumb", "index"), ("index", "middle"),
                            ("middle", "ring"), ("ring", "pinky"))
    )
    + (
        _feature(
            "crossing_index_middle",
            "Crossing: index finger and middle finger",
            "finger_crossing",
            (5, 8, 9, 12, 0, 17),
            "Checks whether fingertip lateral order reverses relative to MCP landmarks, with a minimum margin.",
            "crossing",
            "Geometric indication, not a visibility test; projection and occlusion can be misleading.",
            [
                {"id": "not_crossed", "label": "Not crossed"},
                {"id": "suspected_crossed", "label": "Suspected crossing"},
                {"id": UNKNOWN, "label": "Unknown"},
            ],
        ),
    )
    + tuple(
        _feature(
            f"thumb_tip_{finger}_proximity",
            f"Thumb tip near {_FINGER_LABELS[finger]} tip",
            "thumb_position",
            (4, _FINGER_POINTS[finger][-1], 0, 9),
            "Thumb-to-fingertip distance divided by wrist-to-middle-MCP distance.",
            "thumb_position",
            "Distance does not establish contact; perspective and landmark estimation affect the ratio. Use contact as an expected reference only when fingertips visibly touch.",
            [
                {"id": "contact", "label": "contact"},
                {"id": "near", "label": "Near"},
                {"id": "far", "label": "Far"},
                {"id": UNKNOWN, "label": "Unknown"},
            ],
        )
        for finger in ("index", "middle", "ring", "pinky")
    )
)


def schema_document():
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "labels": list(LETTERS),
        "parameters": PARAMETER_DEFAULTS,
        "features": list(FEATURE_SCHEMA),
        "limitations": [
            "MediaPipe world coordinates are estimated 3D landmarks, not direct measurements.",
            "J and Z involve movement; this matrix describes a single static pose.",
            "A returned landmark does not establish visibility of the point or body part.",
        ],
    }
