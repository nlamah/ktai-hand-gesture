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
    {"id": "extended", "label": "gestrekt"},
    {"id": "partly_bent", "label": "licht gebogen"},
    {"id": "bent", "label": "gebogen"},
    {"id": UNKNOWN, "label": "onbekend"},
]
_SPREAD_CATEGORIES = [
    {"id": "not_spread", "label": "niet gespreid"},
    {"id": "spread", "label": "gespreid"},
    {"id": UNKNOWN, "label": "onbekend"},
]
_THUMB_MCP_POSITION_CATEGORIES = [
    {"id": "toward_palm", "label": "gebogen naar handpalm"},
    {"id": "neutral", "label": "neutraal"},
    {"id": "away_from_palm", "label": "gebogen van hand weg"},
    {"id": UNKNOWN, "label": "onbekend"},
]
_FINGER_DEFINITIONS = (
    ("thumb", "Duim", (1, 2, 3, 4), (("mcp", "MCP-gewricht", (1, 2, 3)),
                                     ("ip", "IP-gewricht", (2, 3, 4)))),
    ("index", "Wijsvinger", (5, 6, 7, 8), (("mcp", "MCP-gewricht", (0, 5, 6)),
                                           ("pip", "PIP-gewricht", (5, 6, 7)),
                                           ("dip", "DIP-gewricht", (6, 7, 8)))),
    ("middle", "Middelvinger", (9, 10, 11, 12), (("mcp", "MCP-gewricht", (0, 9, 10)),
                                                  ("pip", "PIP-gewricht", (9, 10, 11)),
                                                  ("dip", "DIP-gewricht", (10, 11, 12)))),
    ("ring", "Ringvinger", (13, 14, 15, 16), (("mcp", "MCP-gewricht", (0, 13, 14)),
                                               ("pip", "PIP-gewricht", (13, 14, 15)),
                                               ("dip", "DIP-gewricht", (14, 15, 16)))),
    ("pinky", "Pink", (17, 18, 19, 20), (("mcp", "MCP-gewricht", (0, 17, 18)),
                                         ("pip", "PIP-gewricht", (17, 18, 19)),
                                         ("dip", "DIP-gewricht", (18, 19, 20)))),
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
        "validity": "Vereist 21 eindige MediaPipe-landmarks en een niet-nul palmschaal.",
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
                f"{finger_label}: MCP-positie"
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
                "Getekende buighoek van MCP naar handpalm of van de hand af, relatief aan het lokale palmvlak."
                if finger == "thumb" and joint_key == "mcp"
                else "Binnenhoek bij het betreffende gewricht; voor vinger-MCP wordt pols→MCP als benadering van de middenhandsrichting gebruikt."
            ),
            (
                "thumb_mcp_position"
                if finger == "thumb" and joint_key == "mcp"
                else "flexion"
            ),
            (
                "De richting is afhankelijk van de handedness-classificatie en de geschatte 3D-landmarks; zonder bekende linker- of rechterhand is de uitkomst onbekend."
                if finger == "thumb" and joint_key == "mcp"
                else "Elke gewrichtshoek wordt apart geclassificeerd; hoeken van een hele vinger worden niet gemiddeld. Vinger-MCP-hoeken zijn door de pols als referentie een geometrische benadering."
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
            f"Spreiding: {_FINGER_LABELS[left]}–{_FINGER_LABELS[right]}",
            "finger_spread",
            (*_FINGER_POINTS[left], *_FINGER_POINTS[right]),
            "Euclidische afstand tussen aangrenzende vingertoppen, gedeeld door de pols-tot-middelknokkelafstand.",
            "spread",
            "De verhouding is schaalinvariant, maar wordt beïnvloed door diepte en de geschatte 3D-landmarks.",
            _SPREAD_CATEGORIES,
        )
        for left, right in (("thumb", "index"), ("index", "middle"),
                            ("middle", "ring"), ("ring", "pinky"))
    )
    + (
        _feature(
            "crossing_index_middle",
            "Kruising: wijs- en middelvinger",
            "finger_crossing",
            (5, 8, 9, 12, 0, 17),
            "Controleert of de laterale volgorde van de twee vingertoppen omslaat t.o.v. de MCP-landmarks, met een minimale marge.",
            "crossing",
            "Geometrische aanwijzing, geen zichtbaarheidstest; projectie en occlusie kunnen misleiden.",
            [
                {"id": "not_crossed", "label": "niet gekruist"},
                {"id": "suspected_crossed", "label": "vermoedelijk gekruist"},
                {"id": UNKNOWN, "label": "onbekend"},
            ],
        ),
    )
    + tuple(
        _feature(
            f"thumb_tip_{finger}_proximity",
            f"Duimtop nabij {_FINGER_LABELS[finger]}top",
            "thumb_position",
            (4, _FINGER_POINTS[finger][-1], 0, 9),
            "Afstand tussen duimtop en vingertop, gedeeld door de pols-tot-middelknokkelafstand.",
            "thumb_position",
            "De afstandsmeting kan contact niet bewijzen; perspectief en landmarkschatting beïnvloeden de verhouding. Gebruik contact als verwachte referentiewaarde alleen wanneer de vingertoppen zichtbaar raken.",
            [
                {"id": "contact", "label": "contact"},
                {"id": "near", "label": "nabij"},
                {"id": "far", "label": "ver weg"},
                {"id": UNKNOWN, "label": "onbekend"},
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
            "MediaPipe-world-coördinaten zijn geschatte 3D-landmarks, geen directe metingen.",
            "J- en Z-gebaren bevatten beweging; de matrix beschrijft alleen één stilstaand posebeeld.",
            "Een landmarkuitvoer bewijst niet dat het punt of lichaamsdeel zichtbaar is.",
        ],
    }
