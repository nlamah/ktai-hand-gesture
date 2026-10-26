"""Explicit editable rules; zero matches or several matches means abstention."""
import math

UNKNOWN = "__unknown__"


def default_rules():
    def extended(*names):
        return {f"geom_{name}_extended": {"min": .5} for name in names}
    def bent(*names):
        return {f"geom_{name}_bent": {"min": .5} for name in names}
    return {"version": 1, "rules": [
        {"label": "fist", "conditions": {**bent("index","middle","ring","pinky"),
            "geom_thumb_extended": {"max": .5},
            "geom_thumb_index_mcp_distance": {"max": .9}}},
        {"label": "open_hand", "conditions": extended("thumb","index","middle","ring","pinky")},
        {"label": "peace", "conditions": {**extended("index","middle"), **bent("ring","pinky"),
            "geom_tip_distance_index_middle": {"min": .35}}},
        {"label": "pointing", "conditions": {**extended("index"), **bent("middle","ring","pinky")}},
        {"label": "thumbs_up", "conditions": {**extended("thumb"), **bent("index","middle","ring","pinky")}},
        {"label": "ok", "conditions": {**extended("middle","ring","pinky"),
            "geom_tip_distance_thumb_index": {"max": .4}, "geom_index_bent": {"min": .5}}},
    ]}


def validate_rules(config, columns=None):
    if not isinstance(config, dict) or not isinstance(config.get("rules"), list):
        raise ValueError("Expected an object containing a 'rules' list.")
    for rule in config["rules"]:
        if not isinstance(rule, dict) or not isinstance(rule.get("label"), str) or not rule["label"].strip() or rule["label"] == UNKNOWN:
            raise ValueError("Each rule requires a valid label name.")
        if not isinstance(rule.get("conditions"), dict) or not rule["conditions"]:
            raise ValueError("Each rule requires at least one condition.")
        for name, bounds in rule["conditions"].items():
            if columns is not None and name not in columns:
                raise ValueError(f"Unknown feature: {name}")
            if not isinstance(bounds, dict) or not bounds or set(bounds) - {"min", "max"}:
                raise ValueError(f"Use min and/or max for {name}.")
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bounds.values()):
                raise ValueError(f"Bounds for {name} must be finite numbers.")
            if bounds.get("min", -math.inf) > bounds.get("max", math.inf):
                raise ValueError(f"Min exceeds max for {name}.")
    return config


def predict_rule(features, config):
    matches, explanations = [], []
    for rule in config["rules"]:
        checks = []
        for name, bounds in rule["conditions"].items():
            value = float(features[name])
            passed = math.isfinite(value) and bounds.get("min", -math.inf) <= value <= bounds.get("max", math.inf)
            checks.append({"feature": name, "value": value, "bounds": bounds, "passed": bool(passed)})
        matched = all(c["passed"] for c in checks)
        if matched:
            matches.append(rule["label"])
        explanations.append({"label": rule["label"], "matched": matched, "checks": checks})
    unique = sorted(set(matches))
    return (unique[0] if len(unique) == 1 else UNKNOWN), explanations
