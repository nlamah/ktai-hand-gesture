"""Deterministic, label-independent hand geometry. No fitting on test data."""
from gesturelab.features.finger_features import compute_finger_features, finger_features_for_sample, FINGERS, unit, angle
from gesturelab.features.landmark_features import landmark_features, landmarks_for_sample
import numpy as np

FEATURE_VERSION = "geometry-v2.0"
FEATURE_SETS = {
    "raw": ("raw_",),
    "normalized": ("norm_",),
    "geometry": ("geom_",),
    "combined": ("norm_", "geom_"),
}




def extract_features(image_landmarks, world_landmarks, width=1, height=1, thresholds=None):
    """Recording/audit features; preserved for the capture and inspection tools."""
    landmarks = landmark_features(image_landmarks, world_landmarks, width, height)
    features = {key: value for key, value in landmarks.items() if not key.startswith('world_')}
    features.update(compute_finger_features(world_landmarks, thresholds))
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Invalid geometric features.")
    return features


def feature_columns(columns, feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return [c for c in columns if c.startswith(FEATURE_SETS[feature_set])]


def finger_table(features):
    rows = []
    for finger in FINGERS:
        prefix = f"geom_{finger}_"
        state = "extended" if features[prefix+"extended"] else "bent" if features[prefix+"bent"] else "intermediate"
        rows.append({"finger": {"thumb":"thumb","index":"index finger","middle":"middle finger","ring":"ring finger","pinky":"little finger"}[finger], "state": state,
                     "angle 1 (degrees)": round(features[prefix+"joint1_deg"], 1),
                     "angle 2 (degrees)": round(features[prefix+"joint2_deg"], 1),
                     "straightness (0–1)": round(features[prefix+"straightness"], 3)})
    return rows




def features_for_sample(sample, thresholds=None):
    """Combine source landmarks and derived finger features for augmented inputs."""
    landmarks = landmarks_for_sample(sample)
    # Preserve the existing column order for saved CSVs and model inputs.
    features = {key: value for key, value in landmarks.items() if not key.startswith('world_')}
    features.update(finger_features_for_sample(sample, thresholds))
    features.update({key: value for key, value in landmarks.items() if key.startswith('world_')})
    return features
