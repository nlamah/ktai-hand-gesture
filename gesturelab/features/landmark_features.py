"""Normalize MediaPipe source coordinates; no derived finger measurements."""
import numpy as np

def landmark_features(image_landmarks, world_landmarks, width=1, height=1):
    """Normalize source landmarks without calculating derived finger features."""
    raw = np.asarray(image_landmarks, dtype=float)
    world = np.asarray(world_landmarks, dtype=float)
    if raw.shape != (21, 3) or world.shape != (21, 3):
        raise ValueError("Expected 21 x 3 image landmarks and 21 x 3 world landmarks.")
    if not (np.isfinite(raw).all() and np.isfinite(world).all()):
        raise ValueError("Landmarks contain non-finite values.")
    if not (np.isfinite(width) and np.isfinite(height) and width>0 and height>0):
        raise ValueError("Image dimensions must be positive and finite.")
    image_metric=raw*np.array([width,height,width])
    image_scale=np.linalg.norm(image_metric[9]-image_metric[0])
    if image_scale<1e-8:
        raise ValueError("Hand size in the image is zero.")
    normalized_image=(image_metric-image_metric[0])/image_scale
    features = {f"raw_{i}_{axis}": float(raw[i, j]) for i in range(21) for j, axis in enumerate("xyz")}
    features.update({f"norm_{i}_{axis}": float(normalized_image[i, j]) for i in range(21) for j, axis in enumerate("xyz")})
    world_scale = np.linalg.norm(world[9] - world[0])
    if world_scale < 1e-8:
        raise ValueError("Measured hand size is zero.")
    normalized_world = (world - world[0]) / world_scale
    features.update({f"world_{i}_{axis}": float(normalized_world[i, j])
                     for i in range(21) for j, axis in enumerate("xyz")})
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Invalid normalized landmarks.")
    return features


def landmarks_for_sample(sample, thresholds=None):
    """126 normalized image/world coordinates, plus raw image coordinates."""
    return landmark_features(sample['image_landmarks'], sample['world_landmarks'],
                             sample['width'], sample['height'])

