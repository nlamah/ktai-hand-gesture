"""Horizontal reflection with anatomical landmark indices and labels preserved."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
from PIL import ImageOps

VERSION = 'horizontal-rgb-reflection-v1'


def mirror_image(image):
    """Reflect RGB pixels left/right, without rotating or changing letter labels."""
    return ImageOps.mirror(image.convert('RGB'))


def pixel_hash(image):
    return sha256(str(image.size).encode() + image.convert('RGB').tobytes()).hexdigest()


def reflected_detection(detection):
    """Exact geometric control, NOT a substitute for detecting mirrored RGB images.

    Landmark IDs retain their anatomical meaning: wrist stays 0, thumb stays 1-4.
    Image x becomes 1-x, world x changes sign, y/z stay fixed, handedness swaps.
    No detector-confidence or detector-equivariance claim follows from this map.
    """
    if detection.get('status') != 'ok':
        raise ValueError('Reflection requires an accepted detection.')
    if detection.get('handedness') not in ('Left', 'Right'):
        raise ValueError('Reflection requires Left or Right handedness.')
    result = deepcopy(detection)
    for key in ('image_landmarks', 'world_landmarks'):
        points = np.asarray(detection[key], dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError('Expected finite 21 x 3 landmarks.')
        points = points.copy()
        points[:, 0] = 1 - points[:, 0] if key == 'image_landmarks' else -points[:, 0]
        result[key] = points.tolist()
    result['handedness'] = 'Left' if detection['handedness'] == 'Right' else 'Right'
    result.pop('features', None)
    return result


def weighted_training_rows(originals, mirrors, labels, mirror_ok):
    """Original-only source population; each source retains total fitting weight 1."""
    originals, mirrors, labels = np.asarray(originals), np.asarray(mirrors), np.asarray(labels)
    mirror_ok = np.asarray(mirror_ok, dtype=bool)
    if originals.shape != mirrors.shape or originals.ndim != 2 or originals.shape[1] != 835:
        raise ValueError('Expected two N x 835 matrices.')
    if len(labels) != len(originals) or mirror_ok.shape != (len(labels),):
        raise ValueError('Source labels or mirror mask do not align.')
    if not np.isfinite(originals).all() or not np.isfinite(mirrors[mirror_ok]).all():
        raise ValueError('Accepted fitting rows must be finite.')
    rows = np.concatenate((originals, mirrors[mirror_ok]))
    targets = np.concatenate((labels, labels[mirror_ok]))
    weights = np.r_[np.where(mirror_ok, .5, 1.), np.full(mirror_ok.sum(), .5)]
    return rows, targets, weights
