"""Shin et al. (2021), Tables 2/3 and Eqs. 1–6, applied to image landmarks.

Table 2 is followed literally, including its unusual wrist exclusions.
Distance z-scores use one photo's 190 distances (population standard deviation).
Axis features are direction cosines as printed in Eqs. 4–6, not joint angles.
See docs/PAPER_SHIN2021.md for the paper's ambiguities and implementation choices.
"""
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

VERSION = 'shin2021-image-table2-cosine-row-zscore-v1'
ALL_PAIRS = tuple(combinations(range(21), 2))
# Paper indices are 1-based. These are exactly the 20 omissions in Table 2.
TABLE2_OMISSIONS = ((1, 2), (1, 5), (1, 9), (1, 13), (1, 17),
                    (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9),
                    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
                    (16, 17), (18, 19), (19, 20), (20, 21))
DISTANCE_PAIRS = tuple(pair for pair in ALL_PAIRS if (pair[0] + 1, pair[1] + 1) not in TABLE2_OMISSIONS)


def _landmarks(values: ArrayLike) -> NDArray[np.float64]:
    """Require finite 21×3 coordinates; never substitute world coordinates."""
    points = np.asarray(values, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError('Shin 2021 requires finite 21 x 3 image landmarks.')
    return points


def _vectors(points: NDArray[np.float64], pairs: tuple) -> NDArray[np.float64]:
    """Orient every vector from the lower-index point toward the higher one."""
    starts, ends = np.asarray(pairs).T
    vectors = points[ends] - points[starts]
    if not np.isfinite(vectors).all():
        raise ValueError('Shin 2021 landmark differences are not finite.')
    return vectors


def raw_distances(landmarks: ArrayLike) -> dict[str, float]:
    """Eq. 1: 190 Euclidean distances in the exact order of Table 2."""
    distances = np.linalg.norm(_vectors(_landmarks(landmarks), DISTANCE_PAIRS), axis=1)
    if not np.isfinite(distances).all():
        raise ValueError('Shin 2021 distances are not finite.')
    return {f'shin_dist_{i:02d}_{j:02d}': float(value)
            for (i, j), value in zip(DISTANCE_PAIRS, distances)}


def distance_features(landmarks: ArrayLike) -> dict[str, float]:
    """Eq. 2: z-score each hand's distance vector; no dataset statistics are fitted."""
    features = raw_distances(landmarks)
    values = np.asarray(list(features.values()))
    deviation = float(values.std(ddof=0))
    if not np.isfinite(deviation) or deviation <= 1e-12:
        raise ValueError('Shin 2021 distance vector has zero or invalid variance.')
    normalized = (values - values.mean()) / deviation
    return dict(zip(features, map(float, normalized)))


def angle_features(landmarks: ArrayLike) -> dict[str, float]:
    """Eqs. 3–6/Table 3: 210 directed pairs × positive x/y/z axis cosines."""
    vectors = _vectors(_landmarks(landmarks), ALL_PAIRS)
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        raise ValueError('Shin 2021 axis features are undefined for zero-length vectors.')
    cosines = np.clip(vectors / lengths[:, None], -1.0, 1.0)
    return {f'shin_angle_{i:02d}_{j:02d}_cos_{axis}': float(cosines[k, a])
            for k, (i, j) in enumerate(ALL_PAIRS) for a, axis in enumerate('xyz')}


def combined_features(landmarks: ArrayLike) -> dict[str, float]:
    """820 inputs: 190 normalized distances followed by 630 axis cosines."""
    return {**distance_features(landmarks), **angle_features(landmarks)}


def distance_for_sample(sample: Mapping[str, Any], thresholds: Any = None) -> dict[str, float]:
    """Use stored MediaPipe image-space x/y/z without aspect or palm correction."""
    return distance_features(sample['image_landmarks'])


def angle_for_sample(sample: Mapping[str, Any], thresholds: Any = None) -> dict[str, float]:
    """Use the image coordinate axes, preserving orientation information."""
    return angle_features(sample['image_landmarks'])


def combined_for_sample(sample: Mapping[str, Any], thresholds: Any = None) -> dict[str, float]:
    """Reconstruct the same paper inputs for training and webcam inference."""
    return combined_features(sample['image_landmarks'])
