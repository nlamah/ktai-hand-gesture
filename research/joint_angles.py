"""Joint angles derived from the exact normalized x/y/z used by Shin."""
import numpy as np

from research.geometry import FINGER_JOINTS, _signed_thumb_mcp_deviation
from research.inputs import ANGLE_COLUMNS

VERSION = 'shin-coordinate-joint-angles-v1'


def joint_angles_batch(landmarks, handedness):
    """14 interior angles plus signed thumb deviation, without axis rescaling.

    These normalized image coordinates are deliberately identical to Shin's
    inputs, not aspect-correct pixels or estimated world coordinates. Angles
    are geometric descriptors in that coordinate system, not clinical angles.
    Handedness is used only by the historical signed-thumb convention.
    """
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite N x 21 x 3 landmarks.')
    if len(handedness) != len(points):
        raise ValueError('One handedness value is required per image.')
    indices = np.array(list(FINGER_JOINTS.values()))
    u = points[:, indices[:, 0]] - points[:, indices[:, 1]]
    v = points[:, indices[:, 2]] - points[:, indices[:, 1]]
    lengths = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=2)
    cosine = np.full(lengths.shape, np.nan)
    np.divide(np.sum(u * v, axis=2), lengths, out=cosine, where=lengths > 1e-12)
    interior = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    signed = [_signed_thumb_mcp_deviation(p, h) for p, h in zip(points, handedness)]
    result = np.c_[interior, [np.nan if x is None else x for x in signed]].astype(np.float32)
    assert result.shape == (len(points), len(ANGLE_COLUMNS))
    return result
