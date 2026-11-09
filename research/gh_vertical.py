"""Turn G/H landmark geometry 180 degrees around the vertical camera axis."""
import numpy as np

from research.mirroring import weighted_training_rows

VERSION = 'gh-camera-y180-calibration-v1'


def rotate_vertical(rows):
    """Apply diag(-1, 1, -1) to directions, preserving intrinsic measurements.

    This is a proper rotation (determinant +1), unlike horizontal reflection.
    Translation cancels in pair directions; a wrist-centred landmark rotation
    produces exactly these descriptors. Detector z is estimated depth, so this
    remains a geometric approximation, not a rendered camera viewpoint.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 835 or not np.isfinite(rows).all():
        raise ValueError('Expected finite N x 835 inputs.')
    result = rows.copy()
    result[:, 190:820].reshape(-1, 210, 3)[:] *= [-1, 1, -1]
    return result


def adaptation_rows(originals, mirrors, labels, mirror_ok, calibration):
    """Existing mirror training plus G/H y180 copies and explicit calibration."""
    rows, targets, weights = weighted_training_rows(originals, mirrors, labels, mirror_ok)
    if calibration is not None:
        crows, cy, cw = weighted_training_rows(
            calibration['features'][:, 0], calibration['features'][:, 1],
            calibration['labels'], calibration['status'][:, 1] == 'ok')
        if set(cy) - {'G', 'H'}:
            raise ValueError('Calibration labels must be G or H.')
        rows = np.concatenate((rows, crows))
        targets = np.concatenate((targets, cy))
        weights = np.r_[weights, cw * 20.]
    gh = np.isin(targets, ['G', 'H'])
    weights[gh] /= 2
    return (np.concatenate((rows, rotate_vertical(rows[gh]))),
            np.concatenate((targets, targets[gh])), np.r_[weights, weights[gh]])
