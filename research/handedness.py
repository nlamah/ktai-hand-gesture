"""Canonical detector-Right geometry, before feature extraction; no RGB redetection."""
from copy import deepcopy

import numpy as np

from research.mirroring import reflected_detection

VERSION = 'detector-right-landmarks-v1'


def right_hand_detection(detection):
    """Reflect detector-Left image/world points, preserving anatomical point IDs.

    Right/Left refer to the detector convention, not verified human handedness.
    Do not guess when handedness is absent. Never mutate the captured geometry.
    """
    if detection.get('status') != 'ok':
        raise ValueError('Canonicalization requires one accepted hand.')
    hand = detection.get('handedness')
    if hand == 'Left':
        return reflected_detection(detection)
    if hand == 'Right':
        return deepcopy(detection)
    raise ValueError('Canonicalization requires detector Left or Right.')


def right_hand_inputs(features, hands):
    """Exact cached-feature equivalent of right_hand_detection for 835 inputs.

    Distances and handedness-aware intrinsic angles are reflection invariant.
    Only the x cosine of each of the 210 image-space pair directions changes.
    This shortcut is checked against raw landmark extraction in the study.
    """
    values = np.asarray(features)
    hands = np.asarray(hands)
    if values.ndim != 2 or values.shape[1] != 835 or hands.shape != (len(values),):
        raise ValueError('Expected N x 835 inputs and N handedness labels.')
    if not np.isin(hands, ('Left', 'Right')).all() or not np.isfinite(values).all():
        raise ValueError('Accepted rows need finite geometry and known handedness.')
    result = values.copy()
    result[hands == 'Left', 190:820:3] *= -1
    return result


def fit_canonical(baseline, originals, labels):
    """One canonical row per source; preserve baseline preprocessing/gamma/priors."""
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.utils.class_weight import compute_class_weight

    preprocessing = deepcopy(baseline[0])
    learner = clone(baseline[-1])
    if hasattr(baseline[-1], '_gamma'):
        classes = np.unique(labels)
        prior = compute_class_weight('balanced', classes=classes, y=labels)
        learner.set_params(gamma=baseline[-1]._gamma,
                           class_weight=dict(zip(classes, prior)))
    learner.fit(preprocessing.transform(originals), labels)
    return make_pipeline(preprocessing, learner)
