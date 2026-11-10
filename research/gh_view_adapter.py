"""G/H view adaptation with a direction-preserving roll and a restricted routing gate."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from research.mirroring import weighted_training_rows
from research.shin import ALL_PAIRS

VERSION = 'gh-index-axis-roll-calibration-v1'
INDEX_DIRECTION = ALL_PAIRS.index((5, 8))


def roll_about_index(rows):
    """Proper 180-degree rotation around MCP-to-tip direction, not pixel rendering.

    This preserves the index pointing direction. Distances and intrinsic world
    angles remain fixed. It cannot synthesize occlusion or correct detector error.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 835 or not np.isfinite(rows).all():
        raise ValueError('Expected finite N x 835 inputs.')
    result = rows.copy()
    directions = rows[:, 190:820].reshape(-1, 210, 3)
    axis = directions[:, INDEX_DIRECTION]
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    if (norm < 1e-8).any():
        raise ValueError('Index direction is undefined.')
    axis = axis / norm
    dot = np.einsum('nij,nj->ni', directions, axis)
    result[:, 190:820] = (2*dot[:, :, None]*axis[:, None, :] - directions).reshape(-1, 630)
    return result


def horizontal_mask(rows, minimum=.8, minimum_projection=.15):
    """Require a visible, mainly horizontal projected index, regardless of side."""
    directions = np.asarray(rows)[:, 190:820].reshape(-1, 210, 3)[:, INDEX_DIRECTION]
    projection = np.linalg.norm(directions[:, :2], axis=1)
    ratio = np.divide(np.abs(directions[:, 0]), projection, out=np.zeros(len(rows)), where=projection > 1e-12)
    return (projection >= minimum_projection) & (ratio >= minimum)


def adaptation_rows(originals, mirrors, labels, mirror_ok, calibration, calibration_weight=20.):
    """Maintain source weight; roll only G/H; explicitly add selected calibration data."""
    rows, targets, weights = weighted_training_rows(originals, mirrors, labels, mirror_ok)
    calibrated, ctargets, cweights = weighted_training_rows(
        calibration['features'][:, 0], calibration['features'][:, 1],
        calibration['labels'], calibration['status'][:, 1] == 'ok')
    if set(ctargets) - {'G', 'H'} or calibration_weight <= 0:
        raise ValueError('Calibration must contain only G/H and have positive weight.')
    rows = np.concatenate((rows, calibrated))
    targets = np.concatenate((targets, ctargets))
    weights = np.r_[weights, cweights*calibration_weight]
    gh = np.isin(targets, ['G', 'H'])
    weights[gh] /= 2
    return (np.concatenate((rows, roll_about_index(rows[gh]))),
            np.concatenate((targets, targets[gh])), np.r_[weights, weights[gh]])


class GHViewGuard(ClassifierMixin, BaseEstimator):
    """Retain baseline predictions unless a horizontal pose receives adapted G/H.

    Both SVMs receive identical preprocessed inputs. This is a 26-class learned
    alternative, not a binary G/H model forced onto every hand.
    """
    def __init__(self, baseline, adapted, minimum_horizontal=.8, minimum_projection=.15):
        self.baseline = baseline
        self.adapted = adapted
        self.minimum_horizontal = minimum_horizontal
        self.minimum_projection = minimum_projection
        self.classes_ = baseline.classes_
        self.n_features_in_ = baseline.n_features_in_

    def fit(self, X, y=None):
        """Validate the two already fitted SVMs; this router does not refit them."""
        check_is_fitted(self.baseline)
        check_is_fitted(self.adapted)
        if not np.array_equal(self.baseline.classes_, self.adapted.classes_):
            raise ValueError('The two SVMs must have identical ordered classes.')
        if np.asarray(X).shape[1] != self.n_features_in_:
            raise ValueError('The routing input width does not match its fitted SVMs.')
        return self

    def adapt_predictions(self, X, baseline_predictions):
        result = np.asarray(baseline_predictions).copy()
        indices = np.flatnonzero(horizontal_mask(X, self.minimum_horizontal, self.minimum_projection))
        if len(indices):
            proposed = self.adapted.predict(np.asarray(X)[indices])
            allowed = np.isin(proposed, ['G', 'H'])
            result[indices[allowed]] = proposed[allowed]
        return result

    def predict(self, X):
        return self.adapt_predictions(X, self.baseline.predict(X))

    def decision_function(self, X):
        values = self.baseline.decision_function(X).copy()
        indices = np.flatnonzero(horizontal_mask(X, self.minimum_horizontal, self.minimum_projection))
        if len(indices):
            proposed = self.adapted.predict(np.asarray(X)[indices])
            selected = indices[np.isin(proposed, ['G', 'H'])]
            if len(selected):
                values[selected] = self.adapted.decision_function(np.asarray(X)[selected])
        return values
