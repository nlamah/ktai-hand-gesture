"""Optional index/middle fingertip weighting after all input normalization."""
from functools import partial
import math
from numbers import Real

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

TIP_DISTANCE_COLUMNS = ('shin_dist_08_12', 'geom_tip_distance_index_middle')


def validate_tip_weight(value, model='svm'):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('Index/middle tip weight must be a finite, nonnegative number.')
    if value != 1 and model != 'svm':
        raise ValueError('Nondefault index/middle tip weighting currently supports SVM only.')
    return float(value)


def weighting_metadata(columns, weight):
    return dict(index_middle_tip_weight=weight,
                tip_distance_columns=[column for column in columns if column in TIP_DISTANCE_COLUMNS],
                application='Multiply selected coordinates after preprocessing, before SVM fit/predict.')


def _output_columns(preprocessing, columns):
    if not isinstance(preprocessing, ColumnTransformer):
        return list(columns)
    # The existing selective scaler returns passthrough columns before scaled additions.
    ordered = [columns[i] for _, _, indices in preprocessing.transformers for i in indices]
    if len(ordered) != len(columns) or set(ordered) != set(columns):
        raise ValueError('Distance weighting requires preprocessing that preserves all input columns.')
    return ordered


def tip_weighting_step(preprocessing, columns, weight):
    if columns is None:
        raise ValueError('Column names are required for fingertip-distance weighting.')
    ordered = _output_columns(preprocessing, columns)
    selected = [i for i, name in enumerate(ordered) if name in TIP_DISTANCE_COLUMNS]
    if not selected:
        raise ValueError('This input has no explicit index/middle tip distance; choose a distance or finger-feature variant.')
    factors = np.ones(len(ordered), dtype=float)
    factors[selected] = weight
    # Persist numpy's multiply and its factors, so prediction needs no custom transform code.
    return FunctionTransformer(partial(np.multiply, factors), validate=True, feature_names_out='one-to-one')
