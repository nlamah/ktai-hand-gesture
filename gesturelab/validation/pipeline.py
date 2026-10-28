"""Observe actual fit/transform rows in a tiny current-production-code probe."""
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gesturelab.training.fitting import _cross_validate
from gesturelab.features.training_inputs import TRAINING_INPUTS
from gesturelab.validation.features import reference_sample
from gesturelab.validation.report import write_json


def fit_transform_probe():
    rows = pd.DataFrame([dict(id=i, label=label, probe_value=value, condition='probe',
        participant=group, session=group) for i, label, value, group in (
        ('a', 'A', 1., 'p1'), ('b', 'B', 3., 'p1'), ('c', 'A', 101., 'p2'), ('d', 'B', 103., 'p2'))]).set_index('id', drop=False)
    plan = dict(model='svm', seed=42, labels=['A', 'B'], test_ids=['heldout_99999'], folds=[
        dict(train_ids=['a', 'b'], validation_ids=['c', 'd']),
        dict(train_ids=['c', 'd'], validation_ids=['a', 'b'])])
    fits, transforms = [], []
    original_fit, original_transform = StandardScaler.fit, StandardScaler.transform
    def fit(scaler, matrix, *args, **kwargs):
        fits.append(matrix[:, 0].tolist())
        return original_fit(scaler, matrix, *args, **kwargs)
    def transform(scaler, matrix, *args, **kwargs):
        result = original_transform(scaler, matrix, *args, **kwargs)
        transforms.append(dict(input=matrix[:, 0].tolist(), mean=scaler.mean_.tolist(), output=result[:, 0].tolist()))
        return result
    with patch.object(StandardScaler, 'fit', new=fit), patch.object(StandardScaler, 'transform', new=transform):
        _, predictions = _cross_validate(rows, ['probe_value'], plan, 'without_features')
    assert fits == [[1., 3.], [101., 103.]]
    assert [t['mean'] for t in transforms] == [[2.], [2.], [102.], [102.]]
    assert transforms[1]['output'] == [99., 101.] and transforms[3]['output'] == [-101., -99.]
    assert set(predictions.id) == {'a', 'b', 'c', 'd'}
    return dict(fit_inputs=fits, transformations=transforms, reserved_marker=99999,
                explanation='One unique numeric marker per constructed photo; this tests current data flow, not ASL performance.')


def metadata_probe():
    sample = reference_sample()
    altered = {**sample, 'id': 'SECRET_ID', 'label': 'SECRET_LABEL', 'filename': 'SECRET_FILE',
               'participant': 'SECRET_PERSON', 'session': 'SECRET_SESSION'}
    checked = {}
    for name, spec in TRAINING_INPUTS.items():
        first, second = spec.features(sample, None), spec.features(altered, None)
        assert first == second
        assert all(isinstance(value, (int, float)) and np.isfinite(value) for value in first.values())
        assert not {'id', 'label', 'filename', 'participant', 'session'} & first.keys()
        checked[name] = len(first)
    return checked


def audit_pipeline(report, out):
    for name, operation, expected in (
        ('fit_transform_rows', fit_transform_probe, 'Fit [1,3] and [101,103] separately; validation transformed with means 2 and 102.'),
        ('metadata_inputs', metadata_probe, 'Changing metadata changes no numeric model input.')):
        try:
            found = operation()
        except Exception as error:
            report.add('pipeline.' + name, 'Execute current production path with controlled inputs.', expected, str(error), False)
        else:
            write_json(out / (name + '.json'), found)
            report.add('pipeline.' + name, 'Execute current production path with controlled inputs.', expected, found, True,
                       files=['gesturelab/validation/pipeline.py', str(out / (name + '.json'))],
                       limitation='Current-code probe by Codex; not an independent human review or historical execution trace.')
    report.add('pipeline.selection_history', 'Inspect available presets and archived run artifacts.',
        'Parameter choices made without outer-fold or reserved-test outcomes.',
        'Current classifiers use fixed recorded profiles. Manual selection chronology is not recorded.',
        limitation='No inference about prior human/Codex choices from reproducibility alone.',
        next_step='Document when features/parameters were chosen and reserve independent data for final claims.')
