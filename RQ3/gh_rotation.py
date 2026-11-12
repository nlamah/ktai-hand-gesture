"""Audit the 26-letter G/H rotation comparison without private photos.

The full diagnostic workflow also replays selected private captures. Those are optional
diagnostics, not part of this reproducible dataset/model audit.
"""
import json
import joblib
import numpy as np
import pandas as pd

from baseline.runner import ROOT
from research.data import atomic_json
from research.inputs import LABELS
from research.studies.gh_rotation_standalone import inputs, frozen_protocol, OUT, WORK
from research.studies.mirror_augmentation import metrics


def verify():
    protocol = json.loads((ROOT / OUT / 'protocol.json').read_text())
    if protocol != frozen_protocol():
        raise ValueError('The G/H experiment has different source code or inputs.')
    X, y, ids, folds = inputs()
    pred = pd.read_csv(OUT / 'predictions.csv', keep_default_na=False)
    saved = pd.read_csv(OUT / 'metrics.csv', keep_default_na=False)
    paired = pd.read_csv(OUT / 'paired_changes.csv', keep_default_na=False)
    np.testing.assert_array_equal(pred.id, ids)
    np.testing.assert_array_equal(pred.label, y)
    checked, replayed, pooled = 0, 0, []
    for fold in ['pooled', '1', '2', '3', '4', '5']:
        keep = np.ones(len(y), bool) if fold == 'pooled' else pred.fold.to_numpy() == int(fold)
        for arm in ['original', 'gh_y180']:
            score, _ = metrics(y[keep], pred[arm].to_numpy()[keep])
            row = saved[(saved.model == arm) & (saved.fold == fold)].iloc[0]
            for key, value in score.items():
                np.testing.assert_allclose(value, row[key], atol=1e-12, rtol=0)
            checked += 1
            if fold == 'pooled':
                pooled.append({'training': arm, **score})
        truth = y[keep]
        a, b = pred.original.to_numpy()[keep], pred.gh_y180.to_numpy()[keep]
        fixed = int(((a != truth) & (b == truth)).sum())
        lost = int(((a == truth) & (b != truth)).sum())
        row = paired[paired.fold == fold].iloc[0]
        np.testing.assert_array_equal(row[['fixed', 'lost', 'net']].to_numpy(int), [fixed, lost, fixed - lost])
    for fold, train, test in folds:
        np.testing.assert_array_equal(pred.fold.to_numpy()[test], np.full(len(test), fold))
        selection = np.array([test[np.flatnonzero(y[test] == letter)[0]] for letter in LABELS])
        original = joblib.load(WORK / f'original_fold{fold}.joblib')
        candidate = joblib.load(WORK / f'gh_y180_fold{fold}.joblib')
        assert original[-1]._gamma == candidate[-1]._gamma
        for arm, model in [('original', original), ('gh_y180', candidate)]:
            np.testing.assert_array_equal(model.predict(X[selection]), pred[arm].to_numpy()[selection])
            replayed += len(selection)
    result = {'status': 'passed', 'metric_rows': checked, 'paired_rows': 6,
              'held_out_model_predictions_replayed': replayed, 'pooled': pooled,
              'private_capture_replay': 'not part of this audit; requires separately supplied local captures',
              'training_data_changed': False}
    atomic_json(OUT / 'verification_core.json', result)
    return result


def main():
    import os
    os.chdir(ROOT)
    print(json.dumps(verify(), indent=2))


if __name__ == '__main__':
    main()
