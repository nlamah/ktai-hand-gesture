"""Load shared 820-input DS rows and the recorded source-group folds."""
import json
from pathlib import Path
import numpy as np


def load_shin_inputs(cache):
    cache = Path(cache)
    metadata = json.loads((cache / 'inputs_metadata.json').read_text())
    summary = json.loads((cache / 'summary.json').read_text())
    with np.load(cache / 'inputs.npz') as saved:
        X = saved['X'][:, :820].copy()
    samples = metadata['samples']
    ids = np.array([r['id'] for r in samples])
    y = np.array([r['label'] for r in samples])
    groups = np.array([r['group'] for r in samples])
    if X.shape != (len(samples), 820) or not np.isfinite(X).all() or len(set(ids)) != len(ids):
        raise ValueError('Invalid shared Shin cache.')
    folds = []
    assigned = np.zeros(len(y), int)
    for record in summary['folds']:
        train = np.flatnonzero(np.isin(groups, record['train_groups']))
        test = np.flatnonzero(np.isin(groups, record['test_groups']))
        if set(groups[train]) & set(groups[test]) or assigned[test].any():
            raise ValueError('Overlapping source groups across folds.')
        if len(train) + len(test) != len(y) or not len(train) or not len(test):
            raise ValueError('Incomplete train/test partition.')
        assigned[test] = record['fold']
        folds.append((record['fold'], train, test))
    if not assigned.all():
        raise ValueError('Each source must be tested exactly once.')
    return X, y, ids, groups, folds
