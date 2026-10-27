"""Create grouped folds or reuse verified assignments in a new code-frozen experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np

from gesturelab.datasets.data import fingerprint
from gesturelab.training.experiments import grouped_folds, groups_for


def _template_plan(source, project, development, selected, grouping, n_folds, seed):
    path = Path(source).resolve()
    path = path / 'split.json' if path.is_dir() else path
    plan = json.loads(path.read_text())
    if plan['dataset_fingerprint'] != fingerprint(project):
        raise ValueError('Split template requires exactly the same unchanged dataset.')
    actual_test = {s['id'] for s in selected if s['role'] == 'test'}
    if (set(plan['development_ids']) != set(development.id) or set(plan['test_ids']) != actual_test
            or set(plan['labels']) != set(development.label)):
        raise ValueError('Split template requires the same selected labels and photo IDs.')
    if (plan['grouping'], plan['n_folds'], plan['seed']) != (grouping, n_folds, seed):
        raise ValueError('Use the template grouping, fold count and seed with --split-from.')
    parent = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'note': 'Assignments only; new feature/training code is frozen separately. Original models are not reused.'}
    return plan, parent


def _template_folds(plan, development, groups):
    index = {sample_id: i for i, sample_id in enumerate(development.id)}
    all_ids, validation_ids, folds = set(index), [], []
    classes = set(development.label)
    for fold in plan['folds']:
        train, valid = fold['train_ids'], fold['validation_ids']
        if len(train) != len(set(train)) or len(valid) != len(set(valid)):
            raise ValueError('Split template contains duplicate photo IDs in a fold.')
        if not train or not valid or set(train) != all_ids - set(valid) or not set(valid) <= all_ids:
            raise ValueError('Split template has invalid training/validation membership.')
        tr, va = np.array([index[i] for i in train]), np.array([index[i] for i in valid])
        if set(groups[tr]) & set(groups[va]):
            raise ValueError('Split template leaks groups across training and validation.')
        if set(development.iloc[tr].label) != classes or set(development.iloc[va].label) != classes:
            raise ValueError('Split template must contain every label in each fold.')
        folds.append((tr, va))
        validation_ids.extend(valid)
    if len(folds) != plan['n_folds'] or len(validation_ids) != len(all_ids) or set(validation_ids) != all_ids:
        raise ValueError('Split template must validate every development photo exactly once.')
    return folds


def development_folds(project, development, selected, grouping, n_folds, seed, split_from=None):
    """Template reuse never loads an old model or disables its code-hash checks."""
    if split_from is None:
        folds, groups = grouped_folds(development, n_folds, grouping, seed)
        return folds, groups, None
    plan, parent = _template_plan(split_from, project, development, selected, grouping, n_folds, seed)
    groups = groups_for(development, grouping)
    return _template_folds(plan, development, groups), groups, parent
