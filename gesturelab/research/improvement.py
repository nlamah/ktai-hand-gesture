"""One explicit data-augmentation refit; no test selection, tuning or model sweep."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import joblib
from sklearn.base import clone

from gesturelab.datasets.data import read_manifest, atomic_json, fingerprint, ensure_project
from gesturelab.features.training_inputs import input_table
from gesturelab.research.models import ModelHandle, register_model
from gesturelab.training.experiment_plan import load_experiment, _save_experiment
from gesturelab.validation.provenance import environment_record
from gesturelab.validation.report import sha256, write_json


def select_development(project, ids=None, condition=None):
    records = read_manifest(project)['samples']
    selected = [s for s in records if ids is None or s['id'] in ids]
    if ids is not None and set(ids) != {s['id'] for s in selected}:
        raise ValueError('Some explicitly selected photo IDs are missing.')
    if ids is None:
        selected = [s for s in selected if s['role'] == 'development'
                    and s.get('provenance', {}).get('protocol_role') != 'test']
    if any(s['role'] != 'development' or s.get('provenance', {}).get('protocol_role') == 'test' for s in selected):
        raise ValueError('Reserved test attempts cannot be selected for an improvement.')
    if condition:
        selected = [s for s in selected if s['condition'] == condition]
    usable = [s for s in selected if s.get('included', True) and s['status'] == 'ok']
    if not usable:
        raise ValueError('No usable new development attempts selected; collect development photos first.')
    return usable, dict(offered=len(selected), usable=len(usable), excluded=[s['id'] for s in selected if s not in usable])


def _copy_selection(sources, target):
    """Keep IDs and copy bytes; exact duplicate pixels are logged, never counted twice."""
    target = ensure_project(target)
    records, seen_ids, pixels, duplicates = [], set(), {}, []
    for source, samples in sources:
        for sample in samples:
            identity, pixel = sample['id'], sample.get('pixel_sha256')
            if identity in seen_ids:
                raise ValueError('Repeated photo ID across source datasets.')
            seen_ids.add(identity)
            if pixel and pixel in pixels:
                if pixels[pixel]['label'] != sample['label']:
                    raise ValueError('Identical pixels have conflicting intended labels.')
                duplicates.append(dict(id=identity, retained_id=pixels[pixel]['id']))
                continue
            record = deepcopy(sample)
            for key, folder in (('original_path', 'originals'), ('image_path', 'images')):
                path = Path(source) / sample[key]
                record[key] = f'{folder}/{identity}{path.suffix}'
                shutil.copy2(path, target / record[key])
            record['role'] = 'development'
            records.append(record)
            if pixel: pixels[pixel] = record
    atomic_json(target / 'manifest.json', dict(schema=2, samples=records))
    return records, duplicates


def _counts(samples):
    return {key: dict(Counter(str(s.get(key, 'unknown')) for s in samples))
            for key in ('label', 'participant', 'condition', 'session')}


def _augmentation_plan(original, target, samples, hypothesis, change, expected):
    plan = deepcopy(original)
    plan.update(project=str(Path(target).resolve()), dataset_fingerprint=fingerprint(target),
        created_at=datetime.now(timezone.utc).isoformat(), development_ids=[s['id'] for s in samples],
        test_ids=[], excluded_development_ids=[], folds=[], n_folds=0, split_parent=None,
        hypothesis=hypothesis, change=change, expected_outcome=expected, synthetic=any(
            s.get('provenance', {}).get('type') == 'synthetic' for s in samples),
        evaluation_note='One final development refit, no CV or test evaluation. Both data amount and composition change.')
    return plan


def _fit_improvement(handle, plan, records, experiment):
    columns = handle.bundle['config']['columns']
    table = input_table(records, plan['feature_config'], handle.metadata['variant']).set_index('id')
    matrix = table[columns].to_numpy()
    estimator = clone(handle.bundle['estimator'])
    started = time.perf_counter()
    estimator.fit(matrix, table.label.to_numpy())
    if fingerprint(plan['project']) != plan['dataset_fingerprint']:
        raise ValueError('The new training dataset changed during fitting; no model is saved.')
    bundle = deepcopy(handle.bundle)
    bundle.update(estimator=estimator, split_sha256=sha256(experiment / 'split.json'), synthetic=plan['synthetic'])
    folder = experiment / handle.metadata['variant']
    folder.mkdir()
    joblib.dump(bundle, folder / 'model.joblib')
    table[columns].to_csv(folder / 'training_features.csv')
    write_json(folder / 'training.json', dict(variant=handle.metadata['variant'], n_train=len(table),
        n_features=len(columns), feature_columns=columns, model_sha256=sha256(folder / 'model.joblib'),
        fit_seconds=time.perf_counter() - started, synthetic=plan['synthetic'],
        note='Cloned actual baseline estimator with identical parameters; all learned preprocessing fitted afresh on development rows only. No CV/test scores.'))
    return folder / 'model.joblib'


def train_improvement(baseline, additions, dataset_out, experiment_out, hypothesis, change, expected,
                      ids=None, all_development=False, condition=None, match_count=None):
    if not ids and not all_development:
        raise ValueError('Explicitly select IDs or --all-development; no automatic training selection.')
    if not all(str(value).strip() for value in (hypothesis, change, expected)):
        raise ValueError('Record a hypothesis, concrete change and expected outcome before fitting.')
    handle = ModelHandle.load(baseline)
    if not handle.metadata['baseline_svm_shin']:
        raise ValueError('This route requires the actual SVM/Shin 820-value baseline.')
    _, original_project, original, records = load_experiment(handle.metadata['experiment'])
    selected, selection = select_development(additions, ids, condition)
    if match_count is not None and len(selected) != match_count:
        raise ValueError('The explicitly requested frontal control must match the requested usable count exactly.')
    if any(s.get('alphabet', handle.metadata['alphabet']) != handle.metadata['alphabet'] or s['label'] not in handle.metadata['labels'] for s in selected):
        raise ValueError('New development alphabet/labels differ from the baseline.')
    sources = [(original_project, [records[i] for i in original['development_ids']]), (Path(additions), selected)]
    return _save_improvement(handle, original, sources, dataset_out, experiment_out, selection, hypothesis, change, expected)


def _save_improvement(handle, original, sources, dataset_out, experiment_out, selection, hypothesis, change, expected):
    dataset_out, experiment_out = Path(dataset_out).resolve(), Path(experiment_out).resolve()
    protected = [Path(s).resolve() for s, _ in sources] + [Path(handle.metadata['experiment']).resolve()]
    if any(path.exists() or any(path.is_relative_to(root) or root.is_relative_to(path) for root in protected)
           for path in (dataset_out, experiment_out)) or dataset_out.is_relative_to(experiment_out) or experiment_out.is_relative_to(dataset_out):
        raise ValueError('Choose separate new dataset/experiment directories outside source data and results.')
    samples, duplicates = _copy_selection(sources, dataset_out)
    plan = _augmentation_plan(original, dataset_out, samples, hypothesis, change, expected)
    _save_experiment(experiment_out, plan, plan['feature_config'])
    environment_record(experiment_out / 'provenance')
    write_json(experiment_out / 'selection.json', dict(baseline=handle.metadata, new_selection=selection,
        source_datasets=[str(p) for p, _ in sources], source_manifest_hashes=[sha256(Path(p) / 'manifest.json') for p, _ in sources],
        original_ids=[s['id'] for s in sources[0][1]], added_ids=[s['id'] for s in sources[1][1]],
        duplicate_pixels_not_recounted=duplicates, counts=_counts(samples), added_counts=_counts(sources[1][1]),
        hypothesis=hypothesis, change=change, expected_outcome=expected,
        limitation='Data amount and composition both change. A matched frontal control is an optional separate explicit run.'))
    model = _fit_improvement(handle, plan, samples, experiment_out)
    register_model(model, experiment_out / 'registered_model.json', experiment_out,
                   handle.metadata['alphabet'], 'Viewpoint development improvement', True)
    return experiment_out
