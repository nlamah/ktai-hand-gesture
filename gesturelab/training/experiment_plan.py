"""Shared experiment plans, dataset validation and feature selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import sklearn

from gesturelab.datasets.data import read_manifest, atomic_json, fingerprint
from gesturelab.features.training_inputs import DEFAULT_VARIANTS, input_source_files
from gesturelab.features.audit import load_thresholds
from gesturelab.training.experiments import estimator, grouped_folds, groups_for
from gesturelab.training.classifiers import classifier_versions
from gesturelab.training.splits import development_folds
from gesturelab.training.distance_weighting import validate_tip_weight
from gesturelab.training.mirror_augmentation import DEFINITION as MIRROR_DEFINITION

VARIANTS = DEFAULT_VARIANTS
CODE_FILES = (
    'features/inspection.py', 'training/experiment_plan.py', 'training/fitting.py', 'training/comparison.py', 'training/experiments.py',
    'training/classifiers.py',
    'training/neural_network.py',
    'training/distance_weighting.py',
    'training/mirror_augmentation.py',
    'training/predictions.py',
    'training/splits.py', 'training/paper_study.py',
    'training/reporting.py',
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_files():
    return sorted(set(CODE_FILES) | input_source_files())


def code_hashes():
    return {name:file_hash(Path(__file__).resolve().parents[1]/name) for name in code_files()}


def _selected_development_samples(project, labels):
    included = [sample for sample in read_manifest(project)['samples'] if sample['included']]
    labels = sorted(set(labels)) if labels is not None else sorted({s['label'] for s in included if s['role'] == 'development'})
    if len(labels) < 2:
        raise ValueError('Choose at least two letters with development photos.')
    selected = [sample for sample in included if sample['label'] in labels]
    if any(sample['status'] == 'pending' for sample in selected):
        raise ValueError('First analyze all selected photos with scripts/analyze_dataset.py.')
    data = pd.DataFrame([sample for sample in selected if sample['status'] == 'ok'])
    if data.empty:
        raise ValueError('No usable hand detections found.')
    development = data[data.role == 'development'].reset_index(drop=True)
    if set(labels) != set(development.label):
        raise ValueError('Not every selected label has usable development photos.')
    return selected, labels, development


def _validate_source_video_groups(selected, grouping):
    source_groups = {}
    for sample in selected:
        if sample.get('provenance', {}).get('type') == 'video_frame':
            group = sample['participant'] if grouping == 'participant' else (sample['participant'], sample['session'])
            source_groups.setdefault(sample['provenance']['source_sha256'], set()).add((str(group), sample['role']))
    if any(len(groups) > 1 for groups in source_groups.values()):
        raise ValueError('Frames from one video must stay in one group and one role.')


def _validate_test_samples(selected, labels, grouping, allow_no_test, allow_incomplete_test=False):
    development = [sample for sample in selected if sample['role'] == 'development']
    test = [sample for sample in selected if sample['role'] == 'test']
    if not test and not allow_no_test:
        raise ValueError('Reserve a new session for testing. To practice without a final test, explicitly use --allow-no-test.')
    if test and set(labels) != {sample['label'] for sample in test} and not allow_incomplete_test:
        raise ValueError('The separate test set must contain every selected label. To explicitly allow missing test letters, use --allow-incomplete-test.')
    if test and set(groups_for(pd.DataFrame(development), grouping)) & set(groups_for(pd.DataFrame(test), grouping)):
        raise ValueError('A group occurs in both development and test. Keep sessions/participants separate.')
    _validate_source_video_groups(selected, grouping)
    return development, test


def _build_experiment_plan(project, development, selected, labels, split, groups, config, parameters):
    dev_all = [sample for sample in selected if sample['role'] == 'development']
    test_all = [sample for sample in selected if sample['role'] == 'test']
    plan={'schema':1,'project':str(project),'created_at':datetime.now(timezone.utc).isoformat(),
          'dataset_fingerprint':fingerprint(project),'feature_config':config,'code_hashes':code_hashes(),
          'labels':labels,**parameters,
          'sklearn_version':sklearn.__version__,'python_version':platform.python_version(),
          'numpy_version':np.__version__,'pandas_version':pd.__version__,
          'development_ids':development.id.tolist(),'test_ids':[s['id'] for s in test_all],
          'excluded_development_ids':[s['id'] for s in dev_all if s['status']!='ok'],
          'missing_test_labels': sorted(set(labels) - {s['label'] for s in test_all}),
          'synthetic':any(s.get('provenance',{}).get('type')=='synthetic' for s in selected),
          'folds':[{'train_ids':development.iloc[tr].id.tolist(),'validation_ids':development.iloc[va].id.tolist(),
                    'train_groups':sorted(set(groups[tr])),'validation_groups':sorted(set(groups[va]))} for tr,va in split],
          'hypothesis':'Compare registered landmark transformations on identical development photo IDs and grouped folds.'}
    return plan


def _save_experiment(out, plan, config):
    out.mkdir(parents=True)
    atomic_json(out / 'split.json', plan)
    atomic_json(out / 'finger_thresholds.json', config)
    snapshot = out / 'code_snapshot'
    snapshot.mkdir()
    for name in code_files():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / name, destination)


def _dataset_evaluation_note(project):
    state = read_manifest(project).get('import', {})
    return ' '.join(filter(None, (state.get('grouping_note'), state.get('test_note'))))


def prepare_experiment(project, out, folds=3, grouping='session', labels=None, model='svm', seed=42, allow_no_test=False, allow_incomplete_test=False, classifier_profile='project', split_from=None, index_middle_tip_weight=1.0, mirror_training=False):
    """Validate the dataset and freeze one shared experiment for every input variant."""
    project, out = Path(project).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('This experiment directory already exists. Use a new name; existing results are preserved.')
    weight = validate_tip_weight(index_middle_tip_weight, model)
    if not isinstance(mirror_training, bool):
        raise ValueError('mirror_training must be a boolean.')
    selected, labels, development = _selected_development_samples(project, labels)
    config = load_thresholds(project)
    split, groups, parent = development_folds(project, development, selected, grouping, int(folds), int(seed), split_from)
    _validate_test_samples(selected, labels, grouping, allow_no_test, allow_incomplete_test)
    estimator(model, len(development), seed, profile=classifier_profile)
    parameters = dict(model=model, seed=int(seed), grouping=grouping, n_folds=int(folds),
                      classifier_profile=classifier_profile, classifier_versions=classifier_versions(model), split_parent=parent,
                      index_middle_tip_weight=weight, mirror_training=mirror_training,
                      augmentation_definition=MIRROR_DEFINITION if mirror_training else None)
    plan = _build_experiment_plan(project, development, selected, labels, split, groups, config, parameters)
    plan['evaluation_note'] = _dataset_evaluation_note(project)
    _save_experiment(out, plan, config)
    return out


def load_experiment(experiment,project_override=None):
    experiment=Path(experiment).resolve()
    plan=json.loads((experiment/'split.json').read_text(encoding='utf-8'))
    project=Path(project_override).resolve() if project_override else Path(plan['project'])
    if not (project/'manifest.json').exists():
        raise ValueError('Project not found. Use --project with the new location of the same dataset.')
    if fingerprint(project)!=plan['dataset_fingerprint']:
        raise ValueError('Dataset changed since splitting. Create a new experiment directory for both variants.')
    if code_hashes()!=plan['code_hashes']:
        raise ValueError('Feature or experiment code changed. Create a new experiment directory for both variants.')
    if sklearn.__version__!=plan['sklearn_version']:
        raise ValueError('Use the same scikit-learn version as when the split was created.')
    if classifier_versions(plan['model']) != plan.get('classifier_versions', {}):
        raise ValueError('Use the same classifier library version as when the split was created.')
    records={s['id']:s for s in read_manifest(project)['samples']}
    ids=plan['development_ids']+plan['test_ids']
    for sample_id in ids:
        sample=records[sample_id]
        image_path=project/sample['image_path']
        if not image_path.is_file():
            raise ValueError('Source images are missing; use the complete dataset export.')
    return experiment,project,plan,records


