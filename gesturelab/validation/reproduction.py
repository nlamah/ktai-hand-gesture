"""Explicit one-variant replay with observed fit inputs; original artifacts stay read-only."""
from collections import Counter
import hashlib
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from gesturelab.training import fitting
from gesturelab.training.experiment_plan import load_experiment
from gesturelab.training.mirror_augmentation import fit_row_ids
from gesturelab.validation.metrics import read_csv, score_rows, fold_summary
from gesturelab.validation.provenance import environment_record, InputWatch
from gesturelab.validation.report import AuditReport, new_output, read_json, sha256, write_json


def _scalers(model, prefix='model'):
    result = []
    if hasattr(model, 'n_samples_seen_'):
        result.append(dict(path=prefix, type=type(model).__name__, fitted_rows=np.asarray(model.n_samples_seen_).tolist()))
    for name, component in getattr(model, 'steps', []):
        result.extend(_scalers(component, prefix + '/' + name))
    for name, component, _ in getattr(model, 'transformers_', []):
        if not isinstance(component, str):
            result.extend(_scalers(component, prefix + '/' + name))
    return result


class FitObserver:
    def __init__(self, plan, out):
        self.plan, self.out, self.events = plan, out, []
        self.training, self.columns = None, None
        self.original_data, self.original_factory = fitting._training_data, fitting._estimator_for

    def training_data(self, *args):
        self.training, self.columns = self.original_data(*args)
        return self.training, self.columns

    def estimator(self, *args):
        model = self.original_factory(*args)
        index = len(self.events)
        ids = (self.plan['folds'][index]['train_ids'] if index < len(self.plan['folds'])
               else self.plan['development_ids'])
        ids = fit_row_ids(self.plan, ids)
        original_fit = model.fit
        def fit(matrix, labels, *more, **kwargs):
            expected = self.training.loc[ids]
            matched = np.array_equal(matrix, expected[self.columns].to_numpy()) and np.array_equal(labels, expected.label.to_numpy())
            if not matched:
                raise ValueError('Observed fit input differs from the declared photo rows.')
            try:
                result = original_fit(matrix, labels, *more, **kwargs)
                self.record(index, ids, matrix, model)
                return result
            finally:
                del model.fit
        model.fit = fit
        return model

    def record(self, index, ids, matrix, model):
        event = dict(stage='fold' if index < len(self.plan['folds']) else 'final_fit', fold=index,
            train_ids=list(ids), rows=len(ids), features=len(self.columns), input_matrix_matches_declared_ids=True,
            input_sha256=hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest(), scalers=_scalers(model))
        self.events.append(event)
        write_json(self.out / 'fit_trace.json', self.events)


def _check_trace(report, observer):
    events = observer.events
    correct = len(events) == len(observer.plan['folds']) + 1
    for event in events:
        for scaler in event['scalers']:
            correct = correct and bool(np.all(np.asarray(scaler['fitted_rows']) == event['rows']))
    report.add('reproduction.actual_fit_inputs', 'Instrument the actual estimator.fit call, compare numeric matrix/labels with declared training rows, inspect fitted scaler counts.',
        'Every fold fit matches its training matrix; final fit uses development only.',
        dict(fits=len(events), scaler_counts_correct=correct), correct, [observer.out / 'fit_trace.json'],
        limitation='The trace proves observed numeric inputs matched declared photo rows. Identical numeric rows cannot identify distinct photos by themselves; no retrospective trace is invented.')


def _compare_replay(report, experiment, run, variant, labels, out):
    original = read_csv(experiment / variant / 'cv_predictions.csv')
    replayed = read_csv(run / variant / 'cv_predictions.csv')
    mapping = lambda rows: Counter((r['id'], r['label'], r['fold'], r['prediction']) for r in rows)
    same = mapping(original) == mapping(replayed)
    a, b = score_rows(original, labels), score_rows(replayed, labels)
    write_json(out / 'prediction_comparison.json', dict(original=a, replay=b,
        original_folds=fold_summary(original, labels), replay_folds=fold_summary(replayed, labels),
        changed_predictions=list((mapping(replayed) - mapping(original)).elements())[:50]))
    report.add('reproduction.predictions_scores', 'Compare OOF mappings and independently recomputed scores, not binary model identity.',
        'Identical predictions/scores under the recorded deterministic settings.',
        dict(original_rows=len(original), replay_rows=len(replayed), same_predictions=same,
             original_macro_f1=a['macro_f1'], replay_macro_f1=b['macro_f1']), same,
        [out / 'prediction_comparison.json'],
        limitation='Matching one classifier variant does not validate every experiment or scientific independence.')


def reproduce_variant(experiment, variant, out, project=None):
    experiment = Path(experiment).resolve()
    source_plan = read_json(experiment / 'split.json')
    source_project = Path(project or source_plan['project']).resolve()
    out = new_output(out, [experiment, source_project])
    report, watch = AuditReport('One-variant reproducibility audit'), InputWatch()
    watch.tree(experiment)
    watch.add(source_project / 'manifest.json', digest=True)
    environment_record(out)
    command = f'python -m scripts.reproduce_experiment --experiment {experiment} --variant {variant} --out <new-output-directory>'
    executing = False
    try:
        _, _, plan, _ = load_experiment(experiment, project)
        executing = True
        observer = _execute_replay(experiment, variant, out, project, plan)
        _check_trace(report, observer)
        _compare_replay(report, experiment, out / 'run', variant, plan['labels'], out)
    except (ValueError, OSError, ImportError) as error:
        report.add('reproduction.available', 'Require matching source, versions, data and saved source predictions; do not disable guards.',
            'Practical one-variant replay with complete evidence.', str(error),
            condition=False if executing else None,
            limitation='Reproducibility was not established.', next_step=command)
    watch.finish(report)
    report.save(out, experiment=str(experiment), variant=variant, command=command,
                model_identity_policy='Binary identity is not required; predictions and independent scores are compared.')
    return out


def _execute_replay(experiment, variant, out, project, plan):
    if not (experiment / variant / 'cv_predictions.csv').is_file():
        raise ValueError('The selected source variant has no saved CV predictions.')
    run = out / 'run'
    run.mkdir()
    shutil.copyfile(experiment / 'split.json', run / 'split.json')
    shutil.copytree(experiment / 'code_snapshot', run / 'code_snapshot')
    observer = FitObserver(plan, out)
    write_json(out / 'source_provenance.json', dict(experiment=str(experiment), variant=variant,
        split_sha256=sha256(experiment / 'split.json'), dataset_fingerprint=plan['dataset_fingerprint'],
        seed=plan['seed'], classifier=plan['model'], profile=plan.get('classifier_profile', 'project'),
        feature_configuration=plan['feature_config'], note='No original files copied back or overwritten.'))
    with patch.object(fitting, '_training_data', observer.training_data), patch.object(fitting, '_estimator_for', observer.estimator):
        fitting.train_variant(run, variant, project)
    return observer
