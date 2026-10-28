"""Read-only comparison of two classifiers on identical saved OOF inputs and rows."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gesturelab.features.training_inputs import validate_model_input
from gesturelab.training.experiments import heatmap
from gesturelab.validation.metrics import read_csv, score_rows, fold_summary
from gesturelab.validation.report import read_json, sha256, write_json, new_output
from gesturelab.validation.splits import canonical_assignment


def _artifact(experiment, variant):
    experiment = Path(experiment).resolve()
    folder = experiment / variant
    plan, training = read_json(experiment / 'split.json'), read_json(folder / 'training.json')
    if training['model_sha256'] != sha256(folder / 'model.joblib'):
        raise ValueError('Model hash differs from the saved training record.')
    if training['split_sha256'] != sha256(experiment / 'split.json'):
        raise ValueError('Training record does not match its saved split.')
    bundle = joblib.load(folder / 'model.joblib')
    validate_model_input(bundle)
    if bundle.get('training_input') != variant or bundle['estimator'].n_features_in_ != len(bundle['config']['columns']):
        raise ValueError('Actual model input variant or dimensions differ from the selected artifact.')
    if set(map(str, bundle['estimator'].classes_)) != set(plan['labels']):
        raise ValueError('Actual model classes differ from the frozen classes.')
    if bundle['split_sha256'] != training['split_sha256']:
        raise ValueError('Saved model does not belong to the selected split.')
    rows = read_csv(folder / 'cv_predictions.csv')
    _validate_oof(rows, plan)
    return dict(experiment=experiment, folder=folder, plan=plan, training=training, bundle=bundle, rows=rows)


def _validate_oof(rows, plan):
    expected = Counter((identity, str(index)) for index, fold in enumerate(plan['folds']) for identity in fold['validation_ids'])
    actual = Counter((r['id'], r['fold']) for r in rows)
    if actual != expected or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('OOF predictions do not match the frozen original validation IDs and folds.')
    if any(r['label'] not in plan['labels'] or r['prediction'] not in plan['labels'] for r in rows):
        raise ValueError('OOF true/predicted labels fall outside the complete frozen classes.')


def _preprocessing(artifact):
    model = artifact['bundle']['estimator']
    steps = getattr(model, 'steps', [])[:-1]
    return [(type(step).__name__, repr(step.get_params(deep=True))) for _, step in steps]


def _check_controls(first, second):
    a, b = first['plan'], second['plan']
    if canonical_assignment(a) != canonical_assignment(b):
        raise ValueError('Classifiers must share original IDs, classes, grouping and every train/validation fold.')
    keys = ('dataset_fingerprint', 'feature_config', 'seed', 'sklearn_version', 'numpy_version', 'pandas_version')
    if any(a.get(k) != b.get(k) for k in keys):
        raise ValueError('Dataset, feature settings, seed or numeric library versions differ.')
    for key, default in (('index_middle_tip_weight', 1.), ('mirror_training', False)):
        if a.get(key, default) != b.get(key, default):
            raise ValueError('Distance weighting or mirror augmentation differs between classifiers.')
    x, y = first['bundle'], second['bundle']
    if x['config']['columns'] != y['config']['columns']:
        raise ValueError('Model input columns/order differ.')
    keys = ('feature_code_hashes', 'feature_config', 'input_definition', 'input_preprocessing', 'feature_version')
    if any(x.get(k) != y.get(k) for k in keys) or _preprocessing(first) != _preprocessing(second):
        raise ValueError('Feature definitions or preprocessing policies differ.')
    classifiers = [v['estimator'].steps[-1][1] if hasattr(v['estimator'], 'steps') else v['estimator'] for v in (x, y)]
    if any(getattr(c, 'class_weight', None) != 'balanced' for c in classifiers):
        raise ValueError('This controlled comparison requires balanced class weighting for both classifiers.')


def _matrix_signature(path):
    """Compare numeric rows by ID, independent of CSV quoting and row ordering."""
    mapping = {}
    with path.open(newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if len(row) != len(header) or row[0] in mapping:
                raise ValueError('Malformed or duplicated training matrix rows.')
            values = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite training matrix values.')
            mapping[row[0]] = hashlib.sha256(values.tobytes()).hexdigest()
    return header, mapping


def _check_matrices(first, second):
    print('Checking saved numeric training matrices; no model is fitted.', flush=True)
    a, b = [_matrix_signature(item['folder'] / 'training_features.csv') for item in (first, second)]
    expected = list(first['plan']['development_ids'])
    if first['plan'].get('mirror_training'):
        expected += [identity + '::horizontal_mirror' for identity in expected]
    columns = ['id'] + first['bundle']['config']['columns']
    if a != b or a[0] != columns or set(a[1]) != set(expected):
        raise ValueError('Saved original/mirrored numeric input rows differ between classifiers.')
    return dict(rows=len(a[1]), features=len(a[0]) - 1, identical_numeric_rows=True,
        canonical_sha256=hashlib.sha256(json.dumps(sorted(a[1].items())).encode()).hexdigest())


def _prediction_pairs(first, second):
    a = {r['id']: r for r in first['rows']}
    b = {r['id']: r for r in second['rows']}
    if any(a[i]['label'] != b[i]['label'] or a[i]['fold'] != b[i]['fold'] for i in a):
        raise ValueError('Saved true labels or OOF folds differ.')
    return [dict(id=i, label=a[i]['label'], fold=a[i]['fold'], baseline=a[i]['prediction'], candidate=b[i]['prediction'],
        baseline_correct=a[i]['label'] == a[i]['prediction'], candidate_correct=b[i]['label'] == b[i]['prediction']) for i in a]


def _write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _export_scores(out, side, artifact):
    scores = score_rows(artifact['rows'], artifact['plan']['labels'])
    folds = fold_summary(artifact['rows'], artifact['plan']['labels'])
    write_json(out / f'{side}_scores.json', dict(pooled=scores, folds=folds))
    _write_csv(out / f'{side}_per_letter.csv', scores['per_class'])
    table = pd.DataFrame(scores['confusion'], index=scores['labels'], columns=scores['predicted_columns'])
    table.to_csv(out / f'{side}_confusion_counts.csv')
    normalized = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    normalized.to_csv(out / f'{side}_confusion_normalized.csv')
    heatmap(normalized, out / f'{side}_confusion.png', f'{side} | actual rows, predicted columns | development OOF')
    return dict(model=side, classifier=artifact['plan']['model'], accuracy=scores['accuracy'], macro_f1=scores['macro_f1'],
        cv_macro_f1_mean=folds['summary']['macro_f1_mean'], cv_macro_f1_std=folds['summary']['macro_f1_std'],
        n=scores['n'], n_errors=sum(r['label'] != r['prediction'] for r in artifact['rows']),
        iteration_limit_reached=artifact['training'].get('fit_diagnostics', {}).get('iteration_limit_reached'))


def _model_evidence(artifact):
    model = artifact['bundle']['estimator']
    classifier = model.steps[-1][1] if hasattr(model, 'steps') else model
    return dict(experiment=str(artifact['experiment']), classifier=type(classifier).__name__,
        model_sha256=sha256(artifact['folder'] / 'model.joblib'),
        split_sha256=sha256(artifact['experiment'] / 'split.json'),
        predictions_sha256=sha256(artifact['folder'] / 'cv_predictions.csv'),
        parameters=repr(classifier.get_params()), class_weight_policy=classifier.class_weight,
        fit_diagnostics=artifact['training'].get('fit_diagnostics'),
        preprocessing=_preprocessing(artifact))


def _report(out, first, second, matrix):
    pairs = _prediction_pairs(first, second)
    summaries = [_export_scores(out, side, item) for side, item in (('baseline', first), ('candidate', second))]
    _write_csv(out / 'metrics.csv', summaries)
    _write_csv(out / 'paired_predictions.csv', pairs)
    corrected = sum(not r['baseline_correct'] and r['candidate_correct'] for r in pairs)
    new = sum(r['baseline_correct'] and not r['candidate_correct'] for r in pairs)
    result = dict(stage='development_oof', variant=first['bundle']['training_input'],
        same_original_assignments=True, same_numeric_training_inputs=matrix, models=[_model_evidence(v) for v in (first, second)],
        metrics=summaries, corrected_errors=corrected, new_errors=new,
        shared_errors=sum(not r['baseline_correct'] and not r['candidate_correct'] for r in pairs),
        macro_f1_difference=summaries[1]['macro_f1'] - summaries[0]['macro_f1'],
        synthetic=first['plan']['synthetic'] or second['plan']['synthetic'],
        comparison_code_sha256=sha256(__file__),
        limitation='Saved artifacts support this controlled development comparison, not unrecorded historic execution. No final test or significance claim. Filename blocks are not verified people.')
    write_json(out / 'result.json', result)
    title = 'SYNTHETIC SOFTWARE TEST' if result['synthetic'] else 'Controlled classifier comparison — development OOF'
    body = f'<html lang="en"><meta charset="utf-8"><title>{title}</title><body><h1>{title}</h1>'
    body += '<p>Same input values, preprocessing policy, original photos, classes and folds; balanced class weighting. No mirror or tip-weight difference.</p>'
    body += pd.DataFrame(summaries).to_html(index=False) + f'<p>Corrected errors: {corrected}; new errors: {new}.</p>'
    body += '<p>' + html.escape(result['limitation']) + '</p>'
    body += ''.join(f'<img style="max-width:100%" src="{side}_confusion.png">' for side in ('baseline', 'candidate'))
    (out / 'report.html').write_text(body + '</body></html>', encoding='utf-8')


def compare_classifiers(baseline, candidate, out, variant='shin2021_combined'):
    first, second = _artifact(baseline, variant), _artifact(candidate, variant)
    _check_controls(first, second)
    matrix = _check_matrices(first, second)
    out = new_output(out, [first['experiment'], second['experiment'], first['plan']['project'], second['plan']['project']])
    _report(out, first, second, matrix)
    return out / 'report.html'
