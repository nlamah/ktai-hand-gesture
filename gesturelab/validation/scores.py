"""Recompute saved predictions independently and compare, never overwrite, reports."""
from collections import Counter
import csv
import math

from gesturelab.validation.metrics import UNKNOWN, read_csv, score_rows, fold_summary, numeric_difference
from gesturelab.validation.report import read_json, write_json


def prediction_membership(rows, plan, records, final=False):
    expected = Counter((i, '') for i in plan['test_ids']) if final else Counter(
        (i, str(f)) for f, fold in enumerate(plan['folds']) for i in fold['validation_ids'])
    actual = Counter((r['id'], '' if final else str(r.get('fold', ''))) for r in rows)
    bad_labels = [r['id'] for r in rows if r['id'] not in records or r['label'] != records[r['id']]['label']]
    bad_predictions = [r['id'] for r in rows if r['prediction'] not in list(plan['labels']) + [UNKNOWN]]
    failed_as_known = [r['id'] for r in rows if final and r['id'] in records
                       and records[r['id']]['status'] != 'ok' and r['prediction'] != UNKNOWN]
    return dict(expected_rows=sum(expected.values()), actual_rows=len(rows),
        missing_pairs=list((expected - actual).elements())[:10], extra_pairs=list((actual - expected).elements())[:10],
        bad_labels=bad_labels[:10], unknown_output_labels=bad_predictions[:10], failed_detection_predicted=failed_as_known[:10])


def _differences(expected, actual, tolerance):
    differences, missing = {}, []
    for name, value in expected.items():
        if name not in actual:
            missing.append(name)
        else:
            difference = numeric_difference(value, actual[name], tolerance)
            if difference:
                differences[name] = difference
    return differences, missing


def compare_numbers(report, name, expected, actual, files, tolerance):
    differences, missing = _differences(expected, actual, tolerance)
    condition = False if differences else (None if missing else True)
    report.add(name, f'Independent integer counts; absolute tolerance {tolerance}; missing evidence is not a pass.',
        expected, dict(differences=differences, missing_fields=missing), condition, files,
        limitation='Possible causes of differences include class-list policy, pooled versus fold averaging, ddof or rounding.')


def _training_scores(report, folder, independent, tolerance):
    info = read_json(folder / 'training.json')
    pooled = independent['pooled']
    keys = ('accuracy', 'macro_f1', 'balanced_accuracy', 'coverage', 'n')
    compare_numbers(report, 'scores.training_pooled', {k: pooled[k] for k in keys}, info.get('cv_pooled', {}),
                    [folder / 'cv_predictions.csv', folder / 'training.json'], tolerance)
    expected = {'cv_' + k: v for k, v in independent['folds']['summary'].items() if k.startswith(('accuracy', 'macro_f1'))}
    compare_numbers(report, 'scores.training_fold_summary', expected, info, [folder / 'training.json'], tolerance)
    path = folder / 'cv_folds.csv'
    if not path.exists():
        report.add('scores.fold_file', 'Read archived fold-score CSV.', 'Saved scores for each frozen fold.', 'File missing.', files=[path])
        return
    saved = read_csv(path)
    for row in saved:
        found = independent['folds']['folds'].get(str(row['fold']))
        compare_numbers(report, 'scores.fold_' + str(row['fold']), {k: found[k] for k in keys} if found else {'n': -1},
                        row, [path], tolerance)
    ids = [r['fold'] for r in saved]
    report.add('scores.fold_coverage', 'Count saved fold rows rather than trusting reported aggregate SD.',
        sorted(independent['folds']['folds']), ids,
        Counter(ids) == Counter(list(independent['folds']['folds'])), [path])


def _class_report(report, directory, variant, scores, tolerance):
    path = directory / (variant + '_per_label.csv')
    if not path.exists():
        report.add('scores.per_class', 'Read archived per-class precision/recall/F1/support.',
                   scores['labels'], 'File missing.', files=[path])
        return
    rows = read_csv(path)
    saved = {r['label']: r for r in rows}
    differences = []
    for expected in scores['per_class']:
        for metric in ('precision', 'recall', 'f1', 'support'):
            missing_support = expected['support'] == 0 and metric in ('recall', 'f1')
            diff = numeric_difference(expected[metric], saved.get(expected['label'], {}).get(metric), tolerance, missing_support)
            if diff:
                differences.append(dict(label=expected['label'], metric=metric, **diff))
    complete = Counter(r['label'] for r in rows) == Counter(scores['labels'])
    report.add('scores.per_class', 'TP/FP/FN counted from photo predictions; precision zero division=0; unsupported recall/F1 undefined in report.',
        'One row per frozen class with matching values.', dict(differences=differences[:30], complete=complete),
        not differences and complete, [path])


def _confusion_report(report, directory, variant, scores):
    path = directory / (variant + '_confusion.csv')
    if not path.exists():
        report.add('scores.confusion', 'Read confusion counts.', scores['predicted_columns'], 'File missing.', files=[path])
        return
    with path.open(newline='') as handle:
        rows = list(csv.reader(handle))
    correct = rows[0][1:] == scores['predicted_columns'] and [r[0] for r in rows[1:]] == scores['labels']
    found = [[float(x) for x in row[1:]] for row in rows[1:]]
    if any(not math.isfinite(x) for row in found for x in row):
        raise ValueError('Non-finite confusion counts.')
    report.add('scores.confusion', 'Direct integer confusion-count comparison; rows actual, final column unknown.',
        scores['confusion'], found, correct and found == scores['confusion'], [path])


def _comparison_report(report, directory, variant, scores, tolerance, folds=None):
    rows = [r for r in read_csv(directory / 'metrics.csv') if r['variant'] == variant]
    expected = {k: scores[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'coverage', 'n')}
    if folds:
        expected.update({'cv_' + k: v for k, v in folds['summary'].items() if k.startswith(('accuracy', 'macro_f1'))})
    compare_numbers(report, 'scores.report_metrics', expected, rows[0] if len(rows) == 1 else {},
                    [directory / 'metrics.csv'], tolerance)
    _class_report(report, directory, variant, scores, tolerance)
    _confusion_report(report, directory, variant, scores)


def audit_variant_scores(report, context, variant, out, tolerance):
    folder = context['experiment'] / variant
    rows = read_csv(folder / 'cv_predictions.csv')
    membership = prediction_membership(rows, context['plan'], context['records'])
    valid = not any(membership[k] for k in ('missing_pairs', 'extra_pairs', 'bad_labels', 'unknown_output_labels'))
    report.add('predictions.' + variant, 'Compare actual (photo ID, fold) multiplicities and labels against frozen mappings.',
        'Exact declared out-of-fold coverage, labels and output classes.', membership, valid,
        [folder / 'cv_predictions.csv', context['experiment'] / 'split.json'])
    independent = dict(pooled=score_rows(rows, context['plan']['labels']), folds=fold_summary(rows, context['plan']['labels']))
    write_json(out / (variant + '_independent_scores.json'), independent)
    _training_scores(report, folder, independent, tolerance)
    found_report = False
    for directory in context['experiment'].glob('comparison_cv*'):
        if (directory / 'metrics.csv').is_file() and (directory / (variant + '_predictions.csv')).is_file():
            found_report = True
            saved = read_csv(directory / (variant + '_predictions.csv'))
            same = Counter((r['id'], r['label'], r['prediction'], r['fold']) for r in rows) == Counter((r['id'], r['label'], r['prediction'], r['fold']) for r in saved)
            report.add('predictions.report_copy', 'Compare actual prediction mappings, independent of row order.', True, same, same,
                       [folder / 'cv_predictions.csv', directory / (variant + '_predictions.csv')])
            _comparison_report(report, directory, variant, independent['pooled'], tolerance, independent['folds'])
    if not found_report:
        report.add('scores.comparison_report', 'Locate published comparison tables.', 'A comparison report exists.', 'None found.')
    return rows, independent


def audit_final_scores(report, context, variant, out, tolerance):
    files = list(context['experiment'].glob('final_test*/' + variant + '_predictions.csv'))
    if not files:
        report.add('scores.final_test', 'Read only already saved final-test predictions; never generate new ones.',
            'Independent scores if a final evaluation was previously recorded.', 'No saved final predictions for ' + variant,
            limitation='Not evaluated is not a failed model or a passed final test.')
    for path in files:
        rows = read_csv(path)
        found = prediction_membership(rows, context['plan'], context['records'], final=True)
        correct = not any(found[k] for k in ('missing_pairs', 'extra_pairs', 'bad_labels', 'unknown_output_labels', 'failed_detection_predicted'))
        report.add('predictions.final_test', 'Every reserved ID once; failed detections must remain unknown.',
                   'Correct held-out membership and rejection handling.', found, correct, [path])
        scores = score_rows(rows, context['plan']['labels'], bool(context['plan'].get('missing_test_labels')))
        write_json(out / (path.parent.name + '_' + variant + '_scores.json'), scores)
        _comparison_report(report, path.parent, variant, scores, tolerance)
