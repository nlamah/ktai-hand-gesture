"""Inspect saved final-fit matrices/models without claiming unrecorded fold execution."""
from collections import Counter
import csv

import joblib
import numpy as np

from gesturelab.features.training_inputs import model_features, validate_model_input
from gesturelab.validation.report import read_json, sha256

RESERVED_COLUMNS = {'id', 'label', 'filename', 'participant', 'session', 'condition', 'role', 'image_path', 'fold'}


def training_matrix_identity(path, expected_ids):
    """Stream all row identities; retain only three example rows for numeric reconstruction."""
    ids, examples = [], []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Training input CSV has malformed rows.')
            ids.append(row[0])
            if len(examples) < 3:
                examples.append(row)
    return dict(header=header, rows=len(ids), unique=len(set(ids)),
                correct=Counter(ids) == Counter(expected_ids), examples=examples)


def _expected_fit_ids(plan):
    expected = list(plan['development_ids'])
    if plan.get('mirror_training'):
        expected += [identity + '::horizontal_mirror' for identity in plan['development_ids']]
    return expected


def audit_artifacts(report, context, variant):
    folder = context['experiment'] / variant
    info = read_json(folder / 'training.json')
    path = folder / 'training_features.csv'
    matrix = training_matrix_identity(path, _expected_fit_ids(context['plan']))
    columns = matrix['header'][1:]
    correct = (matrix['header'][0] == 'id' and columns == info['feature_columns']
               and len(columns) == info['n_features'] and not RESERVED_COLUMNS & set(columns))
    report.add('inputs.final_fit_matrix', 'Stream every saved final-fit row ID and inspect all input column names.',
        'Exactly the declared original and optional mirror-training rows, no test IDs or metadata columns.',
        dict(rows=matrix['rows'], unique=matrix['unique'], correct_ids=matrix['correct'], correct_columns=correct),
        matrix['correct'] and correct, [path], limitation='This CSV describes the final input matrix, not historical per-fold execution.')
    model_path = folder / 'model.joblib'
    intact = sha256(model_path) == info['model_sha256']
    report.add('model.integrity', 'Check saved model against its recorded hash before loading the trusted local artifact.',
               info['model_sha256'], sha256(model_path), intact, [model_path, folder / 'training.json'])
    if not intact:
        return
    bundle = joblib.load(model_path)
    valid = columns == bundle['config']['columns'] and bundle['estimator'].n_features_in_ == len(columns)
    report.add('inputs.saved_column_order', 'Compare CSV, training metadata and fitted model input schema.',
               'Identical column order/count.', dict(features=len(columns), correct=valid), valid, [path, model_path])
    _inference_parity(report, context, bundle, matrix, model_path)


def _inference_parity(report, context, bundle, matrix, path):
    try:
        validate_model_input(bundle)
    except ValueError as error:
        report.add('inputs.inference_parity', 'Validate source hashes before reconstructing saved inputs.',
            'Matching feature code and numeric input order.', str(error), files=[path],
            limitation='Current code cannot stand in for archived feature code; guards were not disabled.')
        return
    errors = []
    for row in matrix['examples']:
        values = model_features(bundle, context['records'][row[0]])
        if list(values) != matrix['header'][1:] or not np.allclose(list(values.values()),
                np.asarray(row[1:], dtype=float), rtol=1e-10, atol=1e-10):
            errors.append(row[0])
    report.add('inputs.inference_parity', 'Reconstruct the first three final-fit rows through the live feature interface.',
        'Same values and exact column order as saved training inputs.', dict(examples=len(matrix['examples']), wrong_ids=errors),
        not errors and bool(matrix['examples']), [path],
        limitation='Three-row reconstruction is a targeted check, not a complete proof for every possible hand.')
