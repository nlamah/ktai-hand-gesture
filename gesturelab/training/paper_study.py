"""Run the five-input Shin 2021 application with identical folds for both classifiers."""
import html
import json
from pathlib import Path

import pandas as pd

from gesturelab.datasets.data import atomic_json
from gesturelab.features.training_inputs import PAPER_COMPARISON
from gesturelab.training.comparison import compare_variants
from gesturelab.training.experiment_plan import prepare_experiment
from gesturelab.training.fitting import train_variant
from gesturelab.training.reporting import summary_html

STUDY_NOTE = 'Application of Shin et al. (2021) to this dataset, not a reproduction of published scores.'
ASSIGNMENT_KEYS = ('dataset_fingerprint', 'labels', 'grouping', 'seed', 'n_folds',
                   'development_ids', 'excluded_development_ids', 'test_ids', 'folds')


def shared_assignment(plan):
    """Classifier-independent assignment, including every exact train/validation ID."""
    return {key: plan[key] for key in ASSIGNMENT_KEYS}


def _prepare_study(project, out, models, options):
    reference = None
    for model in models:
        experiment = prepare_experiment(project, out / model, model=model, **options)
        plan = json.loads((experiment / 'split.json').read_text())
        assignment = shared_assignment(plan)
        if reference is not None and assignment != reference:
            raise ValueError('Classifier experiments do not share identical labels, photo IDs and grouped folds.')
        reference = assignment
    atomic_json(out / 'shared_split.json', reference)


def _study_report(out, rows, note):
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / 'metrics.csv', index=False)
    links = ''.join(f'<li><a href="{model}/{report}">{model}: detailed report</a></li>'
                    for model, report in table[['classifier', 'report']].drop_duplicates().itertuples(index=False, name=None))
    body = ('<html lang="en"><meta charset="utf-8"><title>Shin 2021 application</title>'
            '<body style="font-family:Arial;max-width:1400px;margin:30px auto">'
            '<h1>Shin 2021 application: development cross-validation</h1><p>' + html.escape(note) + '</p>'
            '<p>All classifiers and inputs share the photo IDs, labels and grouped folds in shared_split.json. '
            'This table contains no final-test scores. Scores from synthetic data test software only.</p>'
            + summary_html(table) + '<ul>' + links + '</ul></body></html>')
    (out / 'report.html').write_text(body, encoding='utf-8')


def _train_study(out, models):
    rows = []
    for model in models:
        experiment = out / model
        for variant in PAPER_COMPARISON:
            print(f'Training {model}: {variant}', flush=True)
            train_variant(experiment, variant)
        report = compare_variants(experiment, variants=PAPER_COMPARISON)
        rows.append(pd.read_csv(report / 'metrics.csv').assign(classifier=model, report=report.name + '/report.html'))
    return rows


def run_paper_study(project, out, models=('svm', 'lightgbm'), **split_options):
    """Create a new study; select all dataset letters unless explicitly requested otherwise."""
    out = Path(out).resolve()
    models = tuple(models)
    if out.exists():
        raise FileExistsError('Study output already exists. Choose a new --out; old results are preserved.')
    if not models or len(set(models)) != len(models) or set(models) - {'svm', 'lightgbm'}:
        raise ValueError('Choose distinct study classifiers from svm and lightgbm.')
    _prepare_study(project, out, models, split_options)
    plan = json.loads((out / models[0] / 'split.json').read_text())
    note = ('SYNTHETIC SOFTWARE TEST. ' if plan['synthetic'] else '') + STUDY_NOTE
    note += ' ' + plan.get('evaluation_note', '')
    rows = _train_study(out, models)
    _study_report(out, rows, note)
    atomic_json(out / 'study.json', {'note': note, 'models': models, 'variants': PAPER_COMPARISON,
                'classifier_profile': plan.get('classifier_profile', 'project'), 'final_test_evaluated': False})
    return out
