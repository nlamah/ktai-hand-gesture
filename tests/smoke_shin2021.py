"""Synthetic end-to-end check for five inputs, both classifiers, reports and inference."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import make_demo
from gesturelab.datasets.data import atomic_json, read_manifest
from gesturelab.features.training_inputs import PAPER_COMPARISON, model_features, validate_model_input
from gesturelab.training.comparison import compare_variants
from gesturelab.training.experiment_plan import load_experiment
from gesturelab.training.experiments import predict_bundle
from gesturelab.training.paper_study import run_paper_study, shared_assignment
from gesturelab.training.rules import UNKNOWN


def _synthetic_failures(project):
    manifest = read_manifest(project)
    for role in ('development', 'test'):
        sample = next(s for s in manifest['samples'] if s['role'] == role)
        sample['status'] = 'no_hand'
        sample['provenance']['simulated_detection_failure'] = True
    atomic_json(project / 'manifest.json', manifest)


def _verify_inputs(experiment, plan, records):
    sample = records[plan['development_ids'][0]]
    for variant, count in zip(PAPER_COMPARISON, (126, 207, 190, 630, 820)):
        folder = experiment / variant
        bundle = joblib.load(folder / 'model.joblib')
        validate_model_input(bundle)
        values = model_features(bundle, sample)
        inputs = pd.read_csv(folder / 'training_features.csv', index_col=0)
        assert len(values) == count and list(values) == bundle['config']['columns']
        np.testing.assert_allclose(list(values.values()), inputs.loc[sample['id']].to_numpy(), atol=1e-12)
        assert str(predict_bundle(bundle, [values])[0]) in plan['labels']
        predictions = pd.read_csv(folder / 'cv_predictions.csv')
        assert set(predictions.id) == set(plan['development_ids'])
        assert not set(predictions.id) & set(plan['test_ids'])


def _verify_reports(experiment, plan, records):
    cv = compare_variants(experiment, variants=PAPER_COMPARISON)
    final = compare_variants(experiment, final_test=True, variants=PAPER_COMPARISON)
    assert compare_variants(experiment, final_test=True, variants=PAPER_COMPARISON) == final
    assert len(pd.read_csv(cv / 'cv_folds.csv')) == 15
    assert len(pd.read_csv(cv / 'pairwise_comparisons.csv')) == 10
    assert not (final / 'cv_folds.csv').exists()
    for variant in PAPER_COMPARISON:
        predictions = pd.read_csv(final / (variant + '_predictions.csv'))
        assert set(predictions.id) == set(plan['test_ids'])
        failed = {i for i in plan['test_ids'] if records[i]['status'] != 'ok'}
        assert all(predictions.set_index('id').loc[i, 'prediction'] == UNKNOWN for i in failed)
        for suffix in ('_per_label.csv', '_confusion.csv', '_confusion.png'):
            assert (cv / (variant + suffix)).is_file()
    assert 'SYNTHETIC' in (cv / 'report.html').read_text()
    assert json.loads((cv / 'result.json').read_text())['excluded_development_detections'] == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='examples/shin2021_software_test')
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if root.exists():
        parser.exit(1, 'Use a new output directory for the synthetic software test.\n')
    project = make_demo(root / 'dataset')
    _synthetic_failures(project)
    study = run_paper_study(project, root / 'study', folds=3, grouping='participant')
    assignments = []
    for model in ('svm', 'lightgbm'):
        experiment, _, plan, records = load_experiment(study / model)
        assignments.append(shared_assignment(plan))
        _verify_inputs(experiment, plan, records)
        _verify_reports(experiment, plan, records)
    assert assignments[0] == assignments[1]
    assert len(pd.read_csv(study / 'metrics.csv')) == 10
    print('SYNTHETIC SOFTWARE TEST PASSED: five inputs, SVM/LightGBM, shared IDs/folds, detection failures, reports, inference.')
    print('These scores do not measure real ASL performance.')
    print(study / 'report.html')


if __name__ == '__main__':
    main()
