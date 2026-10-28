"""Controlled SVM/MLP comparison and weighted fits; synthetic software checks only."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.datasets.data import add_image
from gesturelab.features.training_inputs import model_features, validate_model_input
from gesturelab.research.models import register_model, ModelHandle
from gesturelab.training import fitting
from gesturelab.training.classifiers import estimator, classifier_parameters
from gesturelab.training.classifier_comparison import compare_classifiers
from gesturelab.training.experiment_plan import prepare_experiment, load_experiment
from gesturelab.training.neural_network import BalancedMLPClassifier, training_diagnostics
from gesturelab.validation.report import read_json, sha256, write_json


class MLPConfigurationTests(unittest.TestCase):
    def test_fixed_architecture_seed_and_same_shin_preprocessing_as_svm(self):
        parameters = classifier_parameters('mlp', 'shin2021_combined', seed=13)
        self.assertEqual(parameters['hidden_layer_sizes'], (64, 32))
        self.assertEqual(parameters['random_state'], 13)
        self.assertFalse(parameters['early_stopping'])
        svm = estimator('svm', variant='shin2021_combined')
        mlp = estimator('mlp', variant='shin2021_combined')
        self.assertFalse(hasattr(svm, 'steps'))
        self.assertFalse(hasattr(mlp, 'steps'))
        self.assertIsInstance(mlp, MLPClassifier)
        self.assertIsInstance(estimator('mlp', variant='with_features').steps[0][1], StandardScaler)
        with self.assertRaises(ValueError): classifier_parameters('mlp', profile='shin2021_asl')
        with self.assertRaises(ValueError): estimator('mlp', index_middle_tip_weight=4)

    def test_balancing_uses_only_labels_passed_to_each_fit_and_combines_explicit_weights(self):
        model = BalancedMLPClassifier()
        x, y = np.ones((4, 2)), np.array(['U', 'U', 'U', 'R'])
        with patch.object(MLPClassifier, 'fit', autospec=True) as fit:
            model.fit(x, y)
            np.testing.assert_allclose(fit.call_args.kwargs['sample_weight'], [2/3, 2/3, 2/3, 2])
            model.fit(x, y, sample_weight=[1, 2, 1, 1])
            np.testing.assert_allclose(fit.call_args.kwargs['sample_weight'], [2/3, 4/3, 2/3, 2])
        self.assertEqual(model.training_class_counts_, {'R': 1, 'U': 3})
        self.assertEqual(model.class_balance_weights_, {'R': 2., 'U': 2/3})
        self.assertEqual(clone(model).get_params(), model.get_params())

    def test_iteration_limit_is_recorded_instead_of_silently_claiming_convergence(self):
        model = BalancedMLPClassifier(hidden_layer_sizes=(4,), max_iter=1, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit([[0., 1.], [1., 0.], [1., 1.]], ['U', 'V', 'R'])
        self.assertTrue(any(issubclass(w.category, ConvergenceWarning) for w in caught))
        diagnostics = training_diagnostics(model)
        self.assertTrue(diagnostics['iteration_limit_reached'])
        self.assertEqual(diagnostics['optimizer_iterations'], 1)
        self.assertEqual(diagnostics['class_weight_policy'], 'balanced')


def _photo(project, group, label):
    world = skeleton({'U': 'open_hand', 'V': 'peace', 'R': 'pointing', 'J': 'fist', 'Z': 'thumbs_up'}[label])
    world[8, 0] += group * .01
    buffer = BytesIO()
    Image.new('RGB', (32, 32), (group * 40, ord(label), 80)).save(buffer, format='PNG')
    return add_image(project, buffer.getvalue(), 'synthetic.png', label, f's{group}', f'p{group}',
        role='test' if group == 3 else 'development', provenance={'type': 'synthetic'},
        analysis=dict(status='ok', image_landmarks=(world * .2 + .5).tolist(), world_landmarks=world.tolist()))[0]


class MLPTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'data'
        for group in range(4):
            for label in 'UVRJZ': _photo(self.project, group, label)
        self.baseline = prepare_experiment(self.project, self.root / 'baseline')
        self.candidate = prepare_experiment(self.project, self.root / 'mlp', model='mlp', split_from=self.baseline)

    def _train_both(self):
        fitting.train_variant(self.baseline, 'shin2021_combined')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            fitting.train_variant(self.candidate, 'shin2021_combined')

    def test_same_frozen_inputs_fold_fits_balancing_and_no_extra_shin_scaler(self):
        _, _, plan, records = load_experiment(self.candidate)
        table, columns = fitting._training_data(plan, records, 'shin2021_combined')
        original, observed = MLPClassifier.fit, []
        def fit(model, x, y, sample_weight=None):
            stage = len(observed)
            ids = plan['folds'][stage]['train_ids'] if stage < 3 else plan['development_ids']
            np.testing.assert_array_equal(x, table.loc[ids, columns].to_numpy())
            np.testing.assert_array_equal(y, table.loc[ids].label.to_numpy())
            self.assertEqual(len(sample_weight), len(ids))
            observed.append(list(ids))
            return original(model, x, y, sample_weight=sample_weight)
        with patch.object(MLPClassifier, 'fit', fit), patch.object(StandardScaler, 'fit', side_effect=AssertionError('Extra Shin scaler')), warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            fitting.train_variant(self.candidate, 'shin2021_combined')
        self.assertEqual([len(ids) for ids in observed], [10, 10, 10, 15])
        self.assertFalse(set().union(*map(set, observed)) & set(plan['test_ids']))
        info = read_json(self.candidate / 'shin2021_combined/training.json')
        self.assertEqual(info['index_middle_tip_weight'], 1)
        self.assertFalse(info['mirror_training'])
        self.assertEqual(info['fit_diagnostics']['class_weight_policy'], 'balanced')
        self.assertFalse(info['fit_diagnostics']['early_stopping'])

    def test_controlled_report_live_inference_and_original_file_preservation(self):
        before = {p: sha256(p) for p in self.project.rglob('*') if p.is_file()}
        self._train_both()
        before.update({p: sha256(p) for p in self.baseline.rglob('*') if p.is_file()})
        report = compare_classifiers(self.baseline, self.candidate, self.root / 'comparison')
        result = read_json(report.parent / 'result.json')
        self.assertTrue(result['same_original_assignments'])
        self.assertTrue(result['same_numeric_training_inputs']['identical_numeric_rows'])
        self.assertEqual([m['n'] for m in result['metrics']], [15, 15])
        registration = register_model(self.candidate / 'shin2021_combined/model.joblib', self.root / 'registered.json', self.candidate, 'ASL')
        handle = ModelHandle.load(registration)
        self.assertEqual(set(handle.metadata['labels']), set('UVRJZ'))
        validate_model_input(handle.bundle)
        _, _, plan, records = load_experiment(self.candidate)
        for identity in plan['development_ids']:
            sample = records[identity]
            features = model_features(handle.bundle, sample)
            expected = handle.bundle['estimator'].predict([[features[c] for c in handle.bundle['config']['columns']]])[0]
            self.assertEqual(handle.predict(sample, 32, 32), expected)
        self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))
        self.assertIn('SYNTHETIC SOFTWARE TEST', report.read_text())
        with self.assertRaises(FileExistsError): compare_classifiers(self.baseline, self.candidate, report.parent)

    def test_comparison_rejects_changed_weight_augmentation_predictions_and_numeric_values(self):
        self._train_both()
        plan_path = self.candidate / 'split.json'
        original_plan = read_json(plan_path)
        for key, value in (('index_middle_tip_weight', 4), ('mirror_training', True)):
            changed = {**original_plan, key: value}
            # Check the comparison policy separately from on-disk split integrity.
            from gesturelab.training.classifier_comparison import _artifact as load, _check_controls
            a, b = load(self.baseline, 'shin2021_combined'), load(self.candidate, 'shin2021_combined')
            b['plan'] = changed
            with self.assertRaises(ValueError): _check_controls(a, b)
        path = self.candidate / 'shin2021_combined/cv_predictions.csv'
        original_predictions = path.read_bytes()
        predictions = pd.read_csv(path)
        predictions.loc[0, 'fold'] = 99
        predictions.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'OOF'): compare_classifiers(self.baseline, self.candidate, self.root / 'bad_oof')
        path.write_bytes(original_predictions)
        matrix = self.candidate / 'shin2021_combined/training_features.csv'
        values = pd.read_csv(matrix)
        values.iloc[0, 1] += 1
        values.to_csv(matrix, index=False)
        with self.assertRaisesRegex(ValueError, 'numeric'): compare_classifiers(self.baseline, self.candidate, self.root / 'bad_matrix')


if __name__ == '__main__':
    unittest.main()
