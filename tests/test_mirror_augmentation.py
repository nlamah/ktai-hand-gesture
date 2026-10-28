"""Known reflection geometry and actual fit/predict membership; synthetic software only."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.datasets.data import add_image
from gesturelab.features.training_inputs import training_input, model_features, validate_model_input
from gesturelab.research.models import ModelHandle, register_model
from gesturelab.research.improvement import train_improvement
from gesturelab.training import fitting
from gesturelab.training.experiment_plan import prepare_experiment, load_experiment
from gesturelab.training.mirror_augmentation import mirror_sample, training_samples, fit_row_ids, MIRROR_SUFFIX
from gesturelab.validation.report import read_json, sha256
from gesturelab.validation.reproduction import reproduce_variant
from gesturelab.validation.runner import run_audit


def sample(label='R'):
    world = skeleton({'R': 'pointing', 'U': 'open_hand', 'V': 'peace', 'J': 'fist', 'Z': 'thumbs_up'}[label])
    return dict(id='source', label=label, role='development', status='ok', image_landmarks=(world * .2 + .5).tolist(),
                world_landmarks=world.tolist(), handedness='Right', features={'stale': 1}, width=32, height=32)


class MirrorGeometryTests(unittest.TestCase):
    def test_reflection_preserves_y_z_labels_indices_and_source_and_discards_stale_features(self):
        original = sample()
        before = np.asarray(original['image_landmarks']).copy()
        mirrored = mirror_sample(original)
        np.testing.assert_allclose(np.asarray(mirrored['image_landmarks'])[:, 0], 1 - before[:, 0])
        np.testing.assert_array_equal(np.asarray(mirrored['image_landmarks'])[:, 1:], before[:, 1:])
        world = np.asarray(original['world_landmarks'])
        np.testing.assert_array_equal(mirrored['world_landmarks'], world * [-1, 1, 1])
        np.testing.assert_array_equal(original['image_landmarks'], before)
        self.assertEqual(mirrored['label'], 'R')
        self.assertEqual(mirrored['handedness'], 'Left')
        self.assertEqual(mirrored['id'], 'source' + MIRROR_SUFFIX)
        self.assertNotIn('features', mirrored)
        self.assertEqual(original['features'], {'stale': 1})

    def test_distances_are_invariant_but_x_axis_cosines_reverse(self):
        original, mirrored = sample(), mirror_sample(sample())
        first = training_input('shin2021_combined').features(original, None)
        second = training_input('shin2021_combined').features(mirrored, None)
        for key, value in first.items():
            expected = -value if key.startswith('shin_angle_') and key.endswith('_cos_x') else value
            self.assertAlmostEqual(second[key], expected, places=12)
        self.assertTrue(any(abs(first[k] - second[k]) > .1 for k in first))
        for variant in ('finger_features_only', 'with_features', 'all_features'):
            values = training_input(variant).features(mirrored, None)
            self.assertTrue(np.isfinite(list(values.values())).all())

    def test_test_rows_and_invalid_geometry_are_rejected_and_default_is_unchanged(self):
        source = sample()
        self.assertIs(training_samples([source], False)[0], source)
        for forbidden in ({**source, 'role': 'test'}, {**source, 'provenance': {'protocol_role': 'test'}}):
            with self.assertRaises(ValueError): mirror_sample(forbidden)
        with self.assertRaises(ValueError): mirror_sample({**source, 'image_landmarks': [[float('nan')]*3]*21})
        self.assertEqual(fit_row_ids({}, ['a']), ['a'])
        self.assertEqual(fit_row_ids({'mirror_training': True}, ['a']), ['a', 'a' + MIRROR_SUFFIX])


def _photo(project, group, label):
    buffer = BytesIO()
    Image.new('RGB', (32, 32), (group * 40, ord(label), 100)).save(buffer, format='PNG')
    source = sample(label)
    source['image_landmarks'][8][0] += group * .003
    source['world_landmarks'][8][0] += group * .015
    return add_image(project, buffer.getvalue(), 'synthetic.png', label, f's{group}', f'p{group}',
        role='test' if group == 3 else 'development', provenance={'type': 'synthetic'}, analysis=source)[0]


class MirrorTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'data'
        for group in range(4):
            for label in 'UVRJZ': _photo(self.project, group, label)
        self.experiment = prepare_experiment(self.project, self.root / 'mirrored', mirror_training=True,
                                             index_middle_tip_weight=4)

    def test_actual_fit_uses_only_original_and_mirrored_training_parents_validation_stays_original(self):
        _, _, plan, records = load_experiment(self.experiment)
        table, columns = fitting._training_data(plan, records, 'shin2021_combined')
        original_factory, observed = fitting._estimator_for, []
        def observer(*args):
            estimator = original_factory(*args)
            original_fit = estimator.fit
            stage = len(observed)
            def fit(matrix, labels):
                source_ids = plan['folds'][stage]['train_ids'] if stage < 3 else plan['development_ids']
                ids = source_ids + [identity + MIRROR_SUFFIX for identity in source_ids]
                np.testing.assert_array_equal(matrix, table.loc[ids, columns].to_numpy())
                np.testing.assert_array_equal(labels, table.loc[ids].label.to_numpy())
                observed.append(ids)
                try: return original_fit(matrix, labels)
                finally: del estimator.fit
            estimator.fit = fit
            return estimator
        with patch.object(fitting, '_estimator_for', side_effect=observer):
            folder = fitting.train_variant(self.experiment, 'shin2021_combined')
        self.assertEqual([len(ids) for ids in observed], [20, 20, 20, 30])
        predicted = pd.read_csv(folder / 'cv_predictions.csv')
        self.assertEqual(set(predicted.id), set(plan['development_ids']))
        self.assertEqual(len(predicted), 15)
        self.assertFalse(set(table.id) & set(plan['test_ids']))
        self.assertFalse(set(table.id) & {identity + MIRROR_SUFFIX for identity in plan['test_ids']})
        self._check_saved_model(folder, plan, records)

    def _check_saved_model(self, folder, plan, records):
        bundle = joblib.load(folder / 'model.joblib')
        validate_model_input(bundle)
        self.assertEqual(set(bundle['estimator'].classes_), set('UVRJZ'))
        info = read_json(folder / 'training.json')
        self.assertTrue(info['mirror_training'])
        self.assertEqual((info['n_train'], info['n_training_rows'], info['n_mirrored_training_rows']), (15, 30, 15))
        provenance = pd.read_csv(folder / 'training_row_provenance.csv')
        self.assertEqual(len(provenance), 30)
        self.assertEqual(set(provenance.original_id), set(plan['development_ids']))
        registration = register_model(folder / 'model.joblib', self.experiment / 'model.json', self.experiment, 'ASL')
        handle = ModelHandle.load(registration)
        source = records[plan['development_ids'][0]]
        for actual in (source, mirror_sample(source)):
            values = model_features(bundle, actual)
            expected = bundle['estimator'].predict([[values[c] for c in bundle['config']['columns']]])[0]
            self.assertEqual(handle.predict(actual, 32, 32), expected)
        self.assertTrue(handle.metadata['mirror_training'])

    def test_scaler_fit_counts_include_augmented_train_only(self):
        observed, original = [], StandardScaler.fit
        def fit(scaler, matrix, *args, **kwargs):
            observed.append(len(matrix))
            return original(scaler, matrix, *args, **kwargs)
        with patch.object(StandardScaler, 'fit', fit):
            fitting.train_variant(self.experiment, 'shin2021_with_finger_features')
        self.assertEqual(observed, [20, 20, 20, 30])

    def test_mirrored_and_control_experiments_preserve_sources_and_share_folds(self):
        before = {p: sha256(p) for p in self.project.rglob('*') if p.is_file()}
        control = prepare_experiment(self.project, self.root / 'control', split_from=self.experiment,
                                     index_middle_tip_weight=4)
        _, _, a, _ = load_experiment(self.experiment)
        _, _, b, _ = load_experiment(control)
        self.assertEqual(a['folds'], b['folds'])
        self.assertFalse(b['mirror_training'])
        fitting.train_variant(self.experiment, 'shin2021_combined')
        fitting.train_variant(control, 'shin2021_combined')
        audit = run_audit([self.experiment], self.root / 'audit')
        self.assertNotIn('failed', read_json(audit / 'audit.json')['statuses'])
        replay = reproduce_variant(self.experiment, 'shin2021_combined', self.root / 'replay')
        self.assertEqual(read_json(replay / 'audit.json')['statuses'], {'passed': 3})
        self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))

    def test_development_data_refit_preserves_the_selected_mirror_training_policy(self):
        folder = fitting.train_variant(self.experiment, 'shin2021_combined')
        baseline = register_model(folder / 'model.joblib', self.root / 'baseline.json', self.experiment, 'ASL')
        before = sha256(folder / 'model.joblib')
        additions = self.root / 'additions'
        for label in 'UVRJZ': _photo(additions, 4, label)
        result = train_improvement(baseline, additions, self.root / 'new_data', self.root / 'improved',
            'SYNTHETIC SOFTWARE TEST', 'Add synthetic development rows', 'Software contract only', all_development=True)
        info = read_json(result / 'shin2021_combined/training.json')
        self.assertEqual((info['n_train'], info['n_training_rows']), (20, 40))
        self.assertTrue(info['mirror_training'])
        bundle = joblib.load(result / 'shin2021_combined/model.joblib')
        self.assertEqual(bundle['config']['n_training_rows'], 40)
        self.assertEqual(bundle['config']['index_middle_tip_weight'], 4)
        self.assertEqual(sha256(folder / 'model.joblib'), before)


if __name__ == '__main__':
    unittest.main()
