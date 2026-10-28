"""Weighting must survive normalization, model serialization and live prediction."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.datasets.data import add_image
from gesturelab.features.training_inputs import training_input, model_features
from gesturelab.research.models import inspect_model, ModelHandle, register_model
from gesturelab.training.classifiers import estimator
from gesturelab.training.distance_weighting import TIP_DISTANCE_COLUMNS, validate_tip_weight
from gesturelab.training.experiment_plan import prepare_experiment, load_experiment
from gesturelab.training.experiments import predict_bundle
from gesturelab.training.fitting import train_variant
from gesturelab.validation.report import read_json, sha256


class WeightTransformTests(unittest.TestCase):
    def test_weight_one_preserves_original_estimator_and_predictions(self):
        x = np.array([[1., 1.], [2., 4.], [3., 2.], [4., 5.]])
        for variant in ('shin2021_combined', 'finger_features_only'):
            arguments = dict(variant=variant, columns=[TIP_DISTANCE_COLUMNS[0], 'other'])
            legacy_x = x if variant == 'shin2021_combined' else (x - x.mean(axis=0)) / x.std(axis=0)
            old = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced').fit(legacy_x, list('UVRU'))
            neutral = estimator('svm', **arguments, index_middle_tip_weight=1).fit(x, list('UVRU'))
            np.testing.assert_array_equal(old.predict(legacy_x), neutral.predict(x))
            np.testing.assert_allclose(old.decision_function(legacy_x), neutral.decision_function(x))

    def test_weight_applies_after_standard_scaler_and_other_columns_do_not_change(self):
        x = np.array([[1., 4.], [3., 6.], [2., 5.], [4., 9.]])
        columns = ['geom_tip_distance_index_middle', 'geom_other']
        model = estimator('svm', variant='finger_features_only', columns=columns, index_middle_tip_weight=3)
        model.fit(x, list('UVRU'))
        standardized = (x - x.mean(axis=0)) / x.std(axis=0)
        np.testing.assert_allclose(model[:-1].transform(x), standardized * [3, 1])
        self.assertEqual(model.steps[0][1].n_samples_seen_, 4)
        zero = estimator('svm', variant='finger_features_only', columns=columns, index_middle_tip_weight=0)
        zero.fit(x, list('UVRU'))
        np.testing.assert_allclose(zero[:-1].transform(x), standardized * [0, 1])

    def test_selective_scaler_reordering_keeps_both_distances_attached_to_their_columns(self):
        columns = ['geom_tip_distance_index_middle', 'shin_angle_other', 'shin_dist_08_12', 'geom_other']
        x = np.array([[1., .2, .4, 5.], [2., .3, .5, 6.], [4., .8, .6, 9.]])
        model = estimator('svm', variant='shin2021_with_finger_features', columns=columns, index_middle_tip_weight=2)
        model.fit(x, list('UVR'))
        before = model.steps[0][1].transform(x)
        after = model[:-1].transform(x)
        np.testing.assert_allclose(after, before * [1, 2, 2, 1])
        np.testing.assert_allclose(after[:, 0], x[:, 1])
        np.testing.assert_allclose(after[:, 1], x[:, 2] * 2)

    def test_weighted_model_is_cloneable_and_roundtrips_without_custom_prediction_code(self):
        columns = ['shin_dist_08_12', 'other']
        x = np.array([[.2, .3], [.4, .8], [.1, .7]])
        model = estimator('svm', variant='shin2021_combined', columns=columns, index_middle_tip_weight=2)
        model.fit(x, list('UVR'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.joblib'
            joblib.dump(model, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
        np.testing.assert_allclose(model[:-1].transform(x), loaded[:-1].transform(x))
        np.testing.assert_array_equal(model.predict(x), clone(model).fit(x, list('UVR')).predict(x))

    def test_invalid_weights_unsupported_models_and_missing_distances_are_rejected(self):
        for value in (True, -1, float('nan'), float('inf'), '2'):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_tip_weight(value)
        with self.assertRaisesRegex(ValueError, 'SVM only'):
            estimator('random_forest', index_middle_tip_weight=2)
        with self.assertRaisesRegex(ValueError, 'no explicit'):
            estimator('svm', variant='shin2021_angle', columns=['shin_angle_00_01_cos_x'], index_middle_tip_weight=2)
        with self.assertRaisesRegex(ValueError, 'Column names'):
            estimator('svm', variant='shin2021_combined', index_middle_tip_weight=2)
        self.assertEqual(validate_tip_weight(0), 0.)


def _synthetic_photo(project, group, label, role):
    world = skeleton({'U': 'open_hand', 'V': 'peace', 'R': 'pointing', 'J': 'fist', 'Z': 'thumbs_up'}[label])
    buffer = BytesIO()
    Image.new('RGB', (32, 32), (group * 40, ord(label), 50)).save(buffer, format='PNG')
    return add_image(project, buffer.getvalue(), 'synthetic.png', label, f'session{group}', f'person{group}',
        role=role, provenance={'type': 'synthetic'}, analysis=dict(status='ok',
            image_landmarks=(world * .2 + .5).tolist(), world_landmarks=world.tolist()))[0]


class WeightTrainingTests(unittest.TestCase):
    def test_frozen_weight_reaches_every_fit_and_saved_live_pipeline_without_using_test(self):
        from unittest.mock import patch
        from gesturelab.training import fitting
        with tempfile.TemporaryDirectory() as tmp:
            root, observed = Path(tmp), []
            project = root / 'data'
            for group in range(4):
                for label in 'UVRJZ': _synthetic_photo(project, group, label, 'test' if group == 3 else 'development')
            before = sha256(project / 'manifest.json')
            experiment = prepare_experiment(project, root / 'experiment', index_middle_tip_weight=2)
            original = fitting._estimator_for
            def observe(*args):
                result = original(*args)
                observed.append(result)
                return result
            with patch.object(fitting, '_estimator_for', side_effect=observe):
                folder = train_variant(experiment, 'shin2021_combined')
            self.assertEqual(len(observed), 4)
            for model in observed:
                factors = model.steps[0][1].func.args[0]
                self.assertEqual(np.count_nonzero(factors == 2), 1)
            self._assert_artifacts(folder, experiment, project, before)

    def _assert_artifacts(self, folder, experiment, project, before):
        _, _, plan, records = load_experiment(experiment)
        self.assertEqual(plan['index_middle_tip_weight'], 2)
        info = read_json(folder / 'training.json')
        self.assertEqual(info['index_middle_tip_weight'], 2)
        self.assertEqual(info['tip_distance_columns'], ['shin_dist_08_12'])
        bundle = joblib.load(folder / 'model.joblib')
        self.assertEqual(set(bundle['estimator'].classes_), set('UVRJZ'))
        self.assertFalse(set(pd.read_csv(folder / 'training_features.csv').id) & set(plan['test_ids']))
        for sample_id in plan['development_ids']:
            sample = records[sample_id]
            raw = model_features(bundle, sample)
            self.assertEqual(raw, training_input('shin2021_combined').features(sample, plan['feature_config']))
        sample = records[plan['development_ids'][0]]
        registered = register_model(folder / 'model.joblib', experiment / 'registered.json', experiment, 'ASL')
        handle = ModelHandle.load(registered)
        expected = predict_bundle(bundle, [model_features(bundle, sample)])[0]
        self.assertEqual(handle.predict(sample, 32, 32), expected)
        self.assertEqual(inspect_model(folder / 'model.joblib', experiment)['index_middle_tip_weight'], 2)
        self.assertEqual(sha256(project / 'manifest.json'), before)


if __name__ == '__main__':
    unittest.main()
