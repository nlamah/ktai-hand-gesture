"""Feature, split, classifier and inference contracts for the paper application."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.camera.runtime import _CameraSession
from gesturelab.datasets.data import add_image, atomic_json, read_manifest
from gesturelab.features.training_inputs import PAPER_COMPARISON, model_features, validate_model_input
from gesturelab.training.classifiers import classifier_parameters, estimator
from gesturelab.training.comparison import compare_variants
from gesturelab.training.experiment_plan import prepare_experiment, load_experiment
from gesturelab.training.fitting import train_variant
from gesturelab.training.predictions import validate_predictions


def _photo(project, group, label, pose, role):
    world = skeleton(pose)
    data = BytesIO()
    Image.new('RGB', (32, 32), (group * 40, ord(label), 20)).save(data, format='PNG')
    return add_image(project, data.getvalue(), 'photo.png', label, f'session{group}',
                     participant=f'person{group}', role=role, provenance={'type': 'synthetic'},
                     analysis={'status': 'ok', 'image_landmarks': (world * .2 + .5).tolist(),
                               'world_landmarks': world.tolist()})[0]


class PaperTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'dataset'
        for group in range(4):
            for label, pose in [('A', 'open_hand'), ('J', 'pointing'), ('Z', 'peace')]:
                _photo(self.project, group, label, pose, 'test' if group == 3 else 'development')
        self.experiment = prepare_experiment(self.project, self.root / 'experiment',
                                             folds=3, grouping='participant', model='svm')

    def test_all_five_inputs_keep_j_z_ids_person_groups_and_live_column_order(self):
        _, _, plan, records = load_experiment(self.experiment)
        self.assertEqual(plan['labels'], ['A', 'J', 'Z'])
        for fold in plan['folds']:
            train = {records[i]['participant'] for i in fold['train_ids']}
            valid = {records[i]['participant'] for i in fold['validation_ids']}
            self.assertFalse(train & valid)
        for variant, count in zip(PAPER_COMPARISON, (126, 207, 190, 630, 820)):
            folder = train_variant(self.experiment, variant)
            bundle = joblib.load(folder / 'model.joblib')
            validate_model_input(bundle)
            sample = records[plan['development_ids'][0]]
            live = model_features(bundle, sample)
            matrix = pd.read_csv(folder / 'training_features.csv', index_col=0)
            self.assertEqual(len(live), count)
            self.assertEqual(list(live), bundle['config']['columns'])
            np.testing.assert_allclose(list(live.values()), matrix.loc[sample['id']].to_numpy(), atol=1e-12)
        report = compare_variants(self.experiment, variants=PAPER_COMPARISON)
        self.assertEqual(pd.read_csv(report / 'metrics.csv').n_features.tolist(), [126, 207, 190, 630, 820])
        self.assertEqual(len(pd.read_csv(report / 'cv_folds.csv')), 15)
        self.assertEqual(len(pd.read_csv(report / 'pairwise_comparisons.csv')), 10)
        self.assertTrue((report / 'detection_counts.csv').exists())

    def test_preprocessing_statistics_are_fit_on_each_fold_training_rows_only(self):
        from sklearn.preprocessing import StandardScaler
        observed = []
        original = StandardScaler.fit
        def record_fit(scaler, data, *args, **kwargs):
            observed.append(len(data))
            return original(scaler, data, *args, **kwargs)
        with patch.object(StandardScaler, 'fit', new=record_fit):
            train_variant(self.experiment, 'without_features')
        self.assertEqual(observed, [6, 6, 6, 9])
        with patch.object(StandardScaler, 'fit', side_effect=AssertionError('paper angles should not be scaled')):
            train_variant(self.experiment, 'shin2021_angle')

    def test_published_presets_and_controlled_project_profile_are_distinct(self):
        for variant, rounds in zip(PAPER_COMPARISON[-3:], (12, 13, 14)):
            parameters = classifier_parameters('lightgbm', variant, 'shin2021_asl')
            self.assertEqual((parameters['num_leaves'], parameters['min_child_samples']), (100, 25))
            self.assertEqual(parameters['n_estimators'], rounds)
        self.assertEqual(classifier_parameters('svm', 'shin2021_angle', 'shin2021_asl')['gamma'], .001)
        self.assertEqual(classifier_parameters('svm', 'shin2021_combined', 'shin2021_asl')['gamma'], 'scale')
        self.assertEqual(classifier_parameters('svm', 'shin2021_distance', 'project')['C'], 10)
        self.assertIsInstance(estimator('svm', variant='without_features'), Pipeline)
        self.assertNotIsInstance(estimator('svm', variant='shin2021_angle'), Pipeline)

    def test_lightgbm_trains_real_models_and_freezes_its_library_version(self):
        experiment = prepare_experiment(self.project, self.root / 'lightgbm', folds=3,
                                        grouping='participant', model='lightgbm', classifier_profile='shin2021_asl')
        _, _, plan, _ = load_experiment(experiment)
        self.assertIn('lightgbm', plan['classifier_versions'])
        for variant, rounds in zip(PAPER_COMPARISON[-3:], (12, 13, 14)):
            folder = train_variant(experiment, variant)
            bundle = joblib.load(folder / 'model.joblib')
            self.assertEqual(bundle['estimator'].n_estimators, rounds)
            self.assertEqual(bundle['input_definition'], 'shin2021-image-table2-cosine-row-zscore-v1')
            self.assertEqual(set(bundle['estimator'].classes_), {'A', 'J', 'Z'})

    def test_valid_model_outputs_are_rejected_if_photo_folds_are_changed(self):
        folder = train_variant(self.experiment, 'shin2021_distance')
        _, _, plan, records = load_experiment(self.experiment)
        predictions = pd.read_csv(folder / 'cv_predictions.csv')
        predictions.loc[0, 'fold'] = (predictions.loc[0, 'fold'] + 1) % 3
        with self.assertRaisesRegex(ValueError, 'frozen validation folds'):
            validate_predictions({'shin2021_distance': predictions}, plan, records)

    def test_degenerate_live_features_clear_the_prediction_without_stopping_camera(self):
        folder = train_variant(self.experiment, 'shin2021_angle')
        session = _CameraSession({'mode': 'predict', 'project': self.project}, SimpleNamespace(), None, None)
        session.bundle = joblib.load(folder / 'model.joblib')
        session.ml_stable = 'A'
        session.stable_ml.update('A')
        session.detector = SimpleNamespace(analyze=lambda frame: {
            'status': 'ok', 'image_landmarks': np.zeros((21, 3)).tolist()})
        session._analyze_frame(Image.new('RGB', (32, 32)), 100)
        self.assertEqual(session.result['status'], 'invalid_features')
        self.assertIsNone(session.ml_stable)
        self.assertIsNone(session.ml)

    def test_valid_paper_model_is_used_by_the_live_camera_inference_path(self):
        folder = train_variant(self.experiment, 'shin2021_combined')
        session = _CameraSession({'mode': 'predict', 'project': self.project}, SimpleNamespace(), None, None)
        session.bundle = joblib.load(folder / 'model.joblib')
        sample = next(s for s in read_manifest(self.project)['samples'] if s['label'] == 'J')
        session.detector = SimpleNamespace(analyze=lambda frame: dict(sample))
        session._analyze_frame(Image.new('RGB', (32, 32)), 100)
        self.assertEqual(session.result['status'], 'ok')
        self.assertEqual(session.ml, 'J')
        self.assertEqual(session.ml_stable, 'J')

    def test_template_reuses_exact_membership_but_freezes_new_code_and_classifier(self):
        original = (self.experiment / 'split.json').read_bytes()
        target = prepare_experiment(self.project, self.root / 'copied', folds=3, grouping='participant',
                                    model='lightgbm', split_from=self.experiment)
        _, _, before, _ = load_experiment(self.experiment)
        _, _, after, _ = load_experiment(target)
        self.assertEqual(before['folds'], after['folds'])
        self.assertEqual(before['development_ids'], after['development_ids'])
        self.assertEqual(before['test_ids'], after['test_ids'])
        self.assertIsNotNone(after['split_parent'])
        self.assertEqual(after['model'], 'lightgbm')
        self.assertEqual((self.experiment / 'split.json').read_bytes(), original)

    def test_template_rejects_changed_data_and_invalid_group_assignments(self):
        plan = json.loads((self.experiment / 'split.json').read_text())
        plan['folds'][0]['train_ids'].append(plan['folds'][0]['validation_ids'][0])
        template = self.root / 'invalid_split.json'
        atomic_json(template, plan)
        with self.assertRaisesRegex(ValueError, 'invalid training/validation'):
            prepare_experiment(self.project, self.root / 'rejected', folds=3,
                               grouping='participant', split_from=template)
        manifest = read_manifest(self.project)
        manifest['samples'][0]['condition'] = 'changed'
        atomic_json(self.project / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'same unchanged dataset'):
            prepare_experiment(self.project, self.root / 'changed', folds=3,
                               grouping='participant', split_from=self.experiment)

    def test_allow_no_test_remains_explicit_and_all_letters_stay_selected(self):
        manifest = read_manifest(self.project)
        for sample in manifest['samples']:
            if sample['role'] == 'test':
                sample['included'] = False
        atomic_json(self.project / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'allow-no-test'):
            prepare_experiment(self.project, self.root / 'no_test', folds=3, grouping='participant')
        target = prepare_experiment(self.project, self.root / 'allowed', folds=3,
                                    grouping='participant', allow_no_test=True)
        _, _, plan, _ = load_experiment(target)
        self.assertEqual(plan['labels'], ['A', 'J', 'Z'])
        self.assertEqual(plan['test_ids'], [])


if __name__ == '__main__':
    unittest.main()
