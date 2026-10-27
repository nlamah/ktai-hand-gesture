"""Public training contracts: held-out isolation and safe result persistence."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.training import fitting as training_workflow
from gesturelab.training.comparison import compare_variants
from gesturelab.training.experiment_plan import prepare_experiment, load_experiment
from gesturelab.datasets.data import add_image, read_manifest, atomic_json
from gesturelab.features.inspection import extract_features, FEATURE_VERSION


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'dataset'
        for session in range(4):
            for index, label in enumerate(('fist', 'open_hand')):
                image = BytesIO()
                Image.new('RGB', (32, 32), (session * 40, index * 80, 20)).save(image, format='PNG')
                world = skeleton(label)
                raw = world * .2 + .5
                analysis = {
                    'status': 'ok', 'image_landmarks': raw.tolist(),
                    'world_landmarks': world.tolist(),
                    'features': extract_features(raw, world, 32, 32),
                    'feature_version': FEATURE_VERSION,
                }
                add_image(self.project, image.getvalue(), 'photo.png', label, f'round{session}',
                          role='test' if session == 3 else 'development', analysis=analysis)
        self.experiment = prepare_experiment(
            self.project, self.root / 'experiment', folds=3, model='random_forest')

    def test_random_forest_artifacts_contain_only_development_training_data(self):
        _, _, plan, _ = load_experiment(self.experiment)
        out = training_workflow.train_variant(self.experiment, 'with_features')
        training = pd.read_csv(out / 'training_features.csv', index_col=0)
        predictions = pd.read_csv(out / 'cv_predictions.csv')
        self.assertEqual(training.shape, (6, 207))
        self.assertEqual(set(training.index), set(plan['development_ids']))
        self.assertEqual(set(predictions.id), set(plan['development_ids']))
        self.assertTrue(predictions.id.is_unique)
        self.assertFalse(set(predictions.id) & set(plan['test_ids']))
        bundle = joblib.load(out / 'model.joblib')
        self.assertEqual(bundle['estimator'].n_estimators, 150)
        self.assertEqual(bundle['estimator'].n_features_in_, 207)
        self.assertEqual(set(bundle['estimator'].classes_), {'fist', 'open_hand'})
        before = (out / 'model.joblib').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already run'):
            training_workflow.train_variant(self.experiment, 'with_features')
        self.assertEqual((out / 'model.joblib').read_bytes(), before)

    def test_additional_input_variant_can_train_and_compare_on_the_same_split(self):
        with patch('gesturelab.features.finger_features.compute_finger_features', side_effect=AssertionError('geometry called')):
            baseline = training_workflow.train_variant(self.experiment, 'without_features')
        geometry = training_workflow.train_variant(self.experiment, 'finger_features_only')
        self.assertEqual(joblib.load(baseline / 'model.joblib')['estimator'].n_features_in_, 126)
        self.assertEqual(joblib.load(geometry / 'model.joblib')['estimator'].n_features_in_, 81)
        variants = ('without_features', 'finger_features_only')
        report = compare_variants(self.experiment, variants=variants)
        self.assertEqual(pd.read_csv(report / 'metrics.csv').n_features.tolist(), [126, 81])
        self.assertTrue((report / 'report.html').is_file())
        final = compare_variants(self.experiment, final_test=True, variants=variants)
        with patch('gesturelab.training.comparison._final_test_predictions', side_effect=AssertionError('test repeated')):
            self.assertEqual(compare_variants(self.experiment, final_test=True, variants=variants), final)

    def test_incomplete_test_coverage_requires_explicit_opt_in(self):
        manifest = read_manifest(self.project)
        for sample in manifest['samples']:
            if sample['role'] == 'test' and sample['label'] == 'open_hand':
                sample['included'] = False
        atomic_json(self.project / 'manifest.json', manifest)
        output = self.root / 'incomplete_experiment'
        with self.assertRaisesRegex(ValueError, 'allow-incomplete-test'):
            prepare_experiment(self.project, output, folds=3)
        self.assertFalse(output.exists())
        prepare_experiment(self.project, output, folds=3, allow_incomplete_test=True)
        _, _, plan, _ = load_experiment(output)
        self.assertEqual(plan['missing_test_labels'], ['open_hand'])
        self.assertEqual(set(plan['labels']), {'fist', 'open_hand'})

    def test_dataset_change_during_training_prevents_saving_results(self):
        # Loading checks the real dataset; the training module detects a later change.
        with patch.object(training_workflow, 'fingerprint', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'changed during training'):
                training_workflow.train_variant(self.experiment, 'without_features')
        self.assertFalse((self.experiment / 'without_features').exists())

    def test_import_evaluation_limits_are_frozen_and_visible_in_results(self):
        note = 'Filename blocks are not verified participants or recording sessions.'
        manifest = read_manifest(self.project)
        manifest['import'] = {'grouping_note': note, 'complete': True}
        atomic_json(self.project / 'manifest.json', manifest)
        output = prepare_experiment(self.project, self.root / 'noted_experiment', folds=3,
                                    model='random_forest')
        _, _, plan, _ = load_experiment(output)
        self.assertEqual(plan['evaluation_note'], note)
        for variant in ('without_features', 'with_features'):
            training_workflow.train_variant(output, variant)
        report = compare_variants(output)
        self.assertEqual(json.loads((report / 'result.json').read_text())['evaluation_note'], note)
        self.assertIn(note, (report / 'report.html').read_text())


if __name__ == '__main__':
    unittest.main()
