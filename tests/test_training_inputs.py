"""Model input definitions are independent from fitting and dataset splitting."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.features.landmark_features import landmarks_for_sample
from gesturelab.features.training_inputs import (
    TrainingInput, TRAINING_INPUTS, training_input, model_input_metadata,
    model_features, validate_model_input,
)
from gesturelab.training.experiment_plan import code_hashes


class TrainingInputTests(unittest.TestCase):
    def setUp(self):
        world = skeleton('open_hand')
        self.sample = {'image_landmarks': world * .2 + .5,
                       'world_landmarks': world, 'width': 320, 'height': 320}

    def test_baseline_never_calls_finger_feature_calculations(self):
        with patch('gesturelab.features.finger_features.compute_finger_features', side_effect=AssertionError('geometry called')):
            features = training_input('without_features').features(self.sample, None)
        self.assertEqual(len(features), 126)
        self.assertTrue(all(name.startswith(('norm_', 'world_')) for name in features))

    def test_augmented_input_preserves_all_baseline_values(self):
        baseline = training_input('without_features').features(self.sample, None)
        augmented = training_input('with_features').features(self.sample, None)
        geometry = training_input('finger_features_only').features(self.sample, None)
        self.assertEqual(len(augmented), 207)
        self.assertEqual(len(geometry), 81)
        self.assertEqual({name: augmented[name] for name in baseline}, baseline)
        self.assertEqual({name: augmented[name] for name in geometry}, geometry)

    def test_new_registry_entry_can_select_a_different_input_without_training_changes(self):
        definition = TrainingInput('Image coordinates only', (landmarks_for_sample,), ('norm_',), 63)
        with patch.dict(TRAINING_INPUTS, {'image_coordinates_only': definition}):
            features = training_input('image_coordinates_only').features(self.sample, None)
        self.assertEqual(len(features), 63)
        self.assertTrue(all(name.startswith('norm_') for name in features))
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_missing_or_extra_feature_columns_are_rejected(self):
        definition = TrainingInput('One measurement', (lambda sample, thresholds: {'x_a': 1},), ('x_',), 2)
        with self.assertRaisesRegex(ValueError, 'Expected 2 features; found 1'):
            definition.features(self.sample, None)

    def test_overlapping_extractors_cannot_silently_overwrite_measurements(self):
        definition = TrainingInput('Duplicate inputs', (landmarks_for_sample, landmarks_for_sample),
                                   ('norm_',), 63)
        with self.assertRaisesRegex(ValueError, 'duplicate columns'):
            definition.features(self.sample, None)

    def test_input_dependencies_and_provenance_are_owned_by_the_definition(self):
        hashes = code_hashes()
        baseline = model_input_metadata('without_features', None, hashes)
        augmented = model_input_metadata('with_features', None, hashes)
        self.assertIn('features/landmark_features.py', baseline['feature_code_hashes'])
        self.assertNotIn('features/finger_features.py', baseline['feature_code_hashes'])
        self.assertIn('features/finger_features.py', augmented['feature_code_hashes'])
        validate_model_input(baseline)
        baseline['feature_code_hashes']['features/landmark_features.py'] = 'modified'
        with self.assertRaisesRegex(ValueError, 'Model input code changed'):
            validate_model_input(baseline)

    def test_additional_helper_dependency_needs_no_training_code_change(self):
        definition = TrainingInput('Custom selection', (landmarks_for_sample,), ('norm_',), 63,
                                   dependencies=('features/finger_features.py',))
        with patch.dict(TRAINING_INPUTS, {'custom_selection': definition}):
            metadata = model_input_metadata('custom_selection', None, code_hashes())
        self.assertIn('features/finger_features.py', metadata['feature_code_hashes'])

    def test_inference_reconstructs_the_exact_training_input_and_column_order(self):
        for variant in TRAINING_INPUTS:
            with self.subTest(variant=variant):
                expected = training_input(variant).features(self.sample, None)
                bundle = model_input_metadata(variant, None, code_hashes())
                actual = model_features(bundle, self.sample)
                self.assertEqual(actual, expected)
                self.assertEqual(list(actual), list(expected))

    def test_unknown_variant_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'Unknown training input variant'):
            training_input('misspelled')

    def test_hybrid_inputs_are_exact_compositions_of_the_existing_measurements(self):
        paper = training_input('shin2021_combined').features(self.sample, None)
        finger = training_input('finger_features_only').features(self.sample, None)
        landmarks = training_input('without_features').features(self.sample, None)
        hybrid = training_input('shin2021_with_finger_features').features(self.sample, None)
        full = training_input('all_features').features(self.sample, None)
        self.assertEqual(len(hybrid), 901)
        self.assertEqual(len(full), 1027)
        self.assertEqual(hybrid, {**paper, **finger})
        self.assertEqual(full, {**paper, **landmarks, **finger})
        self.assertEqual(list(hybrid)[:820], list(paper))
        self.assertEqual(list(full)[:820], list(paper))
        self.assertFalse(any(name.startswith('raw_') for name in full))

    def test_hybrid_provenance_tracks_every_extractor_it_actually_uses(self):
        hashes = code_hashes()
        hybrid = model_input_metadata('shin2021_with_finger_features', None, hashes)
        full = model_input_metadata('all_features', None, hashes)
        required = {'features/paper_shin2021.py', 'features/finger_features.py'}
        self.assertTrue(required <= hybrid['feature_code_hashes'].keys())
        self.assertNotIn('features/landmark_features.py', hybrid['feature_code_hashes'])
        self.assertTrue(required | {'features/landmark_features.py'} <= full['feature_code_hashes'].keys())


if __name__ == '__main__':
    unittest.main()
