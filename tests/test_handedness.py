"""Canonicalization parity, no mutation, train-only fitting and bundle routing."""
import unittest

import numpy as np

from research.handedness import VERSION, fit_canonical, right_hand_detection, right_hand_inputs
from research.live_model import BUNDLE_VERSION, detection_inputs, model_inputs
from research.mirroring import reflected_detection
from research.studies.compare_angles import classifier
from tests.test_live_model import detection


class HandednessTests(unittest.TestCase):
    def test_both_hands_converge_before_features_without_mutating_capture(self):
        right = detection(); left = reflected_detection(right)
        original_left = np.array(left['image_landmarks']).copy()
        canonical = right_hand_detection(left)
        self.assertEqual(canonical['handedness'], 'Right')
        np.testing.assert_allclose(detection_inputs(canonical), detection_inputs(right), atol=2e-6)
        np.testing.assert_array_equal(left['image_landmarks'], original_left)
        self.assertEqual(left['handedness'], 'Left')
        np.testing.assert_allclose(detection_inputs(right_hand_detection(canonical)), detection_inputs(right))

    def test_cached_shortcut_and_live_bundle_recompute_match_for_both_hands(self):
        right = detection(); left = reflected_detection(right)
        for sample in (left, right):
            x = detection_inputs(sample)
            expected = detection_inputs(right_hand_detection(sample))
            np.testing.assert_allclose(right_hand_inputs(x, [sample['handedness']]), expected, atol=2e-5)
            bundle = {'version': BUNDLE_VERSION, 'hand_normalization': VERSION}
            np.testing.assert_array_equal(model_inputs(bundle, sample), expected)
            np.testing.assert_array_equal(model_inputs({'version': BUNDLE_VERSION}, sample), x)

    def test_unknown_hand_or_recipe_is_rejected_instead_of_guessed(self):
        for hand in (None, '', 'Unknown'):
            d = detection(); d['handedness'] = hand
            with self.assertRaises(ValueError):right_hand_detection(d)
            with self.assertRaises(ValueError):right_hand_inputs(detection_inputs(d), [hand])
        with self.assertRaises(ValueError):right_hand_detection({'status': 'no_hand'})
        with self.assertRaises(ValueError):model_inputs({'hand_normalization':'future'}, detection())

    def test_normalized_fit_preserves_bandwidth_priors_and_preprocessing(self):
        rng = np.random.default_rng(127)
        x = rng.normal(size=(18,835)).astype(np.float32); y = np.array(['A']*12+['B']*6)
        baseline = classifier('svm','shin_angles',len(y)).fit(x,y)
        normalized = right_hand_inputs(x, ['Left']*8+['Right']*10)
        candidate = fit_canonical(baseline, normalized, y)
        np.testing.assert_array_equal(candidate[0].transform(x), baseline[0].transform(x))
        self.assertEqual(candidate[-1]._gamma, baseline[-1]._gamma)
        np.testing.assert_array_equal(candidate[-1].class_weight_, baseline[-1].class_weight_)

    def test_native_choice_is_separate_and_fits_above_round_controls(self):
        from research import live_camera
        key, _, _, rect = next(b for b in live_camera.MODEL_BUTTONS if b[0]=='right_hand')
        self.assertEqual(key, 'right_hand')
        state={'shown':object()}
        live_camera.snapshot_mouse(live_camera.cv2.EVENT_LBUTTONDOWN,rect[0]+5,rect[1]+5,0,state)
        self.assertEqual(state['model_request'],key)
        panel=live_camera.camera_panel(np.zeros((480,640,3),np.uint8),model_key=key)
        self.assertLess(rect[3],live_camera.ROUND_BUTTON[1])
        self.assertGreaterEqual(panel.shape[0],live_camera.ROUND_BUTTON[3]+30)


if __name__ == '__main__':unittest.main()
