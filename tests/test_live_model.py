"""Model/live schema parity and UI separation of target and prediction."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from research import live_model
from research.geometry import measure_hand_features
from research.inputs import encode_angles, LABELS
from research.shin import combined_features
from research.studies.compare_angles import classifier
from tests.test_asl_categorical_features import synthetic_hand


def detection():
    points = synthetic_hand()
    return {'status': 'ok', 'image_landmarks': (points * .2 + .5).tolist(),
            'world_landmarks': points.tolist(), 'handedness': 'Right'}


class LiveModelTests(unittest.TestCase):
    def test_live_encoding_is_exactly_shin_then_float32_angles(self):
        d = detection()
        expected = np.r_[list(combined_features(d['image_landmarks']).values()),
                         encode_angles(measure_hand_features(d['world_landmarks'], 'Right'))].astype(np.float32)
        np.testing.assert_array_equal(live_model.detection_inputs(d)[0], expected)
        self.assertEqual(live_model.detection_inputs(d).shape, (1, 835))
        self.assertEqual(live_model.detection_inputs(d).dtype, np.float32)

    def test_no_hand_never_produces_a_letter(self):
        self.assertIsNone(live_model.predict_letter({}, {'status': 'no_hand'}))
        self.assertIsNone(live_model.predict_letter({}, {'status': 'multiple_hands'}))
        with self.assertRaises(ValueError):
            live_model.detection_inputs({'status': 'no_hand'})

    def test_bundle_roundtrip_and_schema_rejection(self):
        import sklearn
        rng = np.random.default_rng(42)
        model = classifier('svm', 'shin_angles', 52).fit(rng.normal(size=(52, 835)).astype(np.float32), LABELS * 2)
        bundle = {'version': live_model.BUNDLE_VERSION, 'shin_version': live_model.SHIN_VERSION,
                  'angle_columns': list(live_model.ANGLE_COLUMNS), 'sklearn_version': sklearn.__version__,
                  'estimator': model}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(bundle, path)
            loaded = live_model.load_bundle(path)
            result = live_model.predict_letter(loaded, detection())
            self.assertIn(result['letter'], LABELS)
            self.assertEqual(len(result['alternatives']), 2)
            self.assertNotIn(result['letter'], result['alternatives'])
            bundle['angle_columns'].reverse()
            joblib.dump(bundle, path)
            with self.assertRaises(ValueError):
                live_model.load_bundle(path)
            path.write_bytes(b'broken model')
            with self.assertRaises(ValueError):
                live_model.load_bundle(path)

    def test_camera_needs_no_target_and_target_does_not_control_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            path.touch()
            with patch('research.live_model.DEFAULT_MODEL', path), \
                 patch('research.live_model.load_bundle', return_value={}), \
                 patch('research.live_model.predict_letter', return_value={'letter': 'O', 'alternatives': ['D', 'C']}):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
                self.assertEqual(len(app.exception), 0)
                camera = next(t for t in app.toggle if t.label == 'Webcam on')
                self.assertFalse(camera.disabled)
                app.session_state['asl_live_feature_last_detection'] = detection()
                camera.set_value(True).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(app.metric[0].value, 'O')
                app.text_input[0].set_value('D').run()
                self.assertEqual(app.metric[0].value, 'O')
                app.session_state['asl_live_feature_last_detection'] = {'status': 'no_hand'}
                app.run()
                self.assertEqual(app.metric[0].value, '—')
                self.assertEqual(len(app.exception), 0)
                next(t for t in app.toggle if t.label == 'Webcam on').set_value(False).run()
                self.assertEqual(app.metric[0].value, '—')
