"""Snapshot fidelity, prediction replay, labels, atomic failures and UI controls."""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from research.capture import save_capture
from research.live_camera import snapshot_mouse, confirm_expected, BUTTON
from research.live_model import detection_inputs, predict_letter
from research.studies.compare_angles import classifier
from research.inputs import LABELS
from tests.test_live_model import detection


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.bundle = {'version': 'test', 'estimator': classifier('svm', 'shin_angles', 52).fit(rng.normal(size=(52,835)), LABELS * 2)}

    def packet(self):
        raw = np.arange(24*36*3,dtype=np.uint8).reshape(24,36,3)
        d = detection()
        return {'frame': raw, 'model_frame': raw[::2,::2].copy(), 'detection': d,
                'prediction': predict_letter(self.bundle,d), 'captured_at_utc': '2026-10-31T12:34:56+00:00', 'camera': 0}

    def test_capture_is_lossless_and_model_prediction_replays_exactly(self):
        p = self.packet()
        expected = 'A' if p['prediction']['letter'] != 'A' else 'B'
        with tempfile.TemporaryDirectory() as temp:
            folder = save_capture(p,expected,self.bundle,Path(temp)/'model.joblib','test-model-hash',p['frame'],temp)
            r = json.loads((folder/'capture.json').read_text())
            self.assertEqual(r['expected_letter'],expected)
            self.assertTrue(r['is_confusion'])
            self.assertEqual(r['frame_captured_at_utc'],p['captured_at_utc'])
            self.assertEqual(r['model']['sha256'],'test-model-hash')
            self.assertNotIn('estimator',r['model']['metadata'])
            for filename,key in [('frame.png','frame'),('model_input.png','model_frame')]:
                np.testing.assert_array_equal(np.asarray(Image.open(folder/filename)),cv2.cvtColor(p[key],cv2.COLOR_BGR2RGB))
                self.assertEqual(sha256((folder/filename).read_bytes()).hexdigest(),r['files'][filename]['sha256'])
            row = np.array(r['raw_model_features'],np.float32).reshape(1,835)
            np.testing.assert_array_equal(row,detection_inputs(p['detection']))
            self.assertEqual(self.bundle['estimator'].predict(row)[0],r['predicted_letter'])
            np.testing.assert_allclose(self.bundle['estimator'].decision_function(row)[0],[r['decision_scores'][c] for c in LABELS])
            self.assertEqual(len(r['joint_angles_degrees']),15)
            self.assertEqual(len(r['feature_names']),835)
            self.assertEqual(len(r['preprocessed_model_features']),835)

    def test_unknown_expected_and_no_hand_are_not_labelled_as_confusions(self):
        p=self.packet();p['detection']={'status':'no_hand'};p['prediction']=None
        with tempfile.TemporaryDirectory() as temp:
            folder=save_capture(p,None,{},'model.joblib','hash',p['frame'],temp)
            r=json.loads((folder/'capture.json').read_text())
            self.assertIsNone(r['is_confusion']);self.assertIsNone(r['predicted_letter'])
            self.assertIsNone(r['raw_model_features'])

    def test_partial_write_failure_leaves_no_half_capture(self):
        p=self.packet()
        with tempfile.TemporaryDirectory() as temp, patch('research.capture.Image.Image.save',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):save_capture(p,'A',self.bundle,'model.joblib','hash',p['frame'],temp)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_invalid_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):save_capture(self.packet(),'AB',self.bundle,'model.joblib','hash',self.packet()['frame'],temp)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_button_uses_displayed_packet_and_ignores_clicks_during_annotation(self):
        state={'shown':object()}
        snapshot_mouse(cv2.EVENT_LBUTTONDOWN,BUTTON[0]+1,BUTTON[1]+1,0,state)
        self.assertIs(state['requested'],state['shown'])
        state['requested']=None;state['annotating']=True
        snapshot_mouse(cv2.EVENT_LBUTTONDOWN,BUTTON[0]+1,BUTTON[1]+1,0,state)
        self.assertIsNone(state['requested'])

    def test_annotation_accepts_q_as_label_and_escape_cancels(self):
        preview=np.zeros((480,640,3),np.uint8)
        with ExitStack() as stack:
            stack.enter_context(patch('research.live_camera.cv2.imshow'))
            stack.enter_context(patch('research.live_camera.cv2.getWindowProperty',return_value=1))
            stack.enter_context(patch('research.live_camera.cv2.waitKey',side_effect=[ord('q'),13,27]))
            self.assertEqual(confirm_expected(preview),(True,'Q'))
            self.assertEqual(confirm_expected(preview),(False,None))
