"""Live model selection, both input widths, and exact capture/model alignment."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
import json

import joblib
import numpy as np
import sklearn

from research import live_camera
from research.live_model import (BASELINE_BUNDLE_VERSION,SHIN_VERSION,load_bundle,
                                model_inputs,predict_letter)
from research.capture import save_capture
from research.inputs import LABELS
from research.studies.compare_angles import classifier
from tests.test_live_model import detection


class ModelChoiceTests(unittest.TestCase):
    def test_reference_model_accepts_only_image_landmarks_and_saves_820_values(self):
        rng=np.random.default_rng(42)
        model=classifier('svm','shin',52).fit(rng.normal(size=(52,820)),LABELS*2)
        bundle={'version':BASELINE_BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':[],
                'sklearn_version':sklearn.__version__,'estimator':model}
        d=detection();d.pop('world_landmarks')
        self.assertEqual(model_inputs(bundle,d).shape,(1,820))
        pred=predict_letter(bundle,d);self.assertIn(pred['letter'],LABELS)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(bundle,path);load_bundle(path)
            frame=np.zeros((30,40,3),np.uint8)
            packet={'frame':frame,'model_frame':frame,'detection':d,'prediction':pred,'captured_at_utc':'test','capture_round':{'id':'r1','name':'GH-rug','number':2},'session_id':'r1'}
            saved=save_capture(packet,'C',bundle,path,'hash',frame,Path(temp)/'captures')
            record=json.loads((saved/'capture.json').read_text())
            self.assertEqual(record['input_variant'],'shin')
            self.assertEqual(len(record['raw_model_features']),820)
            self.assertEqual(len(record['preprocessed_model_features']),820)
            self.assertEqual(len(record['feature_names']),820)
            self.assertIsNone(record['joint_angles_degrees'])
            self.assertEqual(record['capture_round'],packet['capture_round'])
            self.assertEqual(record['session_id'],'r1')

    def test_model_buttons_are_clickable_except_during_guided_or_frozen_capture(self):
        rect=live_camera.MODEL_BUTTONS[0][-1]
        state={'shown':object()}
        live_camera.snapshot_mouse(live_camera.cv2.EVENT_LBUTTONDOWN,rect[0]+5,rect[1]+5,0,state)
        self.assertEqual(state['model_request'],'reference')
        for flag in ('model_locked','annotating'):
            state={'shown':object(),flag:True}
            live_camera.snapshot_mouse(live_camera.cv2.EVENT_LBUTTONDOWN,rect[0]+5,rect[1]+5,0,state)
            self.assertNotIn('model_request',state)

    def test_hotkeys_switch_models_in_one_capture_and_snapshot_retains_active_model(self):
        capture,detector=MagicMock(),MagicMock();capture.isOpened.return_value=True
        capture.read.return_value=(True,np.zeros((40,60,3),np.uint8));detector.analyze.return_value={'status':'ok'}
        first={'tag':'angles'};ref={'tag':'reference'};tilt={'tag':'tilt'}
        with ExitStack() as stack:
            loader=stack.enter_context(patch.object(live_camera,'load_bundle',side_effect=[first,ref,tilt]))
            stack.enter_context(patch.object(live_camera.Path,'read_bytes',return_value=b'fixture'))
            stack.enter_context(patch.object(live_camera.cv2,'VideoCapture',return_value=capture))
            stack.enter_context(patch.object(live_camera,'Detector',return_value=detector))
            for name in ('namedWindow','imshow','destroyAllWindows','setMouseCallback'):
                stack.enter_context(patch.object(live_camera.cv2,name))
            stack.enter_context(patch.object(live_camera.cv2,'getWindowProperty',return_value=1))
            stack.enter_context(patch.object(live_camera.cv2,'waitKey',side_effect=[ord('1'),32,ord('3'),ord('q')]))
            predictor=stack.enter_context(patch.object(live_camera,'predict_letter',side_effect=lambda b,d:{'letter':{'angles':'A','reference':'R','tilt':'T'}[b['tag']]}))
            stack.enter_context(patch.object(live_camera,'confirm_expected',return_value=(True,'C')))
            saver=stack.enter_context(patch.object(live_camera,'save_capture',return_value=Path('/tmp/fixture')))
            live_camera.run_camera()
            self.assertEqual([c.args[0]['tag'] for c in predictor.call_args_list],['angles','reference','reference','tilt'])
            self.assertIs(saver.call_args.args[2],ref)
            self.assertEqual(Path(saver.call_args.args[3]),live_camera.MODEL_CHOICES[0][2])
            self.assertEqual(loader.call_count,3)
        capture.release.assert_called_once();detector.close.assert_called_once()

    def test_sixth_model_button_and_new_round_have_separate_hit_regions(self):
        from research.model_choices import visible_model_buttons
        rect=visible_model_buttons()[5][-1]
        state={'shown':object()}
        live_camera.snapshot_mouse(live_camera.cv2.EVENT_LBUTTONDOWN,rect[0]+5,rect[1]+5,0,state)
        self.assertEqual(state['model_request'],'right_hand')
        self.assertNotIn('round_request',state)
        state={'shown':object()};rect=live_camera.ROUND_BUTTON
        live_camera.snapshot_mouse(live_camera.cv2.EVENT_LBUTTONDOWN,rect[0]+5,rect[1]+5,0,state)
        self.assertTrue(state['round_request'])
        self.assertNotIn('model_request',state)
