"""Balanced G/H collection, preserved task identity, and participant boundaries."""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import numpy as np

from research.gh_view_protocol import GHViewSession,tasks,VIEWS
from research import live_camera
from research.gh_capture_guidance import instructions,render
from research.capture import save_capture


class GHViewProtocolTests(unittest.TestCase):
    def test_balanced_120_tasks_and_80_task_option(self):
        plan=tasks(3)
        self.assertEqual(len(plan),120);self.assertEqual(len({t['id'] for t in plan}),120)
        counts=Counter((t['participant'],t['letter'],t['view']) for t in plan)
        self.assertEqual(len(counts),24);self.assertEqual(set(counts.values()),{5})
        self.assertEqual(Counter(t['split'] for t in plan),{'development':40,'evaluation':80})
        self.assertEqual(len(tasks(2)),80)
        self.assertEqual(len(tasks(1)),40)
        with self.assertRaises(ValueError):tasks(3,['right','left'])
        self.assertEqual({t['requested_hand'] for t in tasks(2,['left','right']) if t['participant']=='P01'},{'left'})

    def test_resume_retains_view_participant_and_all_failed_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            session=GHViewSession.create(root,'fixture',2)
            self.assertEqual(session.metadata()['viewpoint_instruction_version'],'explicit-gh-pose-v3')
            self.assertEqual(session.metadata()['guidance_display_version'],'large-nl-gh-guidance-v1')
            self.assertIn('do not force a fixed left/right',session.plan['instructions'])
            task=session.current
            folder=session.directory/'test';folder.mkdir()
            (folder/'capture.json').write_text(json.dumps({'protocol':session.metadata(),'expected_letter':task['letter'],'model':{'sha256':'fixture'},'predicted_letter':None}))
            other=GHViewSession(session.directory,'fixture')
            self.assertEqual(len(other.completed),1)
            self.assertNotEqual(other.current['id'],task['id'])
            self.assertTrue(other.metadata()['prediction_hidden'])
            self.assertEqual(other.current['participant'],'P01')
            with self.assertRaises(ValueError):GHViewSession(session.directory,'other-model')

    def test_large_guidance_keeps_camera_pixels_and_distinguishes_letter_hand_and_view(self):
        rng=np.random.default_rng(26)
        preview=rng.integers(0,256,size=(480,640,3),dtype=np.uint8)
        for letter in 'GH':
            for hand in ('left','right'):
                for view in VIEWS:
                    task={'participant':'P01','letter':letter,'view':view,'requested_hand':hand,'repeat':1}
                    info=instructions(task)
                    self.assertEqual(info['letter'],letter)
                    self.assertEqual(info['hand'],'LINKERHAND' if hand=='left' else 'RECHTERHAND')
                    self.assertIn('duim',info['finger_shape'])
                    self.assertEqual(info['palm_normal_degrees'],{'handpalm':0,'schuin_palm':45,'schuin_handrug':135,'handrug':180}[view])
                    panel=render(preview,task,0,40)
                    self.assertEqual(panel.shape,(752,1160,3))
                    np.testing.assert_array_equal(panel[:480,:640],preview)

    def test_withdrawn_session_cannot_resume_or_receive_new_capture(self):
        with tempfile.TemporaryDirectory() as root:
            session=GHViewSession.create(root,'fixture',1)
            (session.directory/'withdrawn.json').write_text('{"status":"withdrawn"}')
            with self.assertRaisesRegex(ValueError,'ingetrokken'):
                GHViewSession(session.directory,'fixture')
            with self.assertRaisesRegex(ValueError,'ingetrokken'):
                save_capture({},'G',{},Path('unused-model'),'fixture',None,session.directory)
            self.assertFalse(list(session.directory.glob('*/capture.json')))

    def test_complete_camera_flow_records_120_no_hand_attempts_with_person_confirmation(self):
        camera,detector=MagicMock(),MagicMock();camera.isOpened.return_value=True
        camera.read.return_value=(True,np.zeros((32,32,3),np.uint8));detector.analyze.return_value={'status':'no_hand'}
        with tempfile.TemporaryDirectory() as root,ExitStack() as stack:
            model=Path(root)/'model.joblib';model.write_bytes(b'fixture')
            session=GHViewSession.create(Path(root)/'sessions',sha256(b'fixture').hexdigest())
            stack.enter_context(patch.object(live_camera,'load_bundle',return_value={}))
            stack.enter_context(patch.object(live_camera.cv2,'VideoCapture',return_value=camera))
            stack.enter_context(patch.object(live_camera,'Detector',return_value=detector))
            for name in ('namedWindow','imshow','destroyAllWindows','setMouseCallback'):
                stack.enter_context(patch.object(live_camera.cv2,name))
            stack.enter_context(patch.object(live_camera.cv2,'getWindowProperty',return_value=1))
            keys=[]
            for person in range(3):keys += [32,13]+[key for _ in range(40) for key in (32,13)]
            stack.enter_context(patch.object(live_camera.cv2,'waitKey',side_effect=keys))
            live_camera.run_camera(model,protocol_session=session)
            self.assertIsNone(session.current);self.assertEqual(len(session.completed),120)
            records=[json.loads(p.read_text()) for p in session.directory.glob('*/capture.json')]
            self.assertTrue(all(r['predicted_letter'] is None for r in records))
            self.assertEqual(Counter(r['protocol']['task']['participant'] for r in records),{'P01':40,'P02':40,'P03':40})
            self.assertTrue(all(r['protocol']['prediction_hidden'] for r in records))
            camera.release.assert_called_once();detector.close.assert_called_once()
