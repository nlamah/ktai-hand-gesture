"""Protocol coverage, resumability, model locking and fixed annotation."""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.protocol import ProtocolSession, tasks
from research.live_camera import confirm_expected


class ProtocolTests(unittest.TestCase):
    def write_capture(self, session, task, name):
        directory = session.directory / name
        directory.mkdir()
        (directory/'capture.json').write_text(json.dumps({'protocol':{'session_id':session.plan['id'],'role':session.plan['role'],'task':task},'expected_letter':task['letter'],'model':{'sha256':session.plan['model_sha256']}}))

    def test_balanced_27_attempts_cover_controls_and_tilts(self):
        plan=tasks()
        self.assertEqual(len({t['id'] for t in plan}),27)
        self.assertEqual(Counter(t['letter'] for t in plan),{'C':9,'D':9,'O':9})
        self.assertTrue(all(v==3 for v in Counter((t['letter'],t['tilt']) for t in plan).values()))

    def test_resume_uses_finalized_captures_even_without_progress_update(self):
        with tempfile.TemporaryDirectory() as root:
            s=ProtocolSession.create(root,'development','model1')
            first=s.current
            self.write_capture(s,first,'one')
            resumed=ProtocolSession(s.directory,'model1')
            self.assertEqual(len(resumed.completed),1)
            self.assertNotEqual(resumed.current['id'],first['id'])
            self.assertEqual(resumed.metadata()['role'],'development')
            self.assertTrue(resumed.metadata()['not_training_data'])

    def test_model_is_locked_and_roles_have_separate_directories(self):
        with tempfile.TemporaryDirectory() as root:
            a=ProtocolSession.create(root,'development','model1')
            b=ProtocolSession.create(root,'evaluation','model1')
            self.assertNotEqual(a.directory,b.directory)
            with self.assertRaises(ValueError):ProtocolSession(b.directory,'model2')
            self.assertEqual(ProtocolSession(b.directory,'model1').plan['role'],'evaluation')

    def test_duplicate_attempts_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            s=ProtocolSession.create(root,'evaluation','model1')
            self.write_capture(s,s.current,'one');self.write_capture(s,s.current,'two')
            with self.assertRaisesRegex(ValueError,'Duplicate'):s.refresh()

    def test_complete_session_has_no_next_task(self):
        with tempfile.TemporaryDirectory() as root:
            s=ProtocolSession.create(root,'evaluation','model1')
            for i,task in enumerate(s.plan['tasks']):self.write_capture(s,task,str(i))
            s.refresh();self.assertIsNone(s.current)
            self.assertEqual(len(s.completed),27)

    def test_guided_confirmation_cannot_change_target_letter(self):
        with ExitStack() as stack:
            stack.enter_context(patch('research.live_camera.cv2.imshow'))
            stack.enter_context(patch('research.live_camera.cv2.getWindowProperty',return_value=1))
            stack.enter_context(patch('research.live_camera.cv2.waitKey',side_effect=[ord('D'),8,13]))
            self.assertEqual(confirm_expected(np.zeros((480,640,3),np.uint8),'C',fixed=True),(True,'C'))

    def test_unfinalized_temporary_capture_is_not_counted(self):
        with tempfile.TemporaryDirectory() as root:
            s=ProtocolSession.create(root,'development','model1')
            self.write_capture(s,s.current,'.capture-interrupted')
            s.refresh()
            self.assertEqual(len(s.completed),0)

    def test_guided_camera_records_all_27_attempts_even_without_detections(self):
        from unittest.mock import MagicMock
        from research import live_camera
        capture, detector = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((32,32,3),np.uint8))
        detector.analyze.return_value = {'status':'no_hand'}
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            model = Path(root)/'model.joblib';model.write_bytes(b'fixture')
            stack.enter_context(patch.object(live_camera,'load_bundle',return_value={}))
            stack.enter_context(patch.object(live_camera.cv2,'VideoCapture',return_value=capture))
            stack.enter_context(patch.object(live_camera,'Detector',return_value=detector))
            for name in ('namedWindow','imshow','destroyAllWindows','setMouseCallback'):
                stack.enter_context(patch.object(live_camera.cv2,name))
            stack.enter_context(patch.object(live_camera.cv2,'getWindowProperty',return_value=1))
            stack.enter_context(patch.object(live_camera.cv2,'waitKey',side_effect=[key for _ in range(27) for key in (32,13)]))
            live_camera.run_camera(model,0,Path(root)/'sessions',guided=True,role='evaluation')
            plans=list((Path(root)/'sessions').glob('*/session.json'))
            self.assertEqual(len(plans),1)
            plan=json.loads(plans[0].read_text())
            session=ProtocolSession(plans[0].parent,plan['model_sha256'])
            self.assertIsNone(session.current)
            records=[json.loads(p.read_text()) for p in session.directory.glob('*/capture.json')]
            self.assertEqual(len(records),27)
            self.assertTrue(all(r['predicted_letter'] is None for r in records))
            self.assertTrue(all(r['protocol']['role']=='evaluation' and r['protocol']['prediction_hidden'] for r in records))
            self.assertTrue(all(r['expected_letter']==r['protocol']['task']['letter'] for r in records))
            capture.release.assert_called_once();detector.close.assert_called_once()
