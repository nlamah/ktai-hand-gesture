"""Named A-Z sessions, frozen examples, guidance flags and one-key captures."""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import numpy as np
from PIL import Image

from research.alphabet_protocol import AlphabetSession,tasks,profile
from research import live_camera
from research.live_model import DEFAULT_MODEL


class AlphabetProtocolTests(unittest.TestCase):
    def setUp(self):
        # Session behaviour must be testable without the private Kaggle images.
        # Asymmetric pixels exercise both mirrored and unmirrored examples.
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        directory = Path(fixture.name)
        examples = {}
        for index, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            pixels = np.zeros((12, 20, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(20, dtype=np.uint8)
            pixels[:, :, 1] = index * 9
            pixels[:, :, 2] = np.arange(12, dtype=np.uint8)[:, None]
            path = directory / f'{letter}.png'
            Image.fromarray(pixels).save(path)
            mirrored = bool(index % 2)
            examples[letter] = {
                'path': str(path),
                'sha256': sha256(path.read_bytes()).hexdigest(),
                'source_hand': 'left' if mirrored else 'right',
                'display_mirror': mirrored,
                'display_hand': 'right',
            }
        config = directory / 'examples.json'
        config.write_text(json.dumps({'examples': examples}), encoding='utf-8')
        example_config = patch('research.alphabet_protocol.EXAMPLES', config)
        example_config.start()
        self.addCleanup(example_config.stop)

    def test_balanced_plan_and_explicit_participant_kind(self):
        plan=tasks();self.assertEqual(len(plan),130)
        self.assertEqual(Counter(t['letter'] for t in plan),{c:5 for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
        self.assertEqual([t['letter'] for t in plan[:26]],list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.assertEqual(profile(' Zoë ', 'development','beginner','left')['name'],'Zoë')
        with self.assertRaises(ValueError):profile('Participant','','unknown','right')
        with self.assertRaises(ValueError):tasks(0)

    def test_name_role_examples_and_guidance_survive_saved_capture_resume(self):
        with tempfile.TemporaryDirectory() as root:
            sha=sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
            s=AlphabetSession.create(root,profile('Zoë','development','familiar','right'),sha)
            s.toggle_guidance();metadata=s.metadata()
            folder=s.directory/'test';folder.mkdir()
            (folder/'capture.json').write_text(json.dumps({'protocol':metadata,'expected_letter':s.current['letter'],'model':{'sha256':sha}}))
            resumed=AlphabetSession(s.directory,sha)
            self.assertEqual(resumed.plan['participant']['name'],'Zoë')
            self.assertEqual(resumed.plan['role'],'development')
            self.assertEqual(resumed.current['letter'],'B')
            self.assertTrue(metadata['operator_guidance']);self.assertFalse(resumed.assistance)
            self.assertEqual(set(resumed.example_images),set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            with self.assertRaises(ValueError):AlphabetSession(s.directory,'different-model')

    def test_right_hand_reference_policy_is_preserved(self):
        import cv2
        with tempfile.TemporaryDirectory() as root:
            sha=sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
            session=AlphabetSession.create(root,profile('Test','new','unknown','right'),sha)
            self.assertFalse(session.mirror_preview)
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                info=session.plan['examples'][letter]
                self.assertEqual(info['display_hand'],'right')
                source=cv2.imread(info['path'])
                expected=cv2.flip(source,1) if info['display_mirror'] else source
                np.testing.assert_array_equal(session.example_images[letter],expected)

    def test_capture_display_settings_cannot_be_changed_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            sha=sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
            session=AlphabetSession.create(root,profile('Test','new','unknown','right'),sha)
            session.plan['display']['preview_mirrored']=True
            (session.directory/'session.json').write_text(json.dumps(session.plan))
            with self.assertRaisesRegex(ValueError,'display format'):
                AlphabetSession(session.directory,sha)

    def test_single_space_saves_no_hand_attempts_without_confirmation_or_example_in_model_image(self):
        camera,detector=MagicMock(),MagicMock();camera.isOpened.return_value=True
        frame=np.full((32,48,3),21,np.uint8);camera.read.return_value=(True,frame)
        detector.analyze.return_value={'status':'no_hand'}
        with tempfile.TemporaryDirectory() as root,ExitStack() as stack:
            sha=sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
            s=AlphabetSession.create(root,profile('Nieuw','new','unknown','right'),sha,repeats=1)
            stack.enter_context(patch.object(s,'capture_cooldown',0))
            stack.enter_context(patch.object(live_camera,'load_bundle',return_value={}))
            stack.enter_context(patch.object(live_camera.cv2,'VideoCapture',return_value=camera))
            stack.enter_context(patch.object(live_camera,'Detector',return_value=detector))
            for name in ('namedWindow','imshow','destroyAllWindows','setMouseCallback'):
                stack.enter_context(patch.object(live_camera.cv2,name))
            stack.enter_context(patch.object(live_camera.cv2,'getWindowProperty',return_value=1))
            stack.enter_context(patch.object(live_camera.cv2,'waitKey',side_effect=[13,ord('h')]+[32]*26))
            confirm=stack.enter_context(patch.object(live_camera,'confirm_expected'))
            live_camera.run_camera(protocol_session=s)
            confirm.assert_not_called()
            self.assertEqual(len(s.completed),26);self.assertIsNone(s.current)
            records=[json.loads(p.read_text()) for p in s.directory.glob('*/capture.json')]
            self.assertEqual(sum(r['protocol']['operator_guidance'] for r in records),1)
            self.assertTrue(all(r['protocol']['participant']['name']=='Nieuw' and r['protocol']['role']=='evaluation' for r in records))
            self.assertTrue(all(r['predicted_letter'] is None and r['protocol']['prediction_hidden'] for r in records))
            for path in s.directory.glob('*/model_input.png'):
                with Image.open(path) as image:np.testing.assert_array_equal(np.asarray(image),frame)
            from research import evaluate_alphabet
            evaluation=Path(root)/'evaluation'
            with patch('sys.argv',['evaluate_alphabet',str(s.directory),'--out',str(evaluation)]):evaluate_alphabet.main()
            summary=json.loads((evaluation/'summary.json').read_text())
            new=next(g for g in summary['groups'] if g['group']=='new_people_only')
            self.assertEqual(new['images'],26);self.assertEqual(new['accepted_detections'],0)
            self.assertEqual(new['models']['reference']['all_attempts']['accuracy'],0)
            self.assertEqual(new['models']['angles']['accepted_only']['images'],0)
        camera.release.assert_called_once();detector.close.assert_called_once()
