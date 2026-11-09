"""Stable preview sizing, raw model frames, no stale predictions and cleanup."""
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from research import live_camera


class LiveCameraTests(unittest.TestCase):
    def test_preview_size_does_not_change_with_frame_or_letter(self):
        for shape in ((480, 640, 3), (720, 1280, 3), (640, 480, 3)):
            frame = np.full(shape, 80, dtype=np.uint8)
            original = frame.copy()
            for letter in ('I', 'W', None):
                result = live_camera.render_preview(frame, letter)
                self.assertEqual(result.shape, (480, 640, 3))
                self.assertEqual(result.dtype, np.uint8)
            np.testing.assert_array_equal(frame, original)

    def patches(self, stack, capture, detector):
        stack.enter_context(patch.object(live_camera, 'load_bundle', return_value={}))
        stack.enter_context(patch.object(live_camera.cv2, 'VideoCapture', return_value=capture))
        stack.enter_context(patch.object(live_camera, 'Detector', return_value=detector))
        for name in ('namedWindow', 'imshow', 'destroyAllWindows', 'setMouseCallback'):
            stack.enter_context(patch.object(live_camera.cv2, name))
        stack.enter_context(patch.object(live_camera.cv2, 'getWindowProperty', return_value=1))

    def test_unmirrored_rgb_input_and_missing_hand_clear_letter(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :320] = [0, 0, 255]  # red on left in raw BGR input
        frame[:, 320:] = [255, 0, 0]
        capture, detector = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
        detector.analyze.side_effect = [{'status': 'ok'}, {'status': 'no_hand'}]
        with ExitStack() as stack:
            self.patches(stack, capture, detector)
            stack.enter_context(patch.object(live_camera.cv2, 'waitKey', side_effect=[-1, ord('q')]))
            stack.enter_context(patch.object(live_camera, 'predict_letter', side_effect=[{'letter': 'V'}, None]))
            preview = stack.enter_context(patch.object(live_camera, 'render_preview', wraps=live_camera.render_preview))
            live_camera.run_camera()
            self.assertEqual([c.args[1] for c in preview.call_args_list], ['V', None])
        image = detector.analyze.call_args_list[0].args[0]
        self.assertEqual(image.getpixel((10, 200)), (255, 0, 0))
        self.assertEqual(image.getpixel((600, 200)), (0, 0, 255))
        capture.release.assert_called_once()
        detector.close.assert_called_once()

    def test_failed_capture_releases_camera(self):
        capture, detector = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with ExitStack() as stack:
            self.patches(stack, capture, detector)
            with self.assertRaisesRegex(RuntimeError, 'No camera frame'):
                live_camera.run_camera()
        capture.release.assert_called_once()
        detector.close.assert_called_once()

    def test_unavailable_camera_does_not_start_detector(self):
        capture, detector = MagicMock(), MagicMock()
        capture.isOpened.return_value = False
        with ExitStack() as stack:
            self.patches(stack, capture, detector)
            with self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
                live_camera.run_camera()
        capture.release.assert_called_once()
        detector.close.assert_not_called()

    def test_snapshot_saves_the_displayed_frame_before_reading_the_next_one(self):
        first = np.full((480, 640, 3), 20, np.uint8)
        second = np.full((480, 640, 3), 80, np.uint8)
        capture, detector = MagicMock(), MagicMock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, first), (True, second)]
        detector.analyze.return_value = {'status': 'ok'}
        with ExitStack() as stack:
            self.patches(stack, capture, detector)
            stack.enter_context(patch.object(live_camera.cv2, 'waitKey', side_effect=[ord(' '), ord('q')]))
            stack.enter_context(patch.object(live_camera, 'predict_letter', side_effect=[{'letter':'V'}, {'letter':'O'}]))
            stack.enter_context(patch.object(live_camera, 'confirm_expected', return_value=(True,'U')))
            saved = stack.enter_context(patch.object(live_camera, 'save_capture', return_value=Path('/tmp/snapshot')))
            live_camera.run_camera()
            packet, expected = saved.call_args.args[:2]
            np.testing.assert_array_equal(packet['frame'], first)
            self.assertEqual(packet['prediction']['letter'], 'V')
            self.assertEqual(expected, 'U')
        capture.release.assert_called_once()

    def test_landmarks_follow_mirrored_letterboxed_preview_without_changing_raw_frame(self):
        frame=np.zeros((640,480,3),np.uint8);original=frame.copy()
        points=np.tile([.25,.5,0.],(21,1))
        result=live_camera.render_preview(frame,'G',{'status':'ok','image_landmarks':points})
        # Portrait frame occupies x=140..499; x=.25 is mirrored to roughly x=409.
        self.assertGreater(int(result[240,409].sum()),0)
        self.assertEqual(int(result[240,230].sum()),0)
        np.testing.assert_array_equal(frame,original)
        plain=live_camera.render_preview(frame,'G',{'status':'no_hand','image_landmarks':points})
        self.assertEqual(int(plain[240,409].sum()),0)

    def test_new_round_and_fifth_model_keep_correct_snapshot_metadata(self):
        capture,detector=MagicMock(),MagicMock();capture.isOpened.return_value=True
        capture.read.return_value=(True,np.zeros((40,60,3),np.uint8))
        detector.analyze.return_value={'status':'no_hand'}
        with ExitStack() as stack:
            self.patches(stack,capture,detector)
            stack.enter_context(patch.object(live_camera.Path,'read_bytes',return_value=b'model'))
            stack.enter_context(patch.object(live_camera.cv2,'waitKey',side_effect=[32,ord('n'),ord('5'),32,ord('q')]))
            stack.enter_context(patch.object(live_camera,'predict_letter',return_value=None))
            stack.enter_context(patch.object(live_camera,'confirm_expected',return_value=(True,'G')))
            saved=stack.enter_context(patch.object(live_camera,'save_capture',return_value=Path('/tmp/capture')))
            live_camera.run_camera(round_name='GH-rug')
            first,second=[c.args[0] for c in saved.call_args_list]
            self.assertEqual(first['capture_round']['name'],'GH-rug')
            self.assertEqual(first['capture_round']['number'],1)
            self.assertEqual(second['capture_round']['number'],2)
            self.assertNotEqual(first['session_id'],second['session_id'])
            self.assertEqual(second['session_id'],second['capture_round']['id'])
            self.assertEqual(Path(saved.call_args.args[3]),live_camera.MODEL_CHOICES[4][2])
        capture.release.assert_called_once()

    def test_unmirrored_preview_keeps_landmark_and_camera_alignment(self):
        frame=np.zeros((480,640,3),np.uint8);frame[:,:320]=[40,80,120]
        points=np.tile([.25,.5,0.],(21,1))
        actual=live_camera.render_preview(frame,None,{'status':'ok','image_landmarks':points},mirror=False)
        self.assertGreater(int(actual[240,160].sum()),0)
        np.testing.assert_array_equal(actual[200,300],frame[200,300])
        np.testing.assert_array_equal(actual[240,480],frame[240,480])
