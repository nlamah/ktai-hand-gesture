"""Paper counts and paired unchanged-PNG control, including resume safety."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from RQ2.paper_analysis import summarize, output_path, digest
from RQ2 import png_control as control


def paired(key, statuses, letter='A'):
    return dict(id=key, letter=letter, **dict(zip(control_statuses(), statuses)),
                recovered_variants=';'.join(v for v, s in zip(control_statuses(), statuses) if s == 'ok'))


def control_statuses():
    return ('erosie', 'dilatatie', 'canny_overlay')


class PaperAnalysisTests(unittest.TestCase):
    def test_overlap_and_rejection_counts_distinguish_images_and_attempts(self):
        pairs = [paired('1', ['ok', 'no_hand', 'ok']), paired('2', ['ok', 'ok', 'no_hand']),
                 paired('3', ['multiple_hands', 'invalid_geometry', 'no_hand']),
                 paired('4', ['no_hand'] * 3)]
        base = [dict(id=str(i), letter='A', status='no_hand') for i in range(1, 5)]
        base.append(dict(id='5', letter='B', status='ok'))
        summary, letters, combinations, other = summarize(base, pairs)
        self.assertEqual(summary['additional_without_dilation'], 1)
        self.assertEqual(summary['other_rejected_images'], 1)
        self.assertEqual(summary['other_rejected_images_by_status'], {'multiple_hands': 1, 'invalid_geometry': 1})
        self.assertEqual(sum(summary['other_rejected_retries_by_status'].values()), 3)
        self.assertEqual(combinations[0]['images'], 1)
        self.assertEqual(letters[0]['recovery_percent'], 50)
        self.assertIsNone(letters[1]['recovery_percent'])

    def test_invalid_export_pairs_are_rejected(self):
        baseline = [dict(id='1', letter='A', status='no_hand')]
        row = paired('1', ['no_hand'] * 3)
        for pairs in [[], [row, row], [dict(row, recovered_variants='erosie')],
                      [dict(row, erosie='unexpected')]]:
            with self.assertRaises(ValueError):
                summarize(baseline, pairs)

    def test_output_cannot_modify_source_or_its_parent(self):
        for output in ['example/run', 'example/run/report', 'example']:
            with self.assertRaises(ValueError):
                output_path('example/run', output)


class PNGControlTests(unittest.TestCase):
    def test_real_png_roundtrip_and_nonzero_recovery_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            source = root / 'A1.jpg'
            Image.new('RGB', (16, 16), (20, 40, 80)).save(source)
            row = dict(id='one', filename='A1.jpg', letter='A', original_path=str(source), source_sha256=digest(source))
            from unittest.mock import Mock
            detector = Mock()
            detector.analyze.side_effect = [{'status': 'no_hand'}, {'status': 'ok', 'n_hands': 1}]
            result = control.check_image(row, root, detector)
            self.assertTrue(result['pixels_equal'])
            self.assertEqual(result['png_status'], 'ok')
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                control.check_image(row, root, detector)

    def test_resume_skips_detection_and_rejects_changed_contract_or_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'run'; run.mkdir(); output = root / 'control'
            source = root / 'A1.jpg'; Image.new('RGB', (16, 16)).save(source)
            settings = dict(detection_confidence=.5, presence_confidence=.5)
            (run / 'settings.json').write_text(json.dumps(settings))
            for name in ['baseline.csv', 'no_hand.csv']:
                (run / name).write_text('fixture')
            row = dict(id='one', filename='A1.jpg', letter='A', original_path=str(source), source_sha256=digest(source))
            with patch.object(control, 'verify'), patch.object(control, 'read_csv', return_value=[row]), \
                    patch.object(control, 'environment', return_value={'test': True}), patch.object(control, 'Detector') as detector:
                detector.return_value.analyze.return_value = {'status': 'no_hand'}
                result = control.control(run, output)
                self.assertEqual(result['png_recoveries_vs_historical'], 0)
                self.assertEqual(detector.return_value.analyze.call_count, 2)
                detector.reset_mock()
                self.assertEqual(control.control(run, output, resume=True), result)
                detector.assert_not_called()
                with self.assertRaises(FileExistsError):
                    control.control(run, output)
                with self.assertRaises(ValueError):
                    control.control(run, output, limit=1, resume=True)
                (output / 'images/one.png').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    control.control(run, output, resume=True)
                self.assertFalse((output / 'control.lock').exists())

    def test_interrupted_control_retains_completed_rows_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'run'; run.mkdir(); output = root / 'control'
            (run / 'settings.json').write_text(json.dumps(dict(detection_confidence=.5, presence_confidence=.5)))
            for name in ['baseline.csv', 'no_hand.csv']:
                (run / name).write_text('fixture')
            rows = []
            for index in range(2):
                source = root / f'A{index+1}.jpg'
                Image.new('RGB', (16, 16), (index, 0, 0)).save(source)
                rows.append(dict(id=str(index), filename=source.name, letter='A',
                                 original_path=str(source), source_sha256=digest(source)))
            with patch.object(control, 'verify'), patch.object(control, 'read_csv', return_value=rows), \
                    patch.object(control, 'environment', return_value={'test': True}), patch.object(control, 'Detector') as detector:
                detector.return_value.analyze.side_effect = [{'status': 'no_hand'}, {'status': 'no_hand'}, RuntimeError('interrupted')]
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    control.control(run, output)
                self.assertFalse((output / 'control.lock').exists())
                self.assertFalse((output / 'summary.json').exists())
                detector.return_value.close.assert_called_once()
                detector.reset_mock()
                detector.return_value.analyze.side_effect = None
                detector.return_value.analyze.return_value = {'status': 'no_hand'}
                result = control.control(run, output, resume=True)
                self.assertEqual(result['images'], 2)
                self.assertEqual(detector.return_value.analyze.call_count, 2)

    def test_version_mismatch_is_not_silently_accepted(self):
        with patch.object(control, 'model_path', return_value=Path(__file__)):
            with self.assertRaisesRegex(ValueError, 'mediapipe'):
                control.environment(dict(mediapipe='wrong'))


if __name__ == '__main__':
    unittest.main()
