"""Synthetic software contracts only; these are not ASL performance measurements."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.datasets.data import add_image, read_manifest
from gesturelab.features.inspection import FEATURE_VERSION
from gesturelab.features.training_inputs import model_features
from gesturelab.research.models import register_model, ModelHandle, compatible_models
from gesturelab.research.protocol import create_protocol, freeze_test
from gesturelab.research.recording import save_attempt, CaptureSchedule, validate_capture, annotate_attempt, resume_schedule
from gesturelab.research.live import LivePair, correctness
from gesturelab.research.evaluation import classify_records, score_evaluation, evaluate_test
from gesturelab.research.improvement import train_improvement, select_development
from gesturelab.research.analysis import analyze_development, export_live
from gesturelab.training.experiment_plan import prepare_experiment
from gesturelab.training.fitting import train_variant
from gesturelab.validation.report import read_json, sha256


def detection(label='A', status='ok'):
    world = skeleton({'A': 'open_hand', 'J': 'pointing', 'Z': 'peace'}[label])
    return dict(status=status, n_hands=1 if status == 'ok' else 0, feature_version=FEATURE_VERSION,
                image_landmarks=(world * .2 + .5).tolist(), world_landmarks=world.tolist())


def photo(project, group, label, role='development'):
    data = BytesIO()
    Image.new('RGB', (32, 32), (group * 30, ord(label), 20)).save(data, format='PNG')
    return add_image(project, data.getvalue(), 'synthetic.png', label, f's{group}', f'p{group}',
                     role=role, provenance={'type': 'synthetic'}, analysis=detection(label))[0]


class StubModel:
    def __init__(self, answer='A'):
        self.answer, self.seen = answer, []
        self.metadata = dict(alphabet='ASL', labels=list('AJZ'), feature_version=FEATURE_VERSION,
                             name='SYNTHETIC SOFTWARE TEST', historical_detector_requirements=None)

    def predict(self, shared, width, height):
        self.seen.append((shared, width, height))
        if isinstance(self.answer, Exception): raise self.answer
        return self.answer


class LiveContractTests(unittest.TestCase):
    def test_save_key_with_zero_countdown_saves_the_displayed_frame(self):
        from gesturelab.research.native import _live_loop
        pair = LivePair(StubModel(), StubModel())
        cv2 = Mock(COLOR_RGB2BGR=1)
        cv2.cvtColor.side_effect = lambda image, _: image
        cv2.waitKey.side_effect = [ord('s'), ord('q')]
        detector = Mock(options=None)
        detector.analyze.return_value = detection()
        red, blue = Image.new('RGB', (32, 32), 'red'), Image.new('RGB', (32, 32), 'blue')
        with tempfile.TemporaryDirectory() as tmp, patch('gesturelab.research.native._frame', side_effect=[red, blue]), \
                patch('gesturelab.research.native._live_view', return_value=red):
            _live_loop(cv2, None, detector, pair, 'software test', tmp, 0, False)
            saved = next(Path(tmp).glob('*/frame.png'))
            with Image.open(saved) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(pair.frame.getpixel((0, 0)), (0, 0, 255))

    def test_same_frame_detector_output_no_stale_predictions_and_freeze_save(self):
        a, b = StubModel('A'), StubModel('Z')
        detector = Mock(options=None)
        detector.analyze.side_effect = [detection(), dict(status='no_hand')]
        pair = LivePair(a, b, smoothing=3)
        frame = Image.new('RGB', (32, 20), 'red')
        pair.set_intended('A')
        first = pair.process(frame, detector)
        self.assertEqual(first['category'], 'only_a_correct')
        self.assertIs(a.seen[0][0], b.seen[0][0])
        self.assertNotIn('label', a.seen[0][0])
        pair.frozen = True
        self.assertIs(pair.process(Image.new('RGB', (10, 10)), detector), first)
        self.assertEqual(detector.analyze.call_count, 1)
        with tempfile.TemporaryDirectory() as tmp:
            saved = pair.save(Path(tmp) / 'source')
            self.assertEqual(read_json(saved / 'observation.json')['frame_id'], first['frame_id'])
            self.assertEqual(Image.open(saved / 'frame.png').getpixel((0, 0)), (255, 0, 0))
            with self.assertRaises(FileExistsError): pair.save(Path(tmp) / 'source')
            export_live(Path(tmp) / 'source', Path(tmp) / 'export')
        pair.frozen = False
        failed = pair.process(frame, detector)
        self.assertEqual(failed['raw_predictions'], [None, None])
        self.assertEqual(failed['smoothed_predictions'], [None, None])
        self.assertEqual(len(a.seen), 1)

    def test_agreement_is_not_correctness_and_label_is_not_a_constraint(self):
        self.assertEqual(correctness('Z', 'Z'), 'same_prediction')
        self.assertEqual(correctness('Z', 'Z', 'A'), 'both_incorrect')
        self.assertEqual(correctness('A', 'A', 'A'), 'both_correct')
        self.assertEqual(correctness('Z', 'A', 'A'), 'only_b_correct')
        pair = LivePair(StubModel('Z'), StubModel('Z'))
        pair.set_intended('A')
        result = pair.process(Image.new('RGB', (10, 10)), SimpleNamespace(analyze=lambda _: detection()))
        self.assertEqual(result['raw_predictions'], ['Z', 'Z'])
        self.assertIsNone(result['smoothed_predictions'])
        pair.models[1].metadata['alphabet'] = 'NGT'
        with self.assertRaises(ValueError): compatible_models(*pair.models)

    def test_same_evaluation_rows_outside_stress_and_model_error_stay_wrong(self):
        models = [StubModel('Z'), StubModel(ValueError('invalid feature'))]
        records = [dict(id='one', label='A', condition='front', participant='p', session='s',
                        width=10, height=10, **detection())]
        rows = classify_records(records, models)
        result = score_evaluation(rows, ['A', 'J'], list('AJZ'))
        self.assertIs(models[0].seen[0][0], models[1].seen[0][0])
        self.assertNotIn('label', models[0].seen[0][0])
        self.assertEqual(result['scores']['a']['n'], result['scores']['b']['n'])
        self.assertEqual(result['scores']['a']['accuracy'], 0)
        self.assertEqual(result['scores']['a']['confusion'][0], [0, 0, 1, 0])
        self.assertEqual(result['model_errors']['b'], 1)


class ProtocolTests(unittest.TestCase):
    def test_seeded_150_slots_and_clock_never_waits_for_a_good_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = create_protocol(Path(tmp) / 'p.json', 'ASL', list('AUVJZR'), 'p', 's')
            protocol = read_json(path)
            with self.assertRaises(ValueError): validate_capture(path, Path(tmp) / 'data')
            self.assertEqual(len(protocol['trials']), 150)
            second = read_json(create_protocol(Path(tmp) / 'q.json', 'ASL', list('AUVJZR'), 'p', 's'))
            self.assertEqual(protocol['trials'], second['trials'])
            self.assertTrue(all(r['alphabet'] == 'ASL' for r in protocol['references'].values()))
            schedule = CaptureSchedule(protocol)
            schedule.toggle(0)
            self.assertFalse(schedule.due(2))
            self.assertTrue(schedule.due(3))
            schedule.toggle(3)
            self.assertFalse(schedule.due(100))
            schedule.toggle(100)
            self.assertTrue(schedule.due(100))
            record = save_attempt(Path(tmp) / 'data', Image.new('RGB', (8, 8)), dict(status='no_hand'),
                protocol, schedule.current, sha256(path), {'synthetic': True})
            schedule.saved(record['id'], 100)
            schedule.retake(101)
            retake = save_attempt(Path(tmp) / 'data', Image.new('RGB', (8, 8)), dict(status='multiple_hands'),
                protocol, schedule.current, sha256(path), {'synthetic': True}, schedule.retake_of)
            self.assertNotEqual(record['id'], retake['id'])
            self.assertEqual(retake['provenance']['retake_of'], record['id'])
            self.assertEqual(len(read_manifest(Path(tmp) / 'data')['samples']), 2)
            resumed = resume_schedule(CaptureSchedule(protocol), Path(tmp) / 'data')
            self.assertEqual(resumed.index, 1)
            self.assertTrue(resumed.paused)
            annotate_attempt(Path(tmp) / 'data', retake['id'], 'Human: gesture unclear', True)
            self.assertTrue((Path(tmp) / 'data' / retake['image_path']).exists())


class ResearchIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'original'
        for group in range(3):
            for label in 'AJZ': photo(self.project, group, label)
        self.experiment = prepare_experiment(self.project, self.root / 'baseline', folds=3, allow_no_test=True)
        folder = train_variant(self.experiment, 'shin2021_combined')
        self.registration = register_model(folder / 'model.joblib', self.root / 'a.json',
            self.experiment, 'ASL', 'SYNTHETIC SOFTWARE TEST', True)
        self.protocol = create_protocol(self.root / 'dev.json', 'ASL', list('AJZ'), 'new_person', 'new_session',
            repetitions=1, countdown=0, interval=0, confirm_label_validity=True,
            conditions=[dict(name='left15', nominal_angle=-15)])
        plan = read_json(self.protocol)
        self.additions = self.root / 'additions'
        for i, trial in enumerate(plan['trials']):
            save_attempt(self.additions, Image.new('RGB', (32, 32), (200 + i, 150, 90)),
                detection(trial['label']), plan, trial, sha256(self.protocol), {'synthetic': True})

    def test_actual_pipelines_and_order_are_preserved(self):
        folder = train_variant(self.experiment, 'finger_features_only')
        other = register_model(folder / 'model.joblib', self.root / 'b.json', self.experiment, 'ASL')
        handles = [ModelHandle.load(self.registration), ModelHandle.load(other)]
        self.assertEqual([len(h.bundle['config']['columns']) for h in handles], [820, 81])
        pair = LivePair(*handles)
        result = pair.process(Image.new('RGB', (32, 32)), SimpleNamespace(analyze=lambda _: detection('J')))
        for index, handle in enumerate(handles):
            values = model_features(handle.bundle, dict(width=32, height=32, **detection('J')))
            direct = handle.bundle['estimator'].predict([[values[c] for c in handle.bundle['config']['columns']]])[0]
            self.assertEqual(result['raw_predictions'][index], direct)
        with self.assertRaises(ValueError):
            register_model(folder / 'model.joblib', self.root / 'wrong.json', self.experiment, 'ASL', require_baseline=True)

    def test_targeted_augmentation_and_analysis_preserve_originals_and_reject_test(self):
        before = {str(p): sha256(p) for root in (self.project, self.experiment) for p in root.rglob('*') if p.is_file()}
        report = analyze_development(self.registration, self.additions, self.root / 'analysis')
        self.assertTrue((report / 'report.html').is_file())
        output = train_improvement(self.registration, self.additions, self.root / 'augmented', self.root / 'improved',
            'SYNTHETIC SOFTWARE TEST', 'Add three new synthetic viewpoints', 'Software contract only', all_development=True)
        improved = ModelHandle.load(output / 'registered_model.json')
        baseline = ModelHandle.load(self.registration)
        self.assertEqual(improved.bundle['estimator'].get_params(), baseline.bundle['estimator'].get_params())
        self.assertEqual(improved.metadata['labels'], list('AJZ'))
        self.assertEqual(read_json(output / 'split.json')['test_ids'], [])
        self.assertEqual(len(read_json(output / 'split.json')['development_ids']), 12)
        self.assertTrue(all(sha256(p) == digest for p, digest in before.items()))
        with self.assertRaises(ValueError):
            train_improvement(self.registration, self.additions, self.root / 'augmented', self.root / 'improved',
                'h', 'c', 'e', all_development=True)
        reserved = photo(self.additions, 6, 'A', 'test')
        with self.assertRaises(ValueError): select_development(self.additions, [reserved['id']])

    def test_sealed_saved_test_and_retest_use_same_ids_without_output_restriction(self):
        path = create_protocol(self.root / 'test.json', 'ASL', list('AJ'), 'p_test', 's_test', role='test',
            repetitions=1, confirm_label_validity=True, conditions=[dict(name='front', nominal_angle=0)])
        seal = freeze_test(path, self.registration, self.registration, self.root / 'seal.json', 'SYNTHETIC SOFTWARE TEST')
        plan, project = read_json(path), self.root / 'test_data'
        with self.assertRaises(ValueError): validate_capture(path, project)
        for i, trial in enumerate(plan['trials']):
            save_attempt(project, Image.new('RGB', (32, 32), (240, i, 10)), detection(trial['label']),
                plan, trial, sha256(path), {'synthetic': True})
        result = evaluate_test(seal, project, self.root / 'test_result')
        first = read_json(result / 'result.json')
        self.assertEqual(first['usable'], 2)
        self.assertEqual(first['scores']['a']['predicted_columns'], ['A', 'J', 'Z', '__unknown__'])
        self.assertEqual(first['previous_evaluations'], 0)
        result2 = evaluate_test(seal, project, self.root / 'retest')
        self.assertTrue(read_json(result2 / 'result.json')['previously_viewed_retest'])
        viewed = freeze_test(path, self.registration, self.registration, self.root / 'viewed.json', 'Technical re-evaluation', True)
        evaluate_test(viewed, project, self.root / 'viewed_result')

    def test_test_capture_never_loads_models_or_shows_predictions(self):
        from gesturelab.research.native import run_protocol_capture
        path = create_protocol(self.root / 'test.json', 'ASL', ['A'], 'test_p', 'test_s', role='test',
            repetitions=1, countdown=0, confirm_label_validity=True, conditions=[dict(name='front', nominal_angle=0)])
        seal = freeze_test(path, self.registration, self.registration, self.root / 'seal.json', 'Software test only')
        source = Mock()
        source.isOpened.return_value = True
        source.read.return_value = True, np.zeros((20, 20, 3), dtype=np.uint8)
        cv2 = Mock(COLOR_BGR2RGB=1, COLOR_RGB2BGR=2, WINDOW_NORMAL=0)
        cv2.VideoCapture.return_value = source
        cv2.cvtColor.side_effect = lambda image, _: image
        cv2.waitKey.side_effect = [ord(' '), ord('q')]
        detector = Mock(options=None)
        detector.analyze.return_value = dict(status='no_hand')
        with patch.dict(sys.modules, {'cv2': cv2}), patch('gesturelab.research.native.Detector', return_value=detector), \
                patch.object(ModelHandle, 'load', side_effect=AssertionError('No model access in test capture')):
            run_protocol_capture(path, self.root / 'camera_data', seal)
        saved = read_manifest(self.root / 'camera_data')['samples']
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['status'], 'no_hand')
        self.assertFalse(saved[0]['provenance']['predictions_exposed'])
        source.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
