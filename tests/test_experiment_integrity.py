"""Frozen experiments must reject changed inputs and incomparable model artifacts."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.datasets.data import add_image, atomic_json, read_manifest
from gesturelab.training import experiment_plan
from gesturelab.training.comparison import compare_variants
from gesturelab.features.training_inputs import DEFAULT_VARIANTS


def _add_photo(project, session, label, color, role):
    world = skeleton('open_hand')
    raw = world * .2 + .5
    image = BytesIO()
    Image.new('RGB', (32, 32), color).save(image, format='PNG')
    add_image(project, image.getvalue(), 'photo.png', label, session, role=role,
              analysis={'status': 'ok', 'image_landmarks': raw.tolist(),
                        'world_landmarks': world.tolist()})


class ExperimentIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'dataset'
        for session in range(4):
            for index, label in enumerate(('A', 'B')):
                _add_photo(self.project, f'round{session}', label, (session * 40, index * 80, 20),
                           'test' if session == 3 else 'development')
        self.experiment = experiment_plan.prepare_experiment(self.project, self.root / 'experiment')
        self.plan = json.loads((self.experiment / 'split.json').read_text())

    def _write_model_artifacts(self):
        """Only hashes are needed: integrity rejection must precede deserialization."""
        hashes = {}
        split_hash = experiment_plan.file_hash(self.experiment / 'split.json')
        for variant in DEFAULT_VARIANTS:
            folder = self.experiment / variant
            folder.mkdir(exist_ok=True)
            model = folder / 'model.joblib'
            model.write_bytes(b'Artifact for hash-validation tests, not a serialized model')
            hashes[variant] = experiment_plan.file_hash(model)
            atomic_json(folder / 'training.json', {'split_sha256': split_hash,
                                                  'model_sha256': hashes[variant]})
        return split_hash, hashes

    def test_metadata_changes_invalidate_a_frozen_experiment(self):
        manifest = read_manifest(self.project)
        manifest['samples'][0]['label'] = 'C'
        atomic_json(self.project / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Dataset changed since splitting'):
            experiment_plan.load_experiment(self.experiment)

    def test_changed_source_hash_is_rejected_without_editing_source_files(self):
        hashes = dict(self.plan['code_hashes'])
        hashes['features/landmark_features.py'] = 'changed'
        with patch.object(experiment_plan, 'code_hashes', return_value=hashes):
            with self.assertRaisesRegex(ValueError, 'Feature or experiment code changed'):
                experiment_plan.load_experiment(self.experiment)

    def test_missing_photo_is_rejected_even_if_manifest_is_unchanged(self):
        sample = read_manifest(self.project)['samples'][0]
        (self.project / sample['image_path']).unlink()
        with self.assertRaisesRegex(ValueError, 'Source images are missing'):
            experiment_plan.load_experiment(self.experiment)

    def test_dataset_can_move_without_changing_experiment_contents(self):
        moved = self.root / 'relocated_dataset'
        shutil.move(self.project, moved)
        with self.assertRaisesRegex(ValueError, 'Project not found'):
            experiment_plan.load_experiment(self.experiment)
        _, project, plan, records = experiment_plan.load_experiment(self.experiment, moved)
        self.assertEqual(project, moved.resolve())
        self.assertEqual(plan, self.plan)
        self.assertEqual(set(records), {sample['id'] for sample in read_manifest(moved)['samples']})

    def test_nested_code_snapshots_match_the_frozen_hashes(self):
        self.assertIn('features/training_inputs.py', self.plan['code_hashes'])
        self.assertIn('training/fitting.py', self.plan['code_hashes'])
        for name, expected in self.plan['code_hashes'].items():
            with self.subTest(source=name):
                snapshot = self.experiment / 'code_snapshot' / name
                self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), expected)

    def test_existing_experiment_is_never_overwritten(self):
        before = (self.experiment / 'split.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            experiment_plan.prepare_experiment(self.project, self.experiment, model='random_forest')
        self.assertEqual((self.experiment / 'split.json').read_bytes(), before)

    def test_modified_model_is_rejected_before_deserialization_or_reporting(self):
        self._write_model_artifacts()
        model = self.experiment / DEFAULT_VARIANTS[1] / 'model.joblib'
        model.write_bytes(b'modified artifact')
        with patch('gesturelab.training.comparison.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'A saved model has changed'):
                compare_variants(self.experiment)
            load.assert_not_called()
        self.assertFalse((self.experiment / 'comparison_cv').exists())

    def test_variants_from_different_splits_cannot_be_compared(self):
        self._write_model_artifacts()
        path = self.experiment / DEFAULT_VARIANTS[1] / 'training.json'
        info = json.loads(path.read_text())
        info['split_sha256'] = 'another split'
        atomic_json(path, info)
        with self.assertRaisesRegex(ValueError, 'The variants use different splits'):
            compare_variants(self.experiment)
        self.assertFalse((self.experiment / 'comparison_cv').exists())

    def test_final_test_for_a_different_model_pair_cannot_be_reused(self):
        split_hash, hashes = self._write_model_artifacts()
        hashes[DEFAULT_VARIANTS[0]] = 'a previously evaluated model'
        final = self.experiment / 'final_test'
        final.mkdir()
        atomic_json(final / 'result.json', {'split_sha256': split_hash, 'model_hashes': hashes})
        before = (final / 'result.json').read_bytes()
        with patch('gesturelab.training.comparison.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'A final test exists for different models'):
                compare_variants(self.experiment, final_test=True)
            load.assert_not_called()
        self.assertEqual((final / 'result.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
