"""Large ASL imports preserve holdouts and resume without silently sampling."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gesturelab.datasets.data import analyze_project, read_manifest
from gesturelab.datasets.kaggle import _append_image, import_kaggle_dataset
from scripts.train_kaggle_dataset import KaggleTrainingRun


class KaggleImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.project = self.root / 'archive', self.root / 'dataset'
        self.train = self.source / 'asl_alphabet_train/asl_alphabet_train'
        self.test = self.source / 'asl_alphabet_test/asl_alphabet_test'
        self.test.mkdir(parents=True)
        for index, label in enumerate(('A', 'B')):
            (self.train / label).mkdir(parents=True)
            for number in (1, 2, 3, 10):
                Image.new('RGB', (24, 24), (index * 100, number * 20, 20)).save(self.train / label / f'{label}{number}.jpg')
            Image.new('RGB', (24, 24), (index * 100, 240, 20)).save(self.test / f'{label}_test.jpg')
        (self.train / 'nothing').mkdir()
        Image.new('RGB', (24, 24)).save(self.train / 'nothing/nothing1.jpg')

    def test_all_numbered_photos_import_with_preserved_test_and_block_groups(self):
        result = import_kaggle_dataset(self.source, self.project, block_size=2)
        manifest = read_manifest(self.project)
        samples = manifest['samples']
        self.assertEqual(result, {'added': 10, 'duplicates': 0, 'processed': 10, 'total': 10})
        self.assertTrue(manifest['import']['complete'])
        self.assertEqual({s['label'] for s in samples}, {'A', 'B'})
        self.assertEqual([s['filename'] for s in samples[:4]], ['A1.jpg', 'A2.jpg', 'A3.jpg', 'A10.jpg'])
        self.assertEqual(samples[0]['session'], samples[1]['session'])
        self.assertNotEqual(samples[1]['session'], samples[2]['session'])
        self.assertEqual(samples[0]['session'], samples[4]['session'])
        dev_groups = {s['provenance']['source_group'] for s in samples if s['role'] == 'development'}
        test = [s for s in samples if s['role'] == 'test']
        self.assertEqual(len(test), 2)
        self.assertFalse(dev_groups & {s['provenance']['source_group'] for s in test})
        for sample in samples:
            self.assertEqual((self.project / sample['original_path']).read_bytes(),
                             (self.source / sample['provenance']['source_file']).read_bytes())

    def test_interrupted_import_resumes_at_saved_entry_without_duplicates(self):
        calls = 0
        def interrupt(*args):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError('interrupted')
            return _append_image(*args)
        with patch('gesturelab.datasets.kaggle._append_image', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                import_kaggle_dataset(self.source, self.project, checkpoint_every=5)
        partial = read_manifest(self.project)
        self.assertEqual(partial['import']['processed'], 3)
        self.assertFalse(partial['import']['complete'])
        with self.assertRaisesRegex(ValueError, 'Finish or resume'):
            analyze_project(self.project)
        result = import_kaggle_dataset(self.source, self.project, resume=True)
        self.assertEqual(result['added'], 10)
        self.assertEqual(len({s['id'] for s in read_manifest(self.project)['samples']}), 10)

    def test_resume_rejects_changed_source_or_block_configuration(self):
        import_kaggle_dataset(self.source, self.project)
        with self.assertRaisesRegex(ValueError, 'same Kaggle source'):
            import_kaggle_dataset(self.source, self.project, block_size=2, resume=True)
        Image.new('RGB', (24, 24), 'red').save(self.train / 'A/A1.jpg')
        with self.assertRaisesRegex(ValueError, 'same Kaggle source'):
            import_kaggle_dataset(self.source, self.project, resume=True)

    def test_identical_test_pixels_cannot_be_used_for_training(self):
        shutil.copyfile(self.train / 'A/A1.jpg', self.test / 'A_test.jpg')
        with self.assertRaisesRegex(ValueError, 'cross development and test'):
            import_kaggle_dataset(self.source, self.project)
        manifest = read_manifest(self.project)
        self.assertFalse(manifest['import']['complete'])
        self.assertFalse(any(s['role'] == 'test' for s in manifest['samples']))

    def test_explicit_test_overlap_exclusion_keeps_all_training_and_records_missing_test(self):
        for label in ('A', 'B'):
            shutil.copyfile(self.train / label / f'{label}1.jpg', self.test / f'{label}_test.jpg')
        result = import_kaggle_dataset(self.source, self.project, exclude_overlapping_test=True)
        manifest = read_manifest(self.project)
        self.assertEqual(result, {'added': 8, 'duplicates': 2, 'processed': 10, 'total': 10})
        self.assertTrue(manifest['import']['complete'])
        self.assertEqual(len(manifest['import']['test_overlaps']), 2)
        self.assertIn('No independent final-test', manifest['import']['test_note'])
        self.assertTrue(all(s['role'] == 'development' for s in manifest['samples']))
        with self.assertRaisesRegex(ValueError, 'explicit test-duplicate policy'):
            import_kaggle_dataset(self.source, self.project, resume=True)
        resumed = import_kaggle_dataset(self.source, self.project, resume=True, exclude_overlapping_test=True)
        self.assertEqual(result, resumed)

    def test_existing_dataset_is_not_silently_adopted_or_overwritten(self):
        self.project.mkdir()
        with self.assertRaises(FileExistsError):
            import_kaggle_dataset(self.source, self.project)
        with self.assertRaisesRegex(ValueError, 'not a resumable Kaggle'):
            import_kaggle_dataset(self.source, self.project, resume=True)

    def test_completed_import_can_be_resumed_without_rewriting_images(self):
        import_kaggle_dataset(self.source, self.project)
        before = read_manifest(self.project)
        with patch('gesturelab.datasets.kaggle._append_image') as append:
            import_kaggle_dataset(self.source, self.project, resume=True)
            append.assert_not_called()
        self.assertEqual(read_manifest(self.project), before)

    def test_both_variants_train_and_comparison_does_not_use_final_test(self):
        from argparse import Namespace
        args = Namespace(project=str(self.project), out=str(self.root / 'experiment'), folds=3)
        run = KaggleTrainingRun(args)
        with patch('scripts.train_kaggle_dataset.prepare_experiment') as prepare, \
             patch('scripts.train_kaggle_dataset.load_experiment'), \
             patch('scripts.train_kaggle_dataset.train_variant') as train, \
             patch('scripts.train_kaggle_dataset.compare_variants', return_value=Path('comparison')) as compare:
            run.train_models()
        self.assertEqual(prepare.call_args.kwargs['model'], 'random_forest')
        self.assertEqual(prepare.call_args.kwargs['grouping'], 'source')
        self.assertEqual([call.args[1] for call in train.call_args_list], ['without_features', 'with_features'])
        self.assertFalse(compare.call_args.kwargs.get('final_test', False))


if __name__ == '__main__':
    unittest.main()
