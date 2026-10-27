"""COCO imports preserve labels and prevent augmentation/test leakage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gesturelab.datasets.coco import import_coco_dataset
from gesturelab.datasets.data import read_manifest
from gesturelab.training.experiments import groups_for, grouped_folds


def _write_split(root, split, number):
    folder = root / split
    folder.mkdir(parents=True)
    images, annotations = [], []
    for index, letter in enumerate(('A', 'B')):
        name = f'{letter}{number}_jpg.rf.original.jpg'
        Image.new('RGB', (32, 32), (number * 60, index * 80, 30)).save(folder / name)
        images.append({'id': index, 'file_name': name})
        annotations.append({'id': index, 'image_id': index, 'category_id': index + 1, 'bbox': [0, 0, 32, 32]})
    document = {'images': images, 'annotations': annotations,
                'categories': [{'id': 0, 'name': 'Letters'}, {'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]}
    (folder / '_annotations.coco.json').write_text(json.dumps(document))


class COCOImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.project = self.root / 'coco', self.root / 'dataset'
        for number, split in enumerate(('train', 'valid', 'test')):
            _write_split(self.source, split, number)

    def edit(self, split, change):
        path = self.source / split / '_annotations.coco.json'
        document = json.loads(path.read_text())
        change(document)
        path.write_text(json.dumps(document))

    def test_categories_splits_and_original_annotations_are_preserved(self):
        result = import_coco_dataset(self.source, self.project)
        samples = read_manifest(self.project)['samples']
        self.assertEqual(result, {'added': 6, 'duplicates': 0, 'total': 6})
        self.assertEqual({s['label'] for s in samples}, {'A', 'B'})
        self.assertEqual(sum(s['role'] == 'test' for s in samples), 2)
        self.assertTrue(all(s['status'] == 'pending' and s['participant'] == 'unknown' for s in samples))
        self.assertEqual({s['provenance']['dataset_split'] for s in samples}, {'train', 'valid', 'test'})
        self.assertFalse((self.project / 'references/ngt_alphabet').exists())
        for split in ('train', 'valid', 'test'):
            self.assertEqual((self.project / f'sources/coco_{split}.json').read_bytes(),
                             (self.source / split / '_annotations.coco.json').read_bytes())

    def test_augmented_versions_stay_in_one_source_group_across_folds(self):
        def add_augmentation(document):
            name = 'A0_jpg.rf.augmented.jpg'
            Image.new('RGB', (32, 32), 'red').save(self.source / 'train' / name)
            document['images'].append({'id': 3, 'file_name': name})
            document['annotations'].append({'id': 3, 'image_id': 3, 'category_id': 1})
        self.edit('train', add_augmentation)
        import_coco_dataset(self.source, self.project)
        frame = pd.DataFrame(read_manifest(self.project)['samples'])
        development = frame[frame.role == 'development'].reset_index(drop=True)
        folds, groups = grouped_folds(development, n_splits=2, grouping='source')
        self.assertEqual(list(groups).count('coco:A0_jpg'), 2)
        for train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
        self.assertFalse(set(groups) & set(groups_for(frame[frame.role == 'test'], 'source')))

    def test_original_source_cannot_cross_development_and_test(self):
        (self.source / 'test/A2_jpg.rf.original.jpg').rename(self.source / 'test/A0_jpg.rf.original.jpg')
        self.edit('test', lambda d: d['images'][0].update(file_name='A0_jpg.rf.original.jpg'))
        with self.assertRaisesRegex(ValueError, 'source image'):
            import_coco_dataset(self.source, self.project)
        self.assertFalse(self.project.exists())

    def test_identical_pixels_cannot_cross_development_and_test(self):
        source = self.source / 'train/A0_jpg.rf.original.jpg'
        (self.source / 'test/A2_jpg.rf.original.jpg').write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Identical pixels'):
            import_coco_dataset(self.source, self.project)
        self.assertFalse(self.project.exists())

    def test_multiple_annotations_do_not_silently_pick_one_letter(self):
        self.edit('train', lambda d: d['annotations'].append({'id': 99, 'image_id': 0, 'category_id': 2}))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            import_coco_dataset(self.source, self.project)
        self.assertFalse(self.project.exists())

    def test_image_paths_cannot_escape_the_annotation_directory(self):
        self.edit('train', lambda d: d['images'][0].update(file_name='../outside.jpg'))
        with self.assertRaisesRegex(ValueError, 'invalid image path'):
            import_coco_dataset(self.source, self.project)
        self.assertFalse(self.project.exists())

    def test_existing_dataset_is_not_overwritten(self):
        self.project.mkdir()
        marker = self.project / 'keep.txt'
        marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            import_coco_dataset(self.source, self.project)
        self.assertEqual(marker.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
