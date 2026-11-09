"""Check detector CLI configuration and rejected-image reporting."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.dataset import main
from research.detection_report import report


class DetectionWorkflowTests(unittest.TestCase):
    def test_cli_passes_custom_thresholds_and_explicit_reprocessing(self):
        with patch('sys.argv', ['dataset', 'analyze', '--project', 'trial',
                               '--detection-confidence', '0.3', '--presence-confidence', '0.4', '--reprocess']), \
                patch('research.dataset.analyze_project') as analyze:
            main()
        analyze.assert_called_once_with(Path('trial'), checkpoint_every=1000, cache_features=False,
                                        min_hand_detection_confidence=.3,
                                        min_hand_presence_confidence=.4, reprocess=True)

    def test_report_excludes_holdout_and_pending_from_rejection_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = [dict(id=str(i), label='A', status=status, included=True,
                            role=role, filename=f'A{i}.jpg', image_path=f'images/{i}.png',
                            original_path=f'originals/{i}.jpg')
                       for i, (status, role) in enumerate([
                           ('ok', 'development'), ('no_hand', 'development'),
                           ('pending', 'development'), ('multiple_hands', 'test')])]
            (root / 'manifest.json').write_text(json.dumps({'schema': 2, 'samples': samples}))
            report(root, root / 'report')
            with (root / 'report/rejected_images.csv').open() as handle:
                rejected = list(csv.DictReader(handle))
            self.assertEqual([s['id'] for s in rejected], ['1'])
            self.assertTrue(Path(rejected[0]['image_path']).is_absolute())
            with (root / 'report/by_letter.csv').open() as handle:
                counts = list(csv.DictReader(handle))
            self.assertEqual(counts[0]['total'], '3')
            self.assertEqual(counts[0]['pending'], '1')
            self.assertAlmostEqual(float(counts[0]['accepted_percent']), 100 / 3)
