"""Relocating an RQ2 checkout preserves exports, resume and gallery links."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from PIL import Image

from experiments.hand_detection_recovery import path_utils
from research import recovery_experiment as recovery


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('experiments/hand_detection_recovery')


def connection(run):
    run.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(run / 'progress.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS records (stage TEXT, key TEXT, value TEXT, PRIMARY KEY(stage,key))')
    return db


class RecoveryPathTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'data/asl-alphabet-download/asl_alphabet_train/N/N47.jpg'
        source.parent.mkdir(parents=True)
        Image.new('RGB', (32, 32), (60, 90, 120)).save(source)
        run = root / EXPERIMENT / 'runs/check with space'
        for name in ('data.py', 'morphology_pilot.py', 'recovery_experiment.py'):
            (root / 'research').mkdir(exist_ok=True)
            shutil.copy2(ROOT / 'research' / name, root / 'research' / name)
        for name in ('path_utils.py', 'verify_results.py', 'analyze.py', 'plot_results.py', 'gallery_template.html'):
            (root / EXPERIMENT).mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / EXPERIMENT / name, root / EXPERIMENT / name)
        model = root / 'models/test.task'
        model.parent.mkdir()
        model.write_bytes(b'model-fixture')
        args = SimpleNamespace(source=root / 'data/asl-alphabet-download', limit=1)
        with patch.object(path_utils, 'ROOT', root), patch.object(recovery, 'ROOT', root), \
                patch.object(recovery, 'model_path', return_value=model), \
                patch.object(recovery, 'Detector') as detector, redirect_stdout(StringIO()):
            detector.return_value.analyze.side_effect = lambda image: {'status': 'no_hand', 'n_hands': 0}
            config, entries = recovery.configuration(args)
            db = connection(run)
            try:
                recovery.save(db, 'state', 'config', config)
                (run / 'settings.json').write_text(json.dumps(config), encoding='utf-8')
                recovery.baseline(db, run, entries)
                recovery.preprocess(db, run)
                recovery.retry(db, run)
            finally:
                db.close()
        return run, config

    def test_new_run_can_resume_after_copying_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / 'first checkout'
            run, expected_config = self.fixture(first)
            second = Path(temporary) / 'colleague checkout'
            shutil.copytree(first, second)
            moved = second / run.relative_to(first)
            with patch.object(path_utils, 'ROOT', second), patch.object(recovery, 'ROOT', second), \
                    patch.object(recovery, 'model_path', return_value=second / 'models/test.task'), \
                    patch.object(recovery, 'Detector') as detector, redirect_stdout(StringIO()):
                detector.return_value.analyze.side_effect = AssertionError('Completed detection must not rerun')
                config, entries = recovery.configuration(SimpleNamespace(
                    source=second / 'data/asl-alphabet-download', limit=1))
                self.assertEqual(config, expected_config)
                db = connection(moved)
                try:
                    self.assertEqual(recovery.records(db, 'state')['config'], config)
                    recovery.baseline(db, moved, entries)
                    recovery.preprocess(db, moved)
                    recovery.retry(db, moved)
                finally:
                    db.close()
                detector.return_value.analyze.assert_not_called()
            for name in ('settings.json', 'baseline.csv', 'processed.csv', 'retry_landmarks.jsonl'):
                text = (moved / name).read_text(encoding='utf-8-sig')
                self.assertNotIn(str(first), text)
                self.assertNotIn(str(second), text)
                self.assertNotIn('\\\\', text)

    def test_gallery_links_work_in_relocated_checkout_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / 'first checkout'
            run, _ = self.fixture(first)
            second = Path(temporary) / 'colleague checkout'
            shutil.copytree(first, second)
            moved = second / run.relative_to(first)
            result = subprocess.run([sys.executable, str(second / EXPERIMENT / 'analyze.py'),
                                     '--run', str(moved)], cwd=temporary,
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            html = (moved / 'review_gallery.html').read_text(encoding='utf-8')
            self.assertNotIn('file:///', html)
            urls = re.findall(r'"(?:original|url)": "([^"]+)"', html)
            self.assertEqual(len(urls), 4)
            for url in urls:
                self.assertFalse(Path(url).is_absolute())
                self.assertTrue((moved / unquote(url)).is_file(), url)

    def test_gallery_url_escapes_spaces_and_fragment_characters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(path_utils, 'ROOT', root):
                url = path_utils.gallery_url('data/example #1.png', root / 'reports')
                self.assertEqual(url, '../data/example%20%231.png')
                self.assertEqual(path_utils.resolve_path('data/example #1.png'), root.resolve() / 'data/example #1.png')

    def test_n47_figures_read_images_from_relocated_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / 'first checkout'
            run, _ = self.fixture(first)
            second = Path(temporary) / 'colleague checkout'
            shutil.copytree(first, second)
            moved = second / run.relative_to(first)
            result = subprocess.run([sys.executable, str(second / EXPERIMENT / 'plot_results.py'),
                                     '--run', str(moved)], cwd=temporary,
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((moved / 'figures/n47_comparison.png').is_file())
            info = json.loads((moved / 'figures/figure_sources.json').read_text(encoding='utf-8'))
            self.assertEqual(info['source_run'], moved.relative_to(second).as_posix())


if __name__ == '__main__':
    unittest.main()
