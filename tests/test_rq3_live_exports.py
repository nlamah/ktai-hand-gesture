"""Live replay exports stay inside the workflow results directory."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from RQ3 import live_examples


class LiveExportTests(unittest.TestCase):
    def test_fixed_replay_exports_without_an_external_document_directory(self):
        study = live_examples.gh_live_diagnostics
        original_path = study.Path
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'results'

            def export():
                out.mkdir()
                study.Path('external/figures/diagnostic.pdf').write_text('figure')
                study.Path('external/table.tex').write_text('table')
                self.assertEqual(study.Path('data/input.npz'), Path('data/input.npz'))

            with patch.object(study, 'OUT', out), patch.object(study, 'main', side_effect=export):
                live_examples.replay_to_results()
            self.assertEqual((out / 'diagnostic.pdf').read_text(), 'figure')
            self.assertEqual((out / 'live_examples_rows.tex').read_text(), 'table')
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ['results'])
        self.assertIs(study.Path, original_path)

    def test_failed_replay_restores_the_original_path_handling(self):
        study = live_examples.gh_live_diagnostics
        original_path = study.Path
        with patch.object(study, 'main', side_effect=ValueError('incomplete inputs')):
            with self.assertRaisesRegex(ValueError, 'incomplete inputs'):
                live_examples.replay_to_results()
        self.assertIs(study.Path, original_path)
