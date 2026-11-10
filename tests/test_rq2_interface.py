"""RQ2 public dispatch and protection of existing run outputs."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from RQ2.experiment import ROOT, main
from RQ2.workflows import commands_for


class RQ2InterfaceTests(unittest.TestCase):
    def test_dry_run_preserves_existing_results_and_never_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentinel = root / 'experiments/hand_detection_recovery/runs/kept/progress.sqlite'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'existing progress')
            with patch('RQ2.experiment.ROOT', root), patch('RQ2.experiment.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['recovery', '--name', 'kept', '--limit', '110', '--dry-run']), 0)
            run.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'existing progress')
            self.assertEqual([p for p in root.rglob('*') if p.is_file()], [sentinel])
            self.assertIn('verify_results.py', output.getvalue())
            self.assertIn('plot_results.py', output.getvalue())

    def test_resume_uses_same_limit_and_runs_from_repo_root(self):
        with patch('RQ2.experiment.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            main(['recovery', '--name', 'kept', '--limit', '110', '--resume'])
        self.assertEqual(run.call_count, 3)
        command = run.call_args_list[0].args[0]
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--limit') + 1], '110')
        for call in run.call_args_list:
            self.assertEqual(call.kwargs['cwd'], ROOT)
            self.assertTrue(call.kwargs['check'])

    def test_verify_and_report_do_not_start_or_clean_a_run(self):
        for action, expected in [('verify', ['verify_results.py']), ('report', ['analyze.py', 'plot_results.py'])]:
            commands = commands_for('recovery', action, name='kept')
            self.assertEqual([Path(c[0]).name for c in commands], expected)
            for command in commands:
                self.assertNotIn('--resume', command)
                self.assertTrue((ROOT / command[0]).is_file())

    def test_failure_stops_following_steps_and_preserves_exit_code(self):
        with patch('RQ2.experiment.subprocess.run', side_effect=subprocess.CalledProcessError(7, ['python'])) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['recovery', '--name', 'failed']), 7)
        self.assertEqual(run.call_count, 1)

    def test_invalid_paths_and_incompatible_flags_are_rejected(self):
        cases = [dict(name='../outside'), dict(name='a/b'), dict(name='C:/outside'),
                 dict(name='ok', limit=0), dict(name='ok', action='verify', resume=True),
                 dict(name='ok', action='report', limit=110), dict(name='ok', reference='other')]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                commands_for('recovery', **options)

    def test_compare_requires_distinct_named_runs(self):
        command = commands_for('compare', name='repeat', reference='first')[0]
        self.assertEqual(command[-1], 'experiments/hand_detection_recovery/runs/repeat')
        self.assertIn('experiments/hand_detection_recovery/runs/first', command)
        for ref in [None, 'repeat', '../outside']:
            with self.assertRaises(ValueError):
                commands_for('compare', name='repeat', reference=ref)

    def test_status_and_list_do_not_run_experiments(self):
        for args in [['list'], ['--status']]:
            with patch('RQ2.experiment.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
            run.assert_not_called()
            self.assertTrue(output.getvalue())
        manifest = json.loads((ROOT / 'RQ2/contribution.json').read_text())
        self.assertEqual(manifest['status'], 'implemented')
        self.assertTrue((ROOT / manifest['documentation']).is_file())
        self.assertTrue((ROOT / 'RQ2/experiment.py').is_file())
        rq1 = json.loads((ROOT / 'RQ1/contribution.json').read_text())
        self.assertNotEqual(rq1['status'], 'implemented')
        self.assertFalse((ROOT / 'RQ1/experiment.py').exists())


if __name__ == '__main__':
    unittest.main()
