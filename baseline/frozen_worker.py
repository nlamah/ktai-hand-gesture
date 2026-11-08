"""Run older frozen studies with their exact source revision in a temporary tree.

Current application code and historical protocol JSON are never edited. Only the
one source file whose exact hash is required is substituted inside this worker.
Dataset/result/model directories are shared with the project, so checkpoints
remain resumable. The required source implementation is supplied with the project.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from baseline.check import digest
from baseline.runner import ROOT


PROTOCOLS = {'mirrors': 'results/mirror_01/extraction_protocol.json',
             'gh_rotation': 'results/rotation_GH_y180_01/protocol.json'}


def required_snapshot(study):
    protocol = json.loads((ROOT / PROTOCOLS[study]).read_text())
    expected = protocol['source_sha256']['research/live_model.py']
    snapshot = json.loads((ROOT / 'baseline/frozen_sources/manifest.json').read_text())
    path = ROOT / snapshot['file']
    if snapshot['sha256'] != expected or digest(path) != expected:
        raise ValueError(f'The required source version for {study} is missing; the protocol is unchanged.')
    return path, snapshot


def command_plan(study, action):
    from RQ3.workflows import MIRRORS, GH_ROTATION, module
    if study == 'gh_rotation':
        return GH_ROTATION if action == 'reproduce' else [['-m', 'RQ3.gh_rotation']]
    if action == 'verify':
        return [module('verify_mirror')]
    if action == 'report':
        return [module('report_mirror')]
    commands = MIRRORS
    if len(list((ROOT / 'runs/mirror_01/detections').glob('chunk_*.npz'))) == 78:
        if (ROOT / 'results/mirror_01/extraction_complete.json').exists():
            commands = commands[1:]
    return commands


def run_frozen(study, action):
    source, snapshot = required_snapshot(study)
    commands = command_plan(study, action)
    print(f'{study}: required live encoder {snapshot["sha256"][:12]}', flush=True)
    with tempfile.TemporaryDirectory(prefix='ktai-frozen-study-') as directory:
        workspace = Path(directory)
        # Copy only source, never gigabytes of data or pre-existing model files.
        for package in ['research', 'baseline', 'RQ3']:
            shutil.copytree(ROOT / package, workspace / package,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(source, workspace / 'research/live_model.py')
        assert digest(workspace / 'research/live_model.py') == snapshot['sha256']
        for name in ['data', 'runs', 'results', 'models', 'assets', 'config', 'captures', 'paper']:
            original = ROOT / name
            if original.exists():
                (workspace / name).symlink_to(original, target_is_directory=True)
            else:
                (workspace / name).mkdir()
        for command in commands:
            print(' '.join([sys.executable, *command]), flush=True)
            subprocess.run([sys.executable, *command], cwd=workspace, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=list(PROTOCOLS))
    parser.add_argument('action', choices=['reproduce', 'verify', 'report'])
    args = parser.parse_args()
    run_frozen(args.study, args.action)


if __name__ == '__main__':
    main()
