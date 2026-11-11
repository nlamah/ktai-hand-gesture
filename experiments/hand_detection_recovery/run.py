"""Run the existing baseline, preprocessing and retry commands in sequence."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]


def prepare_run(run, resume):
    runs = (EXPERIMENT / 'runs').resolve()
    # Path.is_junction exists from Python 3.12; junctions only occur on Windows.
    is_junction = getattr(run, 'is_junction', lambda: False)()
    if run.is_symlink() or is_junction or run.resolve().parent != runs:
        raise ValueError('Run directory must be a direct, ordinary folder inside the experiment runs directory.')
    if run.exists() and not resume:
        print(f'Removing previous results: {run.resolve()}', flush=True)
        shutil.rmtree(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='Run folder name, e.g. full_dataset_01.')
    parser.add_argument('--limit', type=int, help='Optional number of images for a technical check.')
    parser.add_argument('--resume', action='store_true', help='Keep previous results and resume instead of starting clean.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.name):
        parser.error('Use a name containing letters, digits, underscores or hyphens.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    # Keep this lock outside the run folder so cleaning cannot remove the lock.
    lock_dir = EXPERIMENT / '.locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / (args.name + '.lock')).open('a+b') as lock:
        try:
            if sys.platform == 'win32':
                import msvcrt
                if lock.seek(0, 2) == 0:
                    lock.write(b'0')
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            parser.error('A run with this name is already active.')
        return run_steps(args)


def run_steps(args):
    run = EXPERIMENT / 'runs' / args.name
    prepare_run(run, args.resume)
    print(f'Results: {run}', flush=True)
    for index, stage in enumerate(('baseline', 'preprocess', 'retry'), 1):
        command = [sys.executable, '-m', 'research.recovery_experiment',
                   stage, '--run', str(run)]
        if args.limit is not None:
            command.extend(('--limit', str(args.limit)))
        print(f'\nStep {index}/3: {stage}', flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            print(f'Stopped at {stage}. Add --resume to continue with saved progress.', file=sys.stderr)
            return error.returncode
    print('\nGenerating research tables and visual review gallery', flush=True)
    subprocess.run([sys.executable, str(EXPERIMENT / 'analyze.py'), '--run', str(run)],
                   cwd=ROOT, check=True)
    print(f"\nFinished. Open {run / 'analysis.md'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
