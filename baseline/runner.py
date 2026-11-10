"""Run documented commands with the same Python, always from the project root."""
import importlib
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_commands(commands, dry_run=False):
    for arguments in commands:
        command = [sys.executable, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def run_contribution(package):
    """An explicit integration slot, not a fabricated RQ1/RQ2 implementation."""
    import argparse
    parser = argparse.ArgumentParser(description=f'{package}: coauthor contribution; see README.md.')
    parser.add_argument('--status', action='store_true', help='Show contribution status.')
    args, extra = parser.parse_known_args()
    if args.status:
        print((ROOT / package / 'contribution.json').read_text())
        return
    target = ROOT / package / 'experiment.py'
    if not target.exists():
        parser.error(f'{package}/experiment.py has not been supplied. No results were computed. See README.md.')
    sys.argv = [str(target), *extra]
    importlib.import_module(f'{package}.experiment').main()
