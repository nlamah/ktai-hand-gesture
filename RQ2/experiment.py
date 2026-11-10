"""Run and check RQ2 detection recovery from the repository root."""
import argparse
import json
from pathlib import Path
import subprocess
import shlex
import sys

from RQ2.workflows import WORKFLOWS, commands_for


ROOT = Path(__file__).resolve().parents[1]


def run_commands(commands, dry_run=False):
    # Keep detection recovery independent of the classifier imports in baseline.__init__.
    for arguments in commands:
        command = [sys.executable, *arguments]
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', choices=['list', *WORKFLOWS])
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--action', choices=['reproduce', 'verify', 'report'], default='reproduce')
    parser.add_argument('--name', help='Output run name; reproduce replaces an existing run unless --resume is used.')
    parser.add_argument('--limit', type=int, help='First N dataset images; technical check only.')
    parser.add_argument('--resume', action='store_true', help='Keep and resume the named run with the same settings.')
    parser.add_argument('--reference', help='Reference run name for compare; --name identifies the repeat.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing or changing files.')
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(json.loads(Path(__file__).with_name('contribution.json').read_text(encoding='utf-8')), indent=2))
        return 0
    if args.experiment == 'list':
        for key, plan in WORKFLOWS.items():
            print(f'{key:20} {plan["results"]} ({", ".join(plan["actions"])})')
        return 0
    if args.experiment is None:
        parser.error('Choose list, recovery or compare; see --help.')
    try:
        commands = commands_for(args.experiment, args.action, name=args.name,
                                limit=args.limit, resume=args.resume, reference=args.reference)
    except ValueError as error:
        parser.error(str(error))
    try:
        run_commands(commands, dry_run=args.dry_run)
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
