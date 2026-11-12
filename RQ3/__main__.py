"""Run the RQ3 feature and orientation experiments; see RQ3/README.md."""
import argparse
import subprocess

from baseline.runner import ROOT, run_commands
from RQ3.workflows import ACTIONS, WORKFLOWS, commands_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', choices=['list', *WORKFLOWS])
    parser.add_argument('--action', choices=ACTIONS, default='reproduce')
    parser.add_argument('--status', action='store_true', help='Show contribution status.')
    parser.add_argument('--dry-run', action='store_true', help='Show exact commands without executing them.')
    args = parser.parse_args(argv)
    if args.status:
        print((ROOT / 'RQ3' / 'contribution.json').read_text(encoding='utf-8'))
        return 0
    if args.experiment == 'list':
        for key, workflow in WORKFLOWS.items():
            actions = ', '.join(action for action in ACTIONS if action in workflow)
            print(f'{key:20} {workflow["results"]} ({actions})')
        return 0
    if args.experiment is None:
        parser.error('Choose list or an experiment; see --help.')
    try:
        commands = commands_for(args.experiment, args.action)
    except ValueError as error:
        parser.error(str(error))
    try:
        run_commands(commands, args.dry_run)
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
