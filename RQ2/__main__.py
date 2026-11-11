"""Run the RQ2 experiments from the group paper; see RQ2/README.md."""
import argparse
import subprocess

from baseline.runner import ROOT, run_commands
from RQ2.workflows import ACTIONS, WORKFLOWS, commands_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', choices=['list', *WORKFLOWS])
    parser.add_argument('--action', choices=ACTIONS, default='reproduce')
    parser.add_argument('--name', help='Output run name; reproduce replaces an existing run unless --resume is used.')
    parser.add_argument('--limit', type=int, help='First N dataset images; technical check only.')
    parser.add_argument('--resume', action='store_true', help='Keep and resume the named run with the same settings.')
    parser.add_argument('--reference', help='Reference run name for compare; --name identifies the repeat.')
    parser.add_argument('--status', action='store_true', help='Show contribution status.')
    parser.add_argument('--dry-run', action='store_true', help='Show exact commands without executing them.')
    args = parser.parse_args(argv)
    if args.status:
        print((ROOT / 'RQ2' / 'contribution.json').read_text(encoding='utf-8'))
        return 0
    if args.experiment == 'list':
        for key, workflow in WORKFLOWS.items():
            actions = ', '.join(action for action in ACTIONS if action in workflow)
            print(f'{key:20} {workflow["results"]} ({actions})')
        return 0
    if args.experiment is None:
        parser.error('Choose list or an experiment; see --help.')
    try:
        commands = commands_for(args.experiment, args.action, name=args.name, limit=args.limit,
                                resume=args.resume, reference=args.reference)
    except ValueError as error:
        parser.error(str(error))
    try:
        run_commands(commands, args.dry_run)
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
