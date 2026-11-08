"""Run the RQ3 experiments from the group paper; see the main README."""
import argparse
from baseline.runner import run_commands
from RQ3.workflows import WORKFLOWS, commands_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['list', *WORKFLOWS])
    parser.add_argument('--action', choices=['reproduce', 'verify', 'report'], default='reproduce')
    parser.add_argument('--dry-run', action='store_true', help='Show exact commands without executing them.')
    args = parser.parse_args()
    if args.experiment == 'list':
        for key, workflow in WORKFLOWS.items():
            print(f'{key:20} {workflow["results"]}')
        return
    try:
        commands = commands_for(args.experiment, args.action)
    except ValueError as error:
        parser.error(str(error))
    run_commands(commands, args.dry_run)


if __name__ == '__main__':
    main()
