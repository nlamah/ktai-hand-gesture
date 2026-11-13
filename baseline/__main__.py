"""Shared baseline: check, train and transfer inputs."""
import argparse
import json
from baseline.check import status, verify_baseline, verify_inputs
from baseline.runner import run_commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Check installation, inputs and contribution status.')
    sub.add_parser('verify', help='Recompute shared DS scores from saved held-out predictions.')
    sub.add_parser('verify-data', help='Check SHA256 hashes of extracted paper inputs.')
    sub.add_parser('verify-results', help='Check SHA256 hashes of separately supplied RQ3 results.')
    train = sub.add_parser('train', help='Train the DS baseline on the five fixed folds.')
    train.add_argument('--model', choices=['svm', 'lightgbm', 'both'], default='both')
    train.add_argument('--dry-run', action='store_true')
    train.add_argument('--cache', help='Optional input cache for a separate replication.')
    train.add_argument('--out', help='Required with --cache: new output directory under runs/.')
    bundle = sub.add_parser('bundle', help='Export local fixed inputs for instructor reproduction.')
    bundle.add_argument('--output', required=True)
    results = sub.add_parser('bundle-results', help='Export RQ3 CSV/JSON evidence separately from Git.')
    results.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    elif args.command == 'verify':
        print(json.dumps(verify_baseline(), indent=2))
    elif args.command == 'verify-data':
        print(json.dumps(verify_inputs(), indent=2))
    elif args.command == 'verify-results':
        from baseline.bundle import verify_results
        print(json.dumps(verify_results(), indent=2))
    elif args.command == 'bundle':
        from baseline.bundle import create_bundle
        print(json.dumps(create_bundle(args.output), indent=2))
    elif args.command == 'bundle-results':
        from baseline.bundle import create_results_bundle
        print(json.dumps(create_results_bundle(args.output), indent=2))
    else:
        models = ['svm', 'lightgbm'] if args.model == 'both' else [args.model]
        if bool(args.cache) != bool(args.out):
            parser.error('Supply --cache and --out together.')
        if args.cache:
            if args.dry_run:
                print(f'New DS replication: cache={args.cache}, out={args.out}, models={models}')
            else:
                from baseline.train import train_cache
                from baseline.runner import ROOT
                import os
                os.chdir(ROOT)
                train_cache(args.cache, args.out, models)
            return
        run_commands([['-m', 'research.studies.feature_factorial', '--stage', 'train',
                       '--model', model, '--variants', 'DS'] for model in models], args.dry_run)


if __name__ == '__main__':
    main()
