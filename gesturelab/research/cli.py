"""Small command adapters shared by the research CLI and the native webcam app."""
import argparse

from gesturelab.validation.report import read_json


def _command(sub, name, function, help):
    parser = sub.add_parser(name, help=help)
    parser.set_defaults(function=function)
    return parser


def _register(**args):
    from gesturelab.research.models import register_model
    return register_model(**args)


def _protocol(**args):
    from gesturelab.research.protocol import create_protocol
    if args.get('conditions'): args['conditions'] = read_json(args['conditions'])
    return create_protocol(**args)


def _freeze(**args):
    from gesturelab.research.protocol import freeze_test
    return freeze_test(**args)


def _capture(**args):
    from gesturelab.research.native import run_protocol_capture
    return run_protocol_capture(**args)


def _ab(**args):
    from gesturelab.research.native import run_live_ab
    args['model_b'] = args.get('model_b') or args['model_a']
    args['mirror'] = not args.pop('no_mirror')
    return run_live_ab(**args)


def _analyze(**args):
    from gesturelab.research.analysis import analyze_development
    return analyze_development(**args)


def _train(**args):
    from gesturelab.research.improvement import train_improvement
    return train_improvement(**args)


def _evaluate(**args):
    from gesturelab.research.evaluation import evaluate_test
    return evaluate_test(**args)


def _annotate(**args):
    from gesturelab.research.recording import annotate_attempt
    return annotate_attempt(**args)


def _notes(**args):
    from gesturelab.research.analysis import import_notes
    return import_notes(**args)


def _export(**args):
    from gesturelab.research.analysis import export_live
    return export_live(**args)


def add_native_commands(sub):
    capture = _command(sub, 'record-protocol', _capture, 'Record every planned attempt; predictions remain hidden.')
    capture.add_argument('--protocol', dest='protocol_file', required=True)
    capture.add_argument('--project', required=True)
    capture.add_argument('--seal', help='Required for final-test capture; freeze models first.')
    capture.add_argument('--resume', action='store_true', help='Continue missing planned slots without replacing saved attempts.')
    live = _command(sub, 'ab', _ab, 'Same-frame development A/B (or one model if B is omitted).')
    live.add_argument('--model-a', required=True, help='Registered model JSON; use research register first.')
    live.add_argument('--model-b', help='Registered model JSON; defaults to A for one-model exploration.')
    live.add_argument('--out', required=True)
    live.add_argument('--camera', type=int, default=0)
    live.add_argument('--intended-label')
    live.add_argument('--countdown', type=float, default=0)
    live.add_argument('--smoothing', type=int, default=1, help='1 = raw default; larger windows are explicitly separate.')
    live.add_argument('--no-mirror', action='store_true', help='Display only; analysis pixels never change.')


def _registration_commands(sub):
    register = _command(sub, 'register', _register, 'Inspect actual classifier/features and register an existing model.')
    for flag in ('model', 'out', 'alphabet'):
        register.add_argument('--' + flag, required=True)
    for flag in ('experiment', 'name'):
        register.add_argument('--' + flag)
    register.add_argument('--require-baseline', action='store_true')
    freeze = _command(sub, 'freeze', _freeze, 'Lock test protocol and model bytes before capture/evaluation.')
    for flag in ('protocol', 'model-a', 'model-b', 'out', 'hypothesis'):
        freeze.add_argument('--' + flag, required=True)
    freeze.add_argument('--viewed-retest', action='store_true', help='New models on an already evaluated protocol; explicitly not unseen confirmation.')


def _protocol_command(sub):
    parser = _command(sub, 'protocol', _protocol, 'Write a new seeded nominal-viewpoint protocol.')
    for flag in ('out', 'alphabet', 'participant', 'session'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--letters', nargs='+', required=True)
    parser.add_argument('--role', choices=('development', 'test'), default='development')
    for flag, default in (('repetitions', 5), ('camera', 0), ('seed', 42)):
        parser.add_argument('--' + flag, type=int, default=default)
    for flag in ('countdown', 'interval'):
        parser.add_argument('--' + flag, type=float, default=3.)
    parser.add_argument('--angle-method', choices=('camera_view', 'gesture_rotation'), default='camera_view')
    parser.add_argument('--confirm-label-validity', action='store_true')
    for flag in ('conditions', 'references', 'reference-source'):
        parser.add_argument('--' + flag)


def _analysis_commands(sub):
    parser = _command(sub, 'analyze', _analyze, 'Development-only confusion, examples, notes and paired scores.')
    for flag in ('model-a', 'project', 'out'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--model-b')
    parser.add_argument('--letters', nargs='+')
    parser = _command(sub, 'evaluate', _evaluate, 'Evaluate the locked pair on identical saved test detections.')
    for flag in ('seal', 'project', 'out'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--acknowledge-unchecked-overlap', action='store_true')
    parser = _command(sub, 'annotate', _annotate, 'Append a human note/explicit invalidity; keep the photo.')
    for flag in ('project', 'photo-id', 'note'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--invalid', action='store_true')
    parser = _command(sub, 'notes', _notes, 'Import human notes downloaded from an analysis report.')
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--notes', required=True)
    parser = _command(sub, 'export-live', _export, 'Export selected live categories and disagreement images.')
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)


def _train_command(sub):
    parser = _command(sub, 'train', _train, 'One unchanged baseline pipeline refit on original + new development data.')
    for flag in ('baseline', 'additions', 'dataset-out', 'experiment-out', 'hypothesis', 'change', 'expected'):
        parser.add_argument('--' + flag, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--ids', nargs='+')
    choice.add_argument('--all-development', action='store_true')
    parser.add_argument('--condition', help='Optional explicit selection, e.g. front for a separate frontal control.')
    parser.add_argument('--match-count', type=int, help='Require exactly this number of usable added photos; never auto-subsample.')


def execute(parser, args):
    function = args.pop('function')
    args.pop('command', None)
    args.pop('mode', None)
    try:
        result = function(**args)
        if result is not None: print(result)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Research command stopped: {error}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for add in (_registration_commands, _protocol_command, _analysis_commands, _train_command, add_native_commands):
        add(sub)
    execute(parser, vars(parser.parse_args()))
