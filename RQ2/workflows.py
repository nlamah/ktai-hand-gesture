"""Reproduction order for the RQ2 detection-recovery experiment in the group paper.

The numerical implementation stays in research/recovery_experiment.py and
experiments/hand_detection_recovery/. This registry is the maintained public
interface; it does not copy algorithms. Commands contain {name}, {run} and
{reference} placeholders that commands_for fills with validated run names.
"""
import re

ACTIONS = ('reproduce', 'verify', 'report')
EXPERIMENT = 'experiments/hand_detection_recovery'


def script(name, *args):
    return [f'{EXPERIMENT}/{name}.py', *args]


VERIFY = [script('verify_results', '--run', '{run}')]
REPORT = [script('analyze', '--run', '{run}'), script('plot_results', '--run', '{run}')]
# run.py performs baseline, preprocessing, retry and analysis; then verify and plot.
RECOVERY = [script('run', '--name', '{name}'), *VERIFY, script('plot_results', '--run', '{run}')]

WORKFLOWS = {
    'recovery': {'results': EXPERIMENT + '/runs/<name>',
                 'reproduce': RECOVERY, 'verify': VERIFY, 'report': REPORT},
    'compare': {'results': EXPERIMENT + '/runs/<name>/repeatability.json',
                'reproduce': [script('compare_runs', '--reference', '{reference}', '--repeat', '{run}')]},
}


def run_path(name):
    if not name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use a run name starting with a letter or digit, followed by letters, digits, '
                         'underscores or hyphens.')
    return EXPERIMENT + '/runs/' + name


def commands_for(experiment, action='reproduce', *, name, limit=None, resume=False, reference=None):
    if experiment not in WORKFLOWS:
        raise ValueError(f'Unknown experiment: {experiment}')
    if action not in WORKFLOWS[experiment]:
        raise ValueError(f'{experiment} has no separate {action} action.')
    values = {'name': name, 'run': run_path(name)}
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive.')
    if (limit is not None or resume) and (experiment, action) != ('recovery', 'reproduce'):
        raise ValueError('--limit and --resume apply only to recovery reproduction.')
    if experiment == 'compare':
        values['reference'] = run_path(reference)
        if values['reference'] == values['run']:
            raise ValueError('Reference and repeat must be different runs.')
    elif reference is not None:
        raise ValueError('--reference applies only to compare.')
    commands = [[part.format(**values) for part in command] for command in WORKFLOWS[experiment][action]]
    if experiment == 'recovery' and action == 'reproduce':
        if limit is not None:
            commands[0].extend(['--limit', str(limit)])
        if resume:
            commands[0].append('--resume')
    return commands
