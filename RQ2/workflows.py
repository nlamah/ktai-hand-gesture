"""Public commands for the existing RQ2 detection-recovery implementation."""
import re

EXPERIMENT = 'experiments/hand_detection_recovery'
WORKFLOWS = {'recovery': {'results': EXPERIMENT + '/runs/<name>',
                          'actions': ('reproduce', 'verify', 'report')},
             'compare': {'results': EXPERIMENT + '/runs/<name>/repeatability.json',
                         'actions': ('reproduce',)}}


def run_path(name):
    if not name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use a run name starting with a letter or digit, followed by letters, digits, underscores or hyphens.')
    return EXPERIMENT + '/runs/' + name


def commands_for(experiment, action='reproduce', *, name, limit=None, resume=False, reference=None):
    run = run_path(name)
    if experiment not in WORKFLOWS or action not in WORKFLOWS[experiment]['actions']:
        raise ValueError('Unsupported experiment or action.')
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive.')
    if (limit is not None or resume) and (experiment != 'recovery' or action != 'reproduce'):
        raise ValueError('--limit and --resume apply only to recovery reproduction.')
    if experiment == 'compare':
        source = run_path(reference)
        if source == run:
            raise ValueError('Reference and repeat must be different runs.')
        return [[EXPERIMENT + '/compare_runs.py', '--reference', source, '--repeat', run]]
    if reference is not None:
        raise ValueError('--reference applies only to compare.')
    verify = [EXPERIMENT + '/verify_results.py', '--run', run]
    plot = [EXPERIMENT + '/plot_results.py', '--run', run]
    if action == 'verify':
        return [verify]
    if action == 'report':
        return [[EXPERIMENT + '/analyze.py', '--run', run], plot]
    command = [EXPERIMENT + '/run.py', '--name', name]
    if limit is not None:
        command.extend(['--limit', str(limit)])
    if resume:
        command.append('--resume')
    return [command, verify, plot]
