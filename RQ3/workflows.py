"""Reproduction order for the feature comparison and coordinate control.

The numerical implementations retain their frozen research.* module names.
This registry is the maintained public interface; it does not copy algorithms.
"""

ACTIONS = ('reproduce', 'verify', 'report')


def module(name, *args):
    return ['-m', 'research.studies.' + name, *args]



FEATURES = [
    module('feature_factorial', '--stage', 'parity'),
    module('feature_factorial', '--stage', 'train', '--model', 'svm'),
    module('feature_factorial', '--stage', 'train', '--model', 'lightgbm'),
    module('fast_factorial_permutation'),
    module('feature_factorial', '--stage', 'permutation', '--model', 'svm'),
    module('feature_factorial', '--stage', 'permutation', '--model', 'lightgbm'),
    module('feature_factorial', '--stage', 'export'),
    module('verify_feature_factorial'),
    ['-m', 'RQ3.comparison_diagnostics'],
    module('report_feature_factorial'),
]

WORKFLOWS = {
    'diagnostics': {'results': 'results/feature_diagnostics_01',
                    'reproduce': [['-m', 'RQ3.comparison_diagnostics']],
                    'verify': [['-m', 'RQ3.comparison_diagnostics', '--verify']]},
    'features': {'results': 'results/feature_factorial_01', 'reproduce': FEATURES,
                 'verify': [module('verify_feature_factorial')], 'report': [module('report_feature_factorial')]},
    'coordinate_control': {'results': 'results/direction_joint_01',
                           'reproduce': [module('compare_direction_joint'), module('verify_direction_joint')],
                           'verify': [module('verify_direction_joint')]},

}


def commands_for(experiment, action='reproduce'):
    if experiment not in WORKFLOWS:
        raise ValueError(f'Unknown experiment: {experiment}')
    if action not in WORKFLOWS[experiment]:
        raise ValueError(f'{experiment} has no separate {action} action.')
    return [list(command) for command in WORKFLOWS[experiment][action]]
