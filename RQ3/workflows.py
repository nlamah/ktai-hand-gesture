"""Reproduction order for the actual experiments cited in the group paper.

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
    module('report_feature_factorial'),
]
MIRRORS = [module('mirror_dataset', '--workers', '4'),
           module('mirror_augmentation', '--stage', 'cv'),
           module('mirror_augmentation', '--stage', 'train'),
           module('verify_mirror'), module('report_mirror')]
GH_ROTATION = [module('gh_rotation_standalone', '--stage', stage) for stage in ['freeze', 'cv', 'train']]
GH_ROTATION += [['-m', 'RQ3.gh_rotation']]

WORKFLOWS = {
    'features': {'results': 'results/feature_factorial_01', 'reproduce': FEATURES,
                 'verify': [module('verify_feature_factorial')], 'report': [module('report_feature_factorial')]},
    'coordinate_control': {'results': 'results/direction_joint_01',
                           'reproduce': [module('compare_direction_joint'), module('verify_direction_joint')],
                           'verify': [module('verify_direction_joint')]},
    'mirrors': {'results': 'results/mirror_01',
                'reproduce': [['-m', 'baseline.frozen_worker', 'mirrors', 'reproduce']],
                'verify': [['-m', 'baseline.frozen_worker', 'mirrors', 'verify']],
                'report': [['-m', 'baseline.frozen_worker', 'mirrors', 'report']]},
    'normalization': {'results': 'results/handedness_01',
                      'reproduce': [module('handedness_normalization', '--stage', 'geometry'),
                                    module('handedness_normalization', '--stage', 'cv', '--model', 'svm'),
                                    module('handedness_normalization', '--stage', 'cv', '--model', 'lightgbm'),
                                    module('handedness_normalization', '--stage', 'export'),
                                    module('handedness_normalization', '--stage', 'train'),
                                    module('verify_handedness'), module('report_handedness')],
                      'verify': [module('verify_handedness')], 'report': [module('report_handedness')]},
    'gh_rotation': {'results': 'results/rotation_GH_y180_01',
                    'reproduce': [['-m', 'baseline.frozen_worker', 'gh_rotation', 'reproduce']],
                    'verify': [['-m', 'baseline.frozen_worker', 'gh_rotation', 'verify']],
                    'report': [['-m', 'baseline.frozen_worker', 'gh_rotation', 'report']]},
    'occlusion': {'results': 'results/gh_occlusion_01',
                  'reproduce': [module('gh_occlusion', '--stage', 'prepare'),
                                module('gh_occlusion', '--stage', 'screen'),
                                module('gh_occlusion', '--stage', 'train', '--arm', 'noise002'),
                                module('gh_occlusion', '--stage', 'train', '--arm', 'noise005'),
                                module('gh_occlusion', '--stage', 'report'),
                                module('gh_occlusion_rotated_test'), module('verify_gh_occlusion'),
                                module('report_gh_occlusion')],
                  'verify': [module('verify_gh_occlusion')], 'report': [module('report_gh_occlusion')]},
    'live_examples': {'results': 'results/gh_live_diagnostics_20261107',
                      'reproduce': [['-m', 'RQ3.live_examples']],
                      'verify': [['-m', 'RQ3.live_examples']],
                      'report': [['-m', 'RQ3.live_examples']]},
}


def commands_for(experiment, action='reproduce'):
    if experiment not in WORKFLOWS:
        raise ValueError(f'Unknown experiment: {experiment}')
    if action not in WORKFLOWS[experiment]:
        raise ValueError(f'{experiment} has no separate {action} action.')
    return [list(command) for command in WORKFLOWS[experiment][action]]
