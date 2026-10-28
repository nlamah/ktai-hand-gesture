"""Small mathematical oracles, independently specified from production helpers."""
from collections import Counter
import math

import numpy as np

from gesturelab.features.finger_features import angle, compute_finger_features
from gesturelab.features.landmark_features import landmark_features
from gesturelab.features import paper_shin2021 as paper
from gesturelab.features.training_inputs import TRAINING_INPUTS
from gesturelab.validation.report import write_json

COUNTS = {'finger_features_only': 81, 'without_features': 126, 'with_features': 207,
          'shin2021_distance': 190, 'shin2021_angle': 630, 'shin2021_combined': 820,
          'shin2021_with_finger_features': 901, 'all_features': 1027}
TABLE2_ROWS = [15, 18, 17, 16, 16, 14, 13, 12, 12, 10, 9, 8, 8, 6, 5, 4, 4, 2, 1, 0, 0]


def reference_hand():
    """Explicit planar hand; middle wrist-to-MCP length is 2, index joints are right angles."""
    return np.array([[0, 0, 0], [-1, 0, 0], [-2, 0, 0], [-3, 0, 0], [-4, 0, 0],
        [-1, 1, 0], [-1, 2, 0], [-.5, 2, 0], [-.5, 3, 0],
        [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0],
        [1, 1.5, 0], [1, 2.5, 0], [1, 3.5, 0], [1, 4.5, 0],
        [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0]], dtype=float)


def reference_sample():
    hand = reference_hand()
    return dict(image_landmarks=hand * .03 + .5, world_landmarks=hand, width=200, height=100)


def explicit_values():
    points = reference_hand()
    points[2] = [3, 4, 0]
    raw = paper.raw_distances(points)
    geometry = compute_finger_features(reference_hand())
    base = landmark_features(**reference_sample())
    cosines = paper.angle_features(reference_hand())
    expected = {'distance_3_4_5': 5, 'joint_right': 90, 'joint_straight': 180,
        'index_joint1': 90, 'index_joint2': 90, 'middle_straightness': 1,
        'index_straightness': math.sqrt(17) / 5, 'index_middle_tip_distance': math.sqrt(17) / 4,
        'image_index_x': -1, 'world_index_x': -.5, 'axis_negative_x': -1, 'axis_positive_y': 1}
    actual = dict(distance_3_4_5=raw['shin_dist_00_02'], joint_right=angle([1, 0, 0], [0, 0, 0], [0, 1, 0]),
        joint_straight=angle([-1, 0, 0], [0, 0, 0], [1, 0, 0]), index_joint1=geometry['geom_index_joint1_deg'],
        index_joint2=geometry['geom_index_joint2_deg'], middle_straightness=geometry['geom_middle_straightness'],
        index_straightness=geometry['geom_index_straightness'], index_middle_tip_distance=geometry['geom_tip_distance_index_middle'],
        image_index_x=base['norm_5_x'], world_index_x=base['world_5_x'],
        axis_negative_x=cosines['shin_angle_00_01_cos_x'], axis_positive_y=cosines['shin_angle_05_06_cos_y'])
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-9)
    return dict(expected=expected, actual=actual)


def invariant_checks():
    hand = reference_hand()
    sample = reference_sample()
    first = landmark_features(**sample)
    moved = landmark_features(sample['image_landmarks'] * 2 + 3, hand * 3 + [4, -2, 8], 200, 100)
    keys = [k for k in first if k.startswith(('norm_', 'world_'))]
    np.testing.assert_allclose([first[k] for k in keys], [moved[k] for k in keys], atol=1e-12)
    geometry = compute_finger_features(hand)
    shifted = compute_finger_features(hand * 3 + [4, -2, 8])
    np.testing.assert_allclose(list(geometry.values()), list(shifted.values()), atol=1e-5)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = hand @ rotation.T
    np.testing.assert_allclose(list(paper.distance_features(hand).values()),
                               list(paper.distance_features(rotated * 3 + 2).values()), atol=1e-12)
    np.testing.assert_allclose(list(geometry.values()), list(compute_finger_features(rotated).values()), atol=1e-5)
    assert paper.angle_features(hand)['shin_angle_00_01_cos_x'] == -1
    assert paper.angle_features(rotated)['shin_angle_00_01_cos_y'] == -1
    assert paper.angle_features(rotated)['shin_angle_00_01_cos_x'] == 0
    return 'Normalized translations/scales passed; finger/paper distances rotate invariantly; axis cosine changes -x to -y.'


def invalid_checks():
    hand = reference_hand()
    rejected = []
    for name, operation in (
        ('finger_nan', lambda: compute_finger_features(np.full((21, 3), np.nan))),
        ('image_inf', lambda: landmark_features(np.full((21, 3), np.inf), hand)),
        ('bad_shape', lambda: paper.angle_features(np.zeros((20, 3)))),
        ('zero_vectors', lambda: paper.angle_features(np.zeros((21, 3)))),
        ('zero_distance_variance', lambda: paper.distance_features(np.zeros((21, 3)))),
        ('coincident_joint', lambda: angle([0, 0, 0], [0, 0, 0], [1, 0, 0]))):
        try:
            operation()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f'{name} did not reject invalid input.')
    return rejected


def composition_checks():
    sample = reference_sample()
    values = {name: spec.features(sample, None) for name, spec in TRAINING_INPUTS.items()}
    assert {name: len(v) for name, v in values.items()} == COUNTS
    coords = [prefix + f'{i}_{axis}' for prefix in ('norm_', 'world_') for i in range(21) for axis in 'xyz']
    assert list(values['without_features']) == coords
    paper_values, finger = values['shin2021_combined'], values['finger_features_only']
    assert list(paper_values) == list(values['shin2021_distance']) + list(values['shin2021_angle'])
    assert values['shin2021_with_finger_features'] == {**paper_values, **finger}
    assert values['all_features'] == {**paper_values, **values['without_features'], **finger}
    assert list(values['all_features']) == list(paper_values) + coords + list(finger)
    return COUNTS


def paper_table_checks():
    counts = Counter(i for i, j in paper.DISTANCE_PAIRS)
    assert [counts[i] for i in range(21)] == TABLE2_ROWS
    # Independently transcribed from Table 2, not imported from the production omission list.
    omitted = {(1, 2), (1, 5), (1, 9), (1, 13), (1, 17), (2, 3), (3, 4), (4, 5),
               (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
               (16, 17), (18, 19), (19, 20), (20, 21)}
    expected = [(i - 1, j - 1) for i in range(1, 22) for j in range(i + 1, 22) if (i, j) not in omitted]
    assert list(paper.DISTANCE_PAIRS) == expected
    assert list(paper.ALL_PAIRS) == [(i, j) for i in range(21) for j in range(i + 1, 21)]
    return dict(distance_pairs=190, direction_pairs=210, row_counts=TABLE2_ROWS)


def audit_features(report, out):
    inventory = {name: dict(title=s.title, expected_features=s.expected_features, prefixes=s.prefixes,
                           scaled_prefixes=s.scaled_prefixes, definition=s.definition) for name, s in TRAINING_INPUTS.items()}
    write_json(out / 'feature_inventory.json', inventory)
    for name, operation in [('explicit_values', explicit_values), ('invariances', invariant_checks),
                            ('invalid_inputs', invalid_checks), ('composition_order', composition_checks),
                            ('paper_tables', paper_table_checks)]:
        try:
            found = operation()
        except Exception as error:
            report.add('features.' + name, operation.__doc__ or operation.__name__, 'Explicit mathematical contract',
                       str(error), False, ['gesturelab/validation/features.py'])
        else:
            report.add('features.' + name, operation.__doc__ or operation.__name__, 'Explicit mathematical contract',
                       found, True, ['gesturelab/validation/features.py'])
    report.add('features.paper_ambiguities', 'Compare Tables 2/3 and Equations 1–6 with the code mapping.',
        'Exact reproduction needs unambiguous historical coordinates, normalization axis and angle conversion.',
        'Literal pairs/cosines checked. Image coordinates and per-photo ddof=0 z-score remain documented assumptions.',
        files=['docs/PAPER_SHIN2021.md', 'https://doi.org/10.3390/s21175856'],
        limitation='The article does not uniquely determine these implementation details.', next_step='Human review of source formulas and assumptions.')
