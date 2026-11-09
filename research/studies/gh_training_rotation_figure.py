"""Visualize two actual training skeletons and verify the exact G/H y180 inputs."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from research.data import atomic_json
from research.feature_review import CONNECTIONS, hand_crop
from research.gh_vertical import rotate_vertical
from research.shin import ALL_PAIRS, combined_features

OUT = Path('results/gh_training_rotation_01')
FIG = Path('paper/figures/gh_training_rotation.pdf')
CACHE = Path('data/asl_kaggle/research')
PROTOCOL = Path('results/rotation_GH_y180_01/protocol.json')
SOURCES = (('G', '003ed35e38eafb4474345302'), ('H', '00216cb15dd8ab359f453e59'))
DIAGONAL = np.array([-1., 1., -1.])


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def skeleton(axis, points, spans, title):
    for start, end in CONNECTIONS:
        axis.plot(*points[[start, end]].T, color='#008f96', linewidth=2)
    axis.scatter(*points.T, color='#d88400', s=16, depthshade=False)
    axis.scatter(*points[0], color='#222222', s=30, depthshade=False)
    axis.text(*points[0], ' 0', fontsize=9)
    center = (points.min(axis=0)+points.max(axis=0))/2
    lo, hi = center-spans/2, center+spans/2
    axis.plot([0, 0], [lo[1], hi[1]], [0, 0], color='#567144', linestyle='--', linewidth=1)
    axis.set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), zlim=(lo[2], hi[2]),
             xlabel='x', ylabel='y', zlabel='z')
    axis.view_init(elev=16, azim=-56, vertical_axis='y')
    axis.set_box_aspect(spans)
    for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension.set_ticks([0])
        dimension.set_ticklabels(['0'])
        dimension.label.set_size(10)
        dimension.pane.fill = False
    axis.tick_params(labelsize=8, pad=0)
    axis.grid(False)
    axis.set_title(title, fontsize=11, pad=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['matrix_diagonal'] == DIAGONAL.tolist()
    for path, expected in protocol['frozen_inputs_sha256'].items():
        assert digest(path) == expected
    for path in ('research/gh_vertical.py', 'research/studies/gh_rotation_standalone.py'):
        assert digest(path) == protocol['source_sha256'][path]
    metadata = json.loads((CACHE/'inputs_metadata.json').read_text())
    assert digest('data/asl_kaggle/manifest.json') == metadata['hashes']['manifest']
    indices = {row['id']: i for i, row in enumerate(metadata['samples'])}
    raw = json.loads(Path('data/asl_kaggle/manifest.json').read_text())
    samples = {row['id']: row for row in raw['samples'] if row['id'] in dict(SOURCES).values()}
    del raw
    selected_indices = [indices[id_] for _, id_ in SOURCES]
    with np.load(CACHE/'inputs.npz') as cache:
        training = np.c_[cache['X'][selected_indices, :820], cache['angles'][selected_indices]]
    assert training.shape == (2, 835)
    augmented = rotate_vertical(training)
    np.testing.assert_array_equal(training[:, :190], augmented[:, :190])
    np.testing.assert_array_equal(training[:, 820:], augmented[:, 820:])
    np.testing.assert_array_equal(rotate_vertical(augmented), training)
    assert np.linalg.det(np.diag(DIAGONAL)) == 1

    examples, geometry = [], []
    for i, (letter, id_) in enumerate(SOURCES):
        sample = samples[id_]
        assert sample['label'] == letter and metadata['samples'][indices[id_]]['label'] == letter
        points = np.asarray(sample['image_landmarks'], dtype=float)
        assert points.shape == (21, 3) and np.isfinite(points).all()
        original = np.asarray(list(combined_features(points).values()), np.float32)
        np.testing.assert_array_equal(original, training[i, :820])
        centered = points-points[0]
        rotated = centered*DIAGONAL
        transformed = points[0]+rotated
        recomputed = np.asarray(list(combined_features(transformed).values()), np.float32)
        np.testing.assert_allclose(recomputed, augmented[i, :820], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(points[:, 1], transformed[:, 1])
        np.testing.assert_allclose(points[0]+rotated*DIAGONAL, points, atol=1e-12, rtol=0)
        a, b = np.asarray(ALL_PAIRS).T
        np.testing.assert_allclose(np.linalg.norm(points[b]-points[a], axis=1),
                                   np.linalg.norm(rotated[b]-rotated[a], axis=1), atol=1e-12, rtol=0)
        path = Path('data/asl_kaggle')/sample['image_path']
        with Image.open(path) as image:
            crop, box = hand_crop(image, points)
        geometry.append((centered, rotated, crop))
        examples.append({'label': letter, 'id': id_, 'filename': sample['filename'],
                         'training_cache_row': indices[id_], 'image': str(path),
                         'image_sha256': digest(path), 'crop_box': box,
                         'original_image_landmarks': points.tolist(),
                         'original_wrist_centered_landmarks': centered.tolist(),
                         'rotated_wrist_centered_landmarks': rotated.tolist(),
                         'rotated_absolute_landmarks': transformed.tolist(),
                         'augmentation_feature_max_abs_error': float(np.abs(recomputed-augmented[i, :820]).max())})

    spans = np.max([np.ptp(points, axis=0) for pair in geometry for points in pair[:2]], axis=0)*1.35
    fig = plt.figure(figsize=(9, 6.5))
    grid = fig.add_gridspec(2, 3, width_ratios=[.9, 1.12, 1.12],
                           left=.015, right=.97, bottom=.08, top=.93, wspace=.2, hspace=.44)
    for row, ((letter, _), (original, rotated, crop)) in enumerate(zip(SOURCES, geometry)):
        photo = fig.add_subplot(grid[row, 0])
        photo.imshow(crop)
        photo.axis('off')
        photo.set_title(f'{letter} · {examples[row]["filename"]}\nTraining photograph + landmarks', fontsize=11)
        skeleton(fig.add_subplot(grid[row, 1], projection='3d'), original, spans,
                 f'{letter} · original landmarks')
        skeleton(fig.add_subplot(grid[row, 2], projection='3d'), rotated, spans,
                 f'{letter} · 180° around vertical y axis')
    fig.savefig(FIG, bbox_inches='tight')
    fig.savefig(OUT/'gh_training_rotation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    atomic_json(OUT/'provenance.json', {
        'selection': 'Same fixed G1128/H1198 source IDs as the dataset illustrations; both are actual accepted training-cache rows.',
        'rotation': {'diagonal': DIAGONAL.tolist(), 'degrees': 180, 'axis': 'camera y',
                     'pivot': 'wrist (landmark 0)', 'units': 'original normalized image coordinates'},
        'examples': examples,
        'inputs_sha256': {str(path): digest(path) for path in
                          (CACHE/'inputs.npz', CACHE/'inputs_metadata.json', Path('data/asl_kaggle/manifest.json'))},
        'source_sha256': {path: digest(path) for path in
                          ('research/studies/gh_training_rotation_figure.py', 'research/gh_vertical.py',
                           'research/shin.py', 'research/feature_review.py')},
        'figure_sha256': digest(FIG),
        'frozen_experiment_protocol_sha256': digest(PROTOCOL),
        'scope': 'Actual cached training geometry with a verified equivalent visualization of feature augmentation. No RGB hand-back view generated, no re-detection and no refit.',
    })
    atomic_json(OUT/'verification.json', {
        'status': 'passed', 'real_training_examples': 2,
        'original_experiment_input_and_augmentation_hashes_verified': True,
        'cached_original_shin_values_exact': True,
        'rotated_landmark_features_match_actual_augmentation_atol': 1e-6,
        'pair_distances_invariant_atol': 1e-12,
        'all_190_distance_and_15_joint_inputs_unchanged': True,
        'rotation_twice_restores_actual_training_features_exactly': True,
        'y_coordinates_unchanged': True, 'rotation_determinant': 1,
        'models_fitted': 0, 'detector_calls': 0,
        'figure_sha256': digest(FIG), 'provenance_sha256': digest(OUT/'provenance.json'),
    })
    print('Two real training skeletons exported; their rotated features match the actual augmentation.')


if __name__ == '__main__':
    main()
