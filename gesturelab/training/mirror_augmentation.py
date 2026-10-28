"""Training-only reflection of stored landmarks; source photos remain unchanged."""
import numpy as np
import pandas as pd

from gesturelab.features.training_inputs import input_table

MIRROR_SUFFIX = '::horizontal_mirror'
DEFINITION = 'landmark-horizontal-reflection-v1'


def mirror_sample(sample):
    """Reflect image x as 1-x and world x as -x; preserve semantic point indices."""
    if sample.get('role') != 'development' or sample.get('provenance', {}).get('protocol_role') == 'test':
        raise ValueError('Mirror augmentation accepts development samples only.')
    result = dict(sample)
    for key, offset in (('image_landmarks', 1.), ('world_landmarks', 0.)):
        if key not in sample:
            continue
        points = np.asarray(sample[key], dtype=float).copy()
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError('Mirror augmentation requires finite 21 x 3 landmarks.')
        points[:, 0] = offset - points[:, 0]
        result[key] = points.tolist()
    if result.get('handedness') in ('Left', 'Right'):
        result['handedness'] = 'Right' if result['handedness'] == 'Left' else 'Left'
    result.pop('features', None)  # Reconstruct geometry; never reuse cached original features.
    result.update(id=sample['id'] + MIRROR_SUFFIX,
                  augmentation=dict(type=DEFINITION, original_id=sample['id']))
    return result


def training_samples(samples, enabled):
    if not enabled:
        return samples
    mirrored = [mirror_sample(sample) for sample in samples]
    ids = [s['id'] for s in samples + mirrored]
    if len(ids) != len(set(ids)):
        raise ValueError('Mirrored row IDs collide with original photo IDs.')
    return samples + mirrored


def fit_row_ids(plan, source_ids):
    ids = list(source_ids)
    return ids + [identity + MIRROR_SUFFIX for identity in ids] if plan.get('mirror_training') else ids


def training_input_table(samples, thresholds, variant, enabled=False):
    original = input_table(samples, thresholds, variant)
    if not enabled:
        return original
    mirrored = training_samples(samples, True)[len(samples):]
    # Build the two tables separately to limit peak Python-object memory on full datasets.
    reflected = input_table(mirrored, thresholds, variant)
    return pd.concat([original, reflected], ignore_index=True)


def augmentation_metadata(plan):
    enabled = bool(plan.get('mirror_training', False))
    original_count = len(plan['development_ids'])
    return dict(mirror_training=enabled, augmentation_definition=DEFINITION if enabled else None,
        n_original_training_photos=original_count, n_mirrored_training_rows=original_count if enabled else 0,
        n_training_rows=original_count * (2 if enabled else 1),
        augmentation_limitation='Geometric landmark reflection; MediaPipe is not rerun and real left/right detection errors are not simulated.' if enabled else None)


def row_provenance(plan, records):
    rows = []
    for source_id in plan['development_ids']:
        sample = records[source_id]
        for identity in fit_row_ids(plan, [source_id]):
            rows.append(dict(id=identity, original_id=source_id, label=sample['label'],
                transform=DEFINITION if identity != source_id else 'identity',
                participant=sample['participant'], session=sample['session'],
                source_image_path=sample['image_path'], role='development'))
    return rows
