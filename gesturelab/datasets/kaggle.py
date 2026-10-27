"""Resumable ASL Alphabet imports, with supplied test photos kept separate."""
import hashlib
import json
from pathlib import Path
import re

from gesturelab.datasets.data import (
    _store_image, atomic_json, decode_image, project_lock, read_manifest,
)

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
GROUPING_NOTE = (
    'Training groups are consecutive filename-number blocks, not verified people or sessions. '
    'Nearby numbered images stay together; correlation across blocks can remain. '
    'Cross-validation is exploratory and does not establish person-independent accuracy.'
)


def _split_folder(source, name):
    for folder in (source / name / name, source / name):
        if folder.is_dir():
            return folder
    raise ValueError(f'Missing Kaggle folder: {name}.')


def _training_entries(folder, block_size):
    for label in LETTERS:
        directory = folder / label
        if not directory.is_dir():
            continue
        numbered = []
        for path in directory.iterdir():
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            match = re.fullmatch(label + r'(\d+)', path.stem, re.IGNORECASE)
            if not match or int(match[1]) < 1:
                raise ValueError(f'Expected a positive numbered letter filename: {path.name}.')
            numbered.append((int(match[1]), path))
        for number, path in sorted(numbered):
            block = (number - 1) // block_size
            yield {'source': path, 'label': label, 'split': 'train',
                   'source_group': f'kaggle:train:block_{block:04d}', 'number': number}


def _entries(source, block_size):
    entries = list(_training_entries(_split_folder(source, 'asl_alphabet_train'), block_size))
    labels = {entry['label'] for entry in entries}
    if len(labels) < 2:
        raise ValueError('At least two letter directories with training photos are required.')
    test = _split_folder(source, 'asl_alphabet_test')
    for label in sorted(labels):
        path = test / f'{label}_test.jpg'
        if not path.is_file():
            raise ValueError(f'Missing supplied test photo: {path.name}.')
        entries.append({'source': path, 'label': label, 'split': 'test',
                        'source_group': f'kaggle:test:{label}', 'number': None})
    return entries


def _import_config(source, entries, block_size):
    inventory = [(e['source'].relative_to(source).as_posix(), e['source'].stat().st_size,
                  e['source'].stat().st_mtime_ns) for e in entries]
    digest = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    return {'type': 'kaggle_asl_alphabet', 'source': str(source), 'inventory_sha256': digest,
            'block_size': block_size, 'total': len(entries), 'grouping_note': GROUPING_NOTE}


def _initialize_import(manifest, config, resume):
    state = manifest.get('import')
    if state is not None:
        if not resume or any(state.get(key) != value for key, value in config.items()):
            raise ValueError('Resume requires the same Kaggle source files, output and block size.')
        return state
    if resume or manifest['samples']:
        raise ValueError('This is not a resumable Kaggle dataset. Choose a new output directory.')
    state = {**config, 'processed': 0, 'duplicates': 0, 'complete': False}
    manifest['import'] = state
    return state


def _duplicate_image(previous, entry, role, manifest, source, exclude_overlapping_test):
    if previous['label'] != entry['label']:
        raise ValueError('Identical pixels have conflicting labels.')
    if previous['role'] != role:
        if not exclude_overlapping_test or (previous['role'], role) != ('development', 'test'):
            raise ValueError('Identical pixels cross development and test.')
        manifest['import'].setdefault('test_overlaps', []).append({
            'source_file': entry['source'].relative_to(source).as_posix(),
            'label': entry['label'], 'duplicate_of': previous['id']})
    return False


def _append_image(project, manifest, pixels, entry, source, exclude_overlapping_test=False):
    data = entry['source'].read_bytes()
    image = decode_image(data)
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    role = 'test' if entry['split'] == 'test' else 'development'
    previous = pixels.get(digest)
    if previous is not None:
        return _duplicate_image(previous, entry, role, manifest, source, exclude_overlapping_test)
    sample = _store_image(project, data, entry['source'].name, image, digest)
    sample.update(label=entry['label'], role=role, participant='unknown', included=True,
                  session=entry['source_group'], condition='external_asl', provenance={
                      'type': 'kaggle', 'language': 'ASL', 'dataset_split': entry['split'],
                      'source_group': entry['source_group'], 'file_number': entry['number'],
                      'source_file': entry['source'].relative_to(source).as_posix()})
    manifest['samples'].append(sample)
    pixels[digest] = sample
    return True


def _finish_import(manifest):
    state = manifest['import']
    overlaps = len(state.get('test_overlaps', []))
    if overlaps:
        state['test_note'] = f'{overlaps} supplied test photos duplicate training pixels and were excluded as test examples.'
        if not any(s['role'] == 'test' for s in manifest['samples']):
            state['test_note'] += ' No independent final-test photos remain; results are development cross-validation only.'
    state['complete'] = True


def _import_entries(project, manifest, entries, source, checkpoint_every, progress, exclude_overlapping_test):
    state = manifest['import']
    pixels = {sample['pixel_sha256']: sample for sample in manifest['samples']}
    try:
        for index in range(state['processed'], len(entries)):
            added = _append_image(project, manifest, pixels, entries[index], source, exclude_overlapping_test)
            state['duplicates'] += int(not added)
            state['processed'] = index + 1
            if state['processed'] % checkpoint_every == 0:
                atomic_json(project / 'manifest.json', manifest, compact=True)
                if progress:
                    progress(state['processed'], len(entries))
        _finish_import(manifest)
    finally:
        atomic_json(project / 'manifest.json', manifest, compact=True)
    if progress:
        progress(state['processed'], len(entries))
    return {'added': len(manifest['samples']), 'duplicates': state['duplicates'],
            'processed': state['processed'], 'total': len(entries)}


def import_kaggle_dataset(source, project, block_size=500, checkpoint_every=1000, resume=False, progress=None,
                          exclude_overlapping_test=False):
    """Import every A–Z photo; resume only an unchanged, explicitly identified import."""
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('Block size must be a positive integer.')
    if not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError('Checkpoint size must be a positive integer.')
    source, project = Path(source).resolve(), Path(project).resolve()
    if project.exists() and not resume:
        raise FileExistsError('Output project already exists. Use --resume for this Kaggle import, or a new --project.')
    entries = _entries(source, block_size)
    config = _import_config(source, entries, block_size)
    with project_lock(project) as project:
        manifest = read_manifest(project)
        _initialize_import(manifest, config, resume)
        if manifest['import'].get('test_overlaps') and not exclude_overlapping_test:
            raise ValueError('Resume with --exclude-overlapping-test to retain the explicit test-duplicate policy.')
        return _import_entries(project, manifest, entries, source, checkpoint_every, progress, exclude_overlapping_test)
