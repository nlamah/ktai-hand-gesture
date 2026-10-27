"""Import single-letter COCO images while preserving split and source provenance."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from gesturelab.datasets.data import add_image, decode_image, read_manifest

SPLIT_ROLES = {'train': 'development', 'valid': 'development', 'test': 'test'}


def _image_entries(annotation_file, split):
    document = json.loads(annotation_file.read_text())
    categories = {row['id']: row['name'].strip().upper() for row in document['categories']}
    annotations = defaultdict(list)
    for row in document['annotations']:
        annotations[row['image_id']].append(row)
    for image in document['images']:
        rows = annotations[image['id']]
        if len(rows) != 1:
            raise ValueError(f'{split}: each image must have exactly one A–Z annotation.')
        label = categories.get(rows[0]['category_id'], '')
        if len(label) != 1 or label not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            raise ValueError(f'{split}: expected a single letter category, found {label!r}.')
        source = (annotation_file.parent / image['file_name']).resolve()
        if not source.is_relative_to(annotation_file.parent.resolve()) or not source.is_file():
            raise ValueError(f'{split}: missing or invalid image path: {image["file_name"]!r}.')
        yield {'source': source, 'label': label, 'split': split, 'annotation': rows[0],
               'source_group': 'coco:' + Path(image['file_name']).name.split('.rf.', 1)[0]}


def _validated_entries(source):
    entries, groups, pixels = [], {}, {}
    for split, role in SPLIT_ROLES.items():
        annotations = source / split / '_annotations.coco.json'
        if not annotations.is_file():
            raise ValueError(f'Missing COCO annotations for {split}.')
        for entry in _image_entries(annotations, split):
            group = entry['source_group']
            if group in groups and groups[group] != (role, entry['label']):
                raise ValueError('A source image occurs with conflicting labels or across development and test.')
            groups[group] = (role, entry['label'])
            image = decode_image(entry['source'].read_bytes())
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest in pixels and pixels[digest] != (role, entry['label']):
                raise ValueError('Identical pixels occur with conflicting labels or across development and test.')
            pixels[digest] = (role, entry['label'])
            entries.append(entry)
    if not entries:
        raise ValueError('No annotated letter images found.')
    return entries


def _import_entry(entry, source, project):
    split = entry['split']
    annotation = entry['annotation']
    provenance = {
        'type': 'coco', 'language': 'ASL', 'dataset_split': split,
        'source_group': entry['source_group'],
        'source_file': entry['source'].relative_to(source).as_posix(),
        'annotation_id': annotation['id'], 'bbox': annotation.get('bbox'),
        'annotations': 'sources/coco_' + split + '.json',
    }
    return add_image(project, entry['source'].read_bytes(), entry['source'].name,
                     entry['label'], 'coco_' + split, participant='unknown',
                     condition='external_asl', role=SPLIT_ROLES[split], provenance=provenance)[1]


def import_coco_dataset(source, project, progress=None):
    """Validate the full input before creating a new, separate Gesture Lab dataset."""
    source, project = Path(source).resolve(), Path(project).resolve()
    if project.exists():
        raise FileExistsError('Output project already exists. Choose a new --project directory.')
    entries = _validated_entries(source)
    added = 0
    for index, entry in enumerate(entries):
        added += int(_import_entry(entry, source, project))
        if progress:
            progress(index + 1, len(entries))
    folder = project / 'sources'
    for split in SPLIT_ROLES:
        shutil.copyfile(source / split / '_annotations.coco.json', folder / ('coco_' + split + '.json'))
    if (source / 'README.roboflow.txt').is_file():
        shutil.copyfile(source / 'README.roboflow.txt', folder / 'README.roboflow.txt')
    return {'added': added, 'duplicates': len(entries) - added,
            'total': len(read_manifest(project)['samples'])}
