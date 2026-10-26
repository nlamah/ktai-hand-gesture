"""Use individual reference images; never add illustrations as training data."""
import json
from pathlib import Path
import shutil

from gesturelab.datasets.data import ensure_project, project_lock, atomic_json

ALPHABET_IMAGES = Path(__file__).resolve().parents[2] / 'assets' / 'alphabet'
LETTERS = sorted(path.stem for path in ALPHABET_IMAGES.glob('*.png'))


def initialize_alphabet(project):
    """Copy supplied references while preserving custom reference selections."""
    project = ensure_project(project)
    with project_lock(project):
        folder = project / 'references/ngt_alphabet'
        folder.mkdir(parents=True, exist_ok=True)
        index = project / 'references/index.json'
        mapping = json.loads(index.read_text()) if index.exists() else {}
        for letter in LETTERS:
            destination = folder / (letter + '.png')
            if not destination.exists():
                shutil.copyfile(ALPHABET_IMAGES / (letter + '.png'), destination)
            if letter not in mapping or mapping[letter].startswith('references/alphabet/'):
                mapping[letter] = 'references/ngt_alphabet/' + letter + '.png'
        atomic_json(index, mapping)
    return project
