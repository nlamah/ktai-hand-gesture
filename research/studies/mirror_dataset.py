"""Detect every original and horizontal RGB mirror with one frozen MediaPipe recipe."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from PIL import Image

from research.data import Detector, atomic_json, model_path
from research.live_model import detection_inputs
from research.mirroring import VERSION, mirror_image, pixel_hash

_DETECTOR = None


def extract_chunk(task):
    global _DETECTOR
    project, path, samples, protocol_hash = task
    path = Path(path)
    if path.exists():
        with np.load(path) as saved:
            if str(saved['protocol_hash']) != protocol_hash or saved['ids'].tolist() != [s['id'] for s in samples]:
                raise ValueError('Extraction checkpoint does not match this protocol.')
        return len(samples), True
    if _DETECTOR is None:
        _DETECTOR = Detector(.5, .5)
    n = len(samples)
    features = np.full((n, 2, 835), np.nan, dtype=np.float32)
    image_points = np.full((n, 2, 21, 3), np.nan, dtype=np.float32)
    world_points = np.full_like(image_points, np.nan)
    statuses = np.full((n, 2), '', dtype='<U24')
    hands = np.full((n, 2), '', dtype='<U5')
    hand_scores = np.full((n, 2), np.nan, dtype=np.float32)
    hashes = np.full((n, 2), '', dtype='<U64')
    for i, sample in enumerate(samples):
        with Image.open(Path(project) / sample['image_path']) as source:
            original = source.convert('RGB')
        if pixel_hash(original) != sample['pixel_sha256']:
            raise ValueError(f"Source pixels changed: {sample['id']}")
        for view, image in enumerate((original, mirror_image(original))):
            hashes[i, view] = pixel_hash(image)
            result = _DETECTOR.analyze(image)
            statuses[i, view] = result['status']
            hands[i, view] = result.get('handedness', '')
            hand_scores[i, view] = result.get('handedness_score', np.nan)
            if 'image_landmarks' in result:
                image_points[i, view] = result['image_landmarks']
                world_points[i, view] = result['world_landmarks']
            if result['status'] == 'ok':
                row = detection_inputs(result)[0]
                if np.isfinite(row).all():
                    features[i, view] = row
                else:
                    statuses[i, view] = 'invalid_model_geometry'
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, ids=[s['id'] for s in samples],
                        labels=[s['label'] for s in samples], groups=[s['session'] for s in samples],
                        features=features, image_landmarks=image_points, world_landmarks=world_points,
                        status=statuses, handedness=hands, handedness_score=hand_scores,
                        pixel_sha256=hashes, protocol_hash=protocol_hash)
    temporary.replace(path)
    return n, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('data/asl_kaggle'))
    parser.add_argument('--work', type=Path, default=Path('runs/mirror_01'))
    parser.add_argument('--out', type=Path, default=Path('results/mirror_01'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        parser.error('Workers and chunk size must be positive.')
    root = Path.cwd()
    manifest_bytes = (args.project / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    keys = ('id', 'label', 'session', 'image_path', 'pixel_sha256')
    samples = [{k: s[k] for k in keys} for s in manifest['samples']
               if s.get('included') and s.get('role') == 'development' and s['label'] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    del manifest
    if len(samples) != 78000 or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Expected all 78,000 distinct A-Z source images.')
    source_files = ('research/mirroring.py', 'research/studies/mirror_dataset.py', 'research/data.py',
                    'research/live_model.py', 'research/shin.py', 'research/geometry.py', 'research/inputs.py')
    config = {
        'version': VERSION, 'source_images': len(samples), 'views': ['original', 'mirrored_rgb'],
        'manifest_sha256': sha256(manifest_bytes).hexdigest(),
        'detector_sha256': sha256(model_path().read_bytes()).hexdigest(),
        'mode': 'IMAGE', 'num_hands': 2, 'detection_confidence': .5, 'presence_confidence': .5,
        'transformation': 'PIL horizontal RGB reflection at unchanged resolution; no interpolation',
        'landmarks': 'Redetected independently for both views; anatomical IDs unchanged; use returned handedness',
        'features': '820 Shin image-coordinate inputs plus 15 world-joint measurements, identical to live encoding',
        'chunk_size': args.chunk_size,
        'versions': {p: importlib.metadata.version(p) for p in ('mediapipe', 'numpy', 'pillow', 'scikit-learn')},
        'source_sha256': {p: sha256((root / p).read_bytes()).hexdigest() for p in source_files},
    }
    del manifest_bytes
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = args.work / 'detections'
    chunks.mkdir(parents=True, exist_ok=True)
    config_path = args.out / 'extraction_protocol.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Extraction configuration changed; choose fresh paths.')
    atomic_json(config_path, config)
    config_hash = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    tasks = [(str(args.project.resolve()), str(chunks / f'chunk_{start // args.chunk_size:04d}.npz'),
              samples[start:start + args.chunk_size], config_hash)
             for start in range(0, len(samples), args.chunk_size)]
    started, completed = time.monotonic(), 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(extract_chunk, task) for task in tasks]
        for future in as_completed(futures):
            n, reused = future.result()
            completed += n
            print(f'{completed}/{len(samples)} source pairs; {time.monotonic()-started:.1f}s; cached={reused}', flush=True)
    atomic_json(args.out / 'extraction_complete.json', {'source_images': completed, 'detector_inputs': 2*completed,
                'chunks': len(tasks), 'seconds': time.monotonic()-started, 'protocol_hash': config_hash})


if __name__ == '__main__':
    main()
