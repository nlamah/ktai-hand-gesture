"""Every sample retains its original bytes, exact RGB input, label and provenance."""
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import os
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

from gesturelab.features.inspection import FEATURE_VERSION, extract_features
from gesturelab.training.rules import default_rules, UNKNOWN

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temp.replace(path)


def read_manifest(project):
    path = Path(project) / "manifest.json"
    if not path.exists():
        return {"schema": 2, "samples": []}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 2:
        raise ValueError("This project has an unknown dataset format.")
    return manifest


def ensure_project(project):
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    for folder in ("originals", "images", "sources", "reports"):
        (project / folder).mkdir(exist_ok=True)
    if not (project / "manifest.json").exists():
        atomic_json(project / "manifest.json", {"schema": 2, "samples": []})
    if not (project / "rules.json").exists():
        atomic_json(project / "rules.json", default_rules())
    return project


@contextmanager
def project_lock(project):
    """OS lock: automatically released if a process exits or crashes."""
    project = ensure_project(project)
    lock = project / ".write.lock"
    handle = lock.open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            if lock.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ValueError("The project is already being updated. Try again when that operation finishes.") from error
    try:
        yield project
    finally:
        handle.close()


def decode_image(data):
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
        return image


def _validate_image_metadata(label, session, participant, role):
    if not str(label).strip() or label == UNKNOWN or not str(session).strip() or not str(participant).strip():
        raise ValueError("Enter a label, session and participant. __unknown__ is a reserved label.")
    if role not in ("development", "test"):
        raise ValueError("Choose development or test.")


def _existing_image(samples, pixel_hash, label):
    for previous in samples:
        if previous['pixel_sha256'] == pixel_hash:
            if previous['label'] != label:
                raise ValueError(f"This photo already exists with label {previous['label']!r}. Correct that label first.")
            return previous
    return None


def _validate_recording_group(samples, participant, session, role, provenance):
    for previous in samples:
        if provenance and provenance.get('type') == 'video_frame':
            old = previous.get('provenance', {})
            if old.get('source_sha256') == provenance.get('source_sha256') and (
                previous['participant'], previous['session'], previous['role']
            ) != (participant, session, role):
                raise ValueError('All frames from the same source video must keep the same participant, session and role.')
        if (previous['participant'], previous['session']) == (participant, session) and previous['role'] != role:
            raise ValueError('One recording session cannot be both development and test.')


def _store_image(project, data, filename, image, pixel_hash):
    sample_id = pixel_hash[:24]
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'):
        suffix = '.image'
    original_path = f'originals/{sample_id}{suffix}'
    input_path = f'images/{sample_id}.png'
    (project / original_path).write_bytes(data)
    image.save(project / input_path, format='PNG')
    return {
        'id': sample_id, 'filename': Path(filename).name,
        'original_path': original_path, 'image_path': input_path,
        'original_sha256': hashlib.sha256(data).hexdigest(), 'pixel_sha256': pixel_hash,
        'width': image.width, 'height': image.height, 'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def _attach_analysis(sample, analysis):
    if analysis is not None:
        keys = ('status', 'n_hands', 'mediapipe_version', 'feature_version', 'image_landmarks',
                'world_landmarks', 'features', 'handedness', 'handedness_score', 'error')
        sample.update({key: analysis[key] for key in keys if key in analysis})


def add_image(project, data, filename, label, session, participant='participant1', condition='baseline',
              role='development', provenance=None, analysis=None):
    """Preserve source bytes and analysis pixels, deduplicating within the write lock."""
    _validate_image_metadata(label, session, participant, role)
    image = decode_image(data)
    pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    with project_lock(project) as project:
        manifest = read_manifest(project)
        previous = _existing_image(manifest['samples'], pixel_hash, label)
        if previous is not None:
            return previous, False
        _validate_recording_group(manifest['samples'], participant, session, role, provenance)
        sample = _store_image(project, data, filename, image, pixel_hash)
        sample.update(label=label, session=session, participant=participant, condition=condition,
                      role=role, included=True, provenance=provenance or {'type': 'photo'})
        _attach_analysis(sample, analysis)
        manifest['samples'].append(sample)
        atomic_json(project / 'manifest.json', manifest)
    return sample, True


def metadata_table(project):
    records = read_manifest(project)["samples"]
    keys = ["id","filename","created_at","label","session","participant","condition","role","included","status"]
    return pd.DataFrame([{k:s.get(k) for k in keys} for s in records], columns=keys)


def edit_metadata(project, edited):
    mutable = ["label","session","participant","condition","role","included"]
    with project_lock(project) as project:
        manifest = read_manifest(project)
        changes = {str(r["id"]):r for r in edited.to_dict("records")}
        for sample in manifest["samples"]:
            if sample["id"] in changes:
                for key in mutable:
                    value = changes[sample["id"]][key]
                    sample[key] = bool(value) if key == "included" else str(value).strip()
                if any(not sample[k] for k in ["label","session","participant"]) or sample["label"] == UNKNOWN:
                    raise ValueError("Label, session and participant must remain populated.")
                if sample["role"] not in ("development","test"):
                    raise ValueError("Invalid role.")
        seen = {}
        for sample in manifest["samples"]:
            if sample["included"]:
                key = (sample["participant"],sample["session"])
                if key in seen and seen[key] != sample["role"]:
                    raise ValueError("A session occurs in both development and test.")
                seen[key] = sample["role"]
        atomic_json(project / "manifest.json", manifest)


def model_path():
    path = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise ValueError("The model download has an unexpected checksum.")
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("The model file is damaged or replaced. Remove it and download it again.")
    return path


class Detector:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path())),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=.5, min_hand_presence_confidence=.5)
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(self.options)

    def analyze(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        result = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb))
        n = len(result.hand_landmarks)
        base = {"n_hands": n, "mediapipe_version": self.mp.__version__, "feature_version": FEATURE_VERSION}
        if n != 1:
            return {**base, "status": "no_hand" if n == 0 else "multiple_hands"}
        raw = [[p.x,p.y,p.z] for p in result.hand_landmarks[0]]
        world = [[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]]
        try:
            features = extract_features(raw,world,image.width,image.height)
        except ValueError as error:
            return {**base,"status":"invalid_geometry","error":str(error),"image_landmarks":raw,"world_landmarks":world}
        handed = result.handedness[0][0]
        return {**base,"status":"ok","image_landmarks":raw,"world_landmarks":world,"features":features,
                "handedness":handed.category_name,"handedness_score":float(handed.score)}

    def close(self):
        self.model.close()


def analyze_project(project, progress=None):
    with project_lock(project) as project:
        manifest = read_manifest(project)
        pending = [s for s in manifest["samples"] if s["status"] == "pending"]
        if not pending:
            return 0
        detector = Detector()
        try:
            for i, sample in enumerate(pending):
                with Image.open(project / sample["image_path"]) as image:
                    sample.update(detector.analyze(image))
                atomic_json(project / "manifest.json",manifest)
                if progress:
                    progress((i+1)/len(pending))
        finally:
            detector.close()
    return len(pending)


def analysis_table(project):
    rows=[]
    for sample in read_manifest(project)["samples"]:
        if sample["included"] and sample["status"] == "ok":
            if sample.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Outdated features found. Recompute the dataset with this version.")
            metadata = {k:sample[k] for k in ("id","label","session","participant","condition","role","image_path")}
            rows.append({**metadata, **sample["features"]})
    return pd.DataFrame(rows)


def fingerprint(project):
    return hashlib.sha256(json.dumps(read_manifest(project),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def overlay(image, sample):
    result=image.convert("RGB").copy()
    points=sample.get("image_landmarks")
    if not points:
        return result
    draw=ImageDraw.Draw(result)
    xy=[(p[0]*image.width,p[1]*image.height) for p in points]
    for a,b in EDGES:
        draw.line([xy[a],xy[b]],fill=(41,205,173),width=max(2,image.width//200))
    radius=max(3,image.width//130)
    for x,y in xy:
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=(255,180,60))
    return result


def _export_features(project, manifest):
    # Export current calculations, not stale cached flags from earlier thresholds.
    from gesturelab.features.inspection import features_for_sample
    from gesturelab.features.audit import load_thresholds
    thresholds=load_thresholds(project)
    rows=[]
    for sample in manifest['samples']:
        if sample['included'] and sample['status']=='ok':
            metadata={k:sample[k] for k in ('id','label','session','participant','condition','role','image_path')}
            rows.append({**metadata,**features_for_sample(sample,thresholds)})
    table=pd.DataFrame(rows)
    return table, thresholds


def _archive_dataset_files(archive, project, destination):
    for name in ("originals","images","sources","reports","references"):
        folder=project/name
        if folder.exists():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(project) and path.resolve()!=destination:
                    archive.write(path,"dataset/"+str(path.relative_to(project)))
    if (project/"rules.json").exists():
        archive.write(project/"rules.json","dataset/rules.json")
    for filename in ("labels.json","feature_checks.json"):
        if (project/filename).exists():
            archive.write(project/filename,"dataset/"+filename)


def export_project(project, destination):
    project = Path(project).resolve()
    destination = Path(destination).resolve()
    manifest = read_manifest(project)
    table, thresholds = _export_features(project, manifest)
    with ZipFile(destination,"w",ZIP_DEFLATED) as archive:
        archive.writestr("dataset/manifest.json",json.dumps(manifest,indent=2,ensure_ascii=False))
        archive.writestr("dataset/features.csv",table.to_csv(index=False))
        archive.writestr("dataset/README.txt",
            'originals: unchanged source files; images: exact RGB inputs for MediaPipe.\nmanifest.json contains labels, sessions, conditions, hashes, detection status and landmarks.\nfeatures.csv contains current features for recorded images with successful detection.\nThese are recomputed from landmarks using the included finger_thresholds.json.\nThe manifest also retains originally cached features.\nreports contains any results from the earlier app. Include the separate experiments directory yourself.\nExtract this ZIP and open the dataset directory as a project to continue.\n')
        archive.writestr('dataset/finger_thresholds.json',json.dumps(thresholds,indent=2))
        _archive_dataset_files(archive, project, destination)
    return destination
