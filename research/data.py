"""Single Kaggle dataset persistence and default MediaPipe extraction."""
from contextlib import contextmanager
from datetime import datetime,timezone
from io import BytesIO
import hashlib,json,os,sys,tempfile
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from PIL import Image,ImageOps
from research.finger_pilot import compute_finger_features
from research.landmarks import landmark_features
MODEL_URL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_SHA256='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
FEATURE_VERSION='geometry-v2.0'
def default_rules():return {}
def extract_features(raw,world,width,height):
    features={k:v for k,v in landmark_features(raw,world,width,height).items() if not k.startswith('world_')}
    features.update(compute_finger_features(world))
    if not np.isfinite(list(features.values())).all():raise ValueError('Invalid geometric features.')
    return features


def atomic_json(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, indent=None if compact else 2, ensure_ascii=False, allow_nan=False)
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

def model_path():
    path = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
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

def _validate_detector_thresholds(min_hand_detection_confidence, min_hand_presence_confidence):
    for name, value in (("min_hand_detection_confidence", min_hand_detection_confidence),
                        ("min_hand_presence_confidence", min_hand_presence_confidence)):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

class Detector:
    def __init__(self, min_hand_detection_confidence=.5, min_hand_presence_confidence=.5):
        _validate_detector_thresholds(min_hand_detection_confidence, min_hand_presence_confidence)
        import mediapipe as mp
        self.mp = mp
        self.thresholds = {
            "min_hand_detection_confidence": min_hand_detection_confidence,
            "min_hand_presence_confidence": min_hand_presence_confidence,
        }
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path())),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence)
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(self.options)

    def analyze(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        result = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb))
        n = len(result.hand_landmarks)
        base = {"n_hands": n, "image_size": [image.width,image.height], "mediapipe_version": self.mp.__version__,
                "feature_version": FEATURE_VERSION, "detector_thresholds": self.thresholds}
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

def _analyze_pending(project, manifest, pending, detector, progress, checkpoint_every, cache_features):
    completed, saved = 0, 0
    try:
        for sample in pending:
            with Image.open(project / sample['image_path']) as image:
                analysis = detector.analyze(image)
            if not cache_features:
                analysis.pop('features', None)
            sample.update(analysis)
            completed += 1
            if completed % checkpoint_every == 0:
                atomic_json(project / 'manifest.json', manifest, compact=checkpoint_every > 1)
                saved = completed
            if progress:
                progress(completed / len(pending))
    finally:
        if completed != saved:
            atomic_json(project / 'manifest.json', manifest, compact=checkpoint_every > 1)

def analyze_project(project, progress=None, checkpoint_every=1, cache_features=True,
                    min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                    reprocess=False):
    """Analyze pending inputs, or every input when reprocess is requested."""
    if not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError('Checkpoint size must be a positive integer.')
    _validate_detector_thresholds(min_hand_detection_confidence, min_hand_presence_confidence)
    with project_lock(project) as project:
        manifest = read_manifest(project)
        if manifest.get('import', {}).get('complete') is False:
            raise ValueError('Finish or resume the dataset import before analyzing images.')
        pending = [s for s in manifest["samples"]
                   if reprocess or s["status"] == "pending"]
        if not pending:
            return 0
        detector = Detector(min_hand_detection_confidence, min_hand_presence_confidence)
        try:
            _analyze_pending(project, manifest, pending, detector, progress, checkpoint_every, cache_features)
        finally:
            detector.close()
    return len(pending)
