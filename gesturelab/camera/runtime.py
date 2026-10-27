"""Local camera window, guided still-photo capture and continuous prediction.

Camera access only happens in run_camera(), after the user starts the window.
Scheduling, rendering and storage are separately testable without camera hardware.
"""
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from gesturelab.datasets.data import (Detector, add_image, atomic_json, decode_image, ensure_project,
                   overlay, project_lock, read_manifest)
from gesturelab.training.experiments import predict_bundle
from gesturelab.features.inspection import FEATURE_VERSION
from gesturelab.features.training_inputs import model_features, validate_model_input
from gesturelab.training.rules import UNKNOWN


def new_session_name():
    return datetime.now().strftime("session_%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:4]


def reference_for(project, label):
    """Prefer a chosen reference; otherwise only use included development photos."""
    project=Path(project)
    index=project/"references/index.json"
    mapping=json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    if label in mapping:
        path=(project/mapping[label]).resolve()
        if path.is_relative_to(project.resolve()) and path.is_file():
            return path,"Custom reference"
    for sample in read_manifest(project)["samples"]:
        if sample["label"]==label and sample["included"] and sample["role"]=="development" and sample["status"]=="ok":
            return project/sample["image_path"],"Existing development photo"
    return None,"Choose a reference photo for this label first"


def save_reference(project,label,data):
    label=label.strip()
    if not label or label==UNKNOWN:
        raise ValueError("Enter a valid label.")
    image=decode_image(data)
    with project_lock(project) as project:
        folder=project/"references"
        folder.mkdir(exist_ok=True)
        name=hashlib.sha256(label.encode()).hexdigest()[:24]+".png"
        image.save(folder/name)
        index=folder/"index.json"
        mapping=json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
        mapping[label]="references/"+name
        atomic_json(index,mapping)
    return folder/name


def load_reference(project,label):
    """Enlarge the hand in existing photos using its stored landmarks."""
    path,_=reference_for(project,label)
    if path is None:
        return None
    with Image.open(path) as source:
        image=source.convert("RGB").copy()
    for sample in read_manifest(project)["samples"]:
        if (Path(project)/sample["image_path"]).resolve()==path.resolve() and sample.get("image_landmarks"):
            xy=np.asarray(sample["image_landmarks"],dtype=float)[:,:2]*[image.width,image.height]
            low,high=xy.min(axis=0),xy.max(axis=0)
            margin=float(max(high-low))*.25
            box=(max(0,int(low[0]-margin)),max(0,int(low[1]-margin)),
                 min(image.width,int(high[0]+margin)),min(image.height,int(high[1]+margin)))
            if box[2]-box[0]>10 and box[3]-box[1]>10:
                image=image.crop(box)
            break
    return image


@dataclass
class CaptureSchedule:
    labels: list
    photos_per_label: int=3
    interval: float=3.0
    change_delay: float=5.0
    index: int=0
    paused: bool=True
    deadline: float | None=None

    def __post_init__(self):
        self.labels=[str(label).strip() for label in self.labels]
        if not self.labels or any(not label or label==UNKNOWN for label in self.labels):
            raise ValueError("Choose at least one valid gesture.")
        if len(set(self.labels))!=len(self.labels):
            raise ValueError("Use each label once in the list.")
        if not isinstance(self.photos_per_label,int) or not 1<=self.photos_per_label<=100:
            raise ValueError("Choose 1 to 100 photos per gesture.")
        if not all(math.isfinite(v) and 0.5<=v<=120 for v in (self.interval,self.change_delay)):
            raise ValueError("Times must be between 0.5 and 120 seconds.")

    @property
    def total(self):
        return len(self.labels)*self.photos_per_label

    @property
    def done(self):
        return self.index>=self.total

    @property
    def label(self):
        return self.labels[min(self.index//self.photos_per_label,len(self.labels)-1)]

    @property
    def shot(self):
        return self.index%self.photos_per_label+1

    @property
    def delay(self):
        return self.change_delay if self.index%self.photos_per_label==0 else self.interval

    def toggle_pause(self):
        self.paused=not self.paused
        self.deadline=None  # Always give a full countdown after resuming.

    def tick(self,now,hand_ok):
        if self.done:
            return False,"Done! All photos have been saved."
        if self.paused:
            return False,"Press SPACE to start / resume"
        if not hand_ok:
            self.deadline=None
            return False,"Show exactly one hand; countdown waiting"
        if self.deadline is None:
            self.deadline=now+self.delay
        remaining=max(0,self.deadline-now)
        return remaining<=0,f"Photo in {math.ceil(remaining)} s"

    def commit(self):
        if self.done:
            raise ValueError("Recording has already finished.")
        self.index+=1
        self.deadline=None


class StableLabel:
    def __init__(self,window=5):
        self.history=deque(maxlen=max(1,int(window)))

    def update(self,label):
        if label is None:
            self.history.clear()
            return None
        self.history.append(str(label))
        winner,count=Counter(self.history).most_common(1)[0]
        return winner if count>len(self.history)/2 else None


def save_camera_photo(project,image,result,label,session,participant="participant1",condition="baseline",role="development"):
    if result.get("status")!="ok" or result.get("feature_version")!=FEATURE_VERSION:
        raise ValueError("No usable hand detection for this photo.")
    buffer=BytesIO()
    image.convert("RGB").save(buffer,format="PNG")
    return add_image(project,buffer.getvalue(),datetime.now().strftime("webcam_%Y%m%d_%H%M%S_%f.png"),
                     label,session,participant,condition,role,
                     {"type":"guided_webcam","captured_at":datetime.now(timezone.utc).isoformat(),
                      "mirrored_input":False},analysis=result)


@lru_cache(maxsize=24)
def font(size):
    for name in ("DejaVuSans.ttf","Arial.ttf","/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit_text(draw,text,xy,width,size=23,fill="#eaf2ff"):
    text=str(text)
    face=font(size)
    while text and draw.textlength(text,font=face)>width:
        text=text[:-1]
    draw.text(xy,text,font=face,fill=fill)


def camera_canvas(frame,result,reference=None,title="",subtitle="",message="",details=(),mirror=True,progress=None):
    """Reference on the left, live view on the right; raw training input stays unmirrored."""
    canvas=Image.new("RGB",(1200,800),"#101e32")
    draw=ImageDraw.Draw(canvas)
    fit_text(draw,title,(32,20),1130,32)
    fit_text(draw,subtitle,(32,66),1130,18,fill="#a9bfd8")
    draw.rounded_rectangle((24,107,452,530),18,fill="#20334d")
    draw.rounded_rectangle((472,107,1176,635),18,fill="#20334d")
    if reference is not None:
        ref=ImageOps.mirror(reference) if mirror else reference
        view=ImageOps.contain(ref.convert("RGB"),(404,385))
        canvas.paste(view,(238-view.width//2,320-view.height//2))
    else:
        fit_text(draw,"Live recognition",(48,185),375,26)
        fit_text(draw,"Trained model prediction",(48,228),375,22)
    preview=overlay(frame,result)
    if mirror:
        preview=ImageOps.mirror(preview)
    preview=ImageOps.contain(preview,(680,504))
    canvas.paste(preview,(824-preview.width//2,371-preview.height//2))
    draw=ImageDraw.Draw(canvas)
    for i,line in enumerate(details[:3]):
        fit_text(draw,line,(35,547+i*28),420,19)
    fit_text(draw,message,(32,657),1130,27,fill="#6ce4c7")
    if progress is not None:
        draw.rounded_rectangle((32,710,1168,720),5,fill="#30445c")
        if progress>0:
            draw.rounded_rectangle((32,710,32+int(1136*min(1,progress)),720),5,fill="#34bfa4")
    fit_text(draw,"SPACE pause/start    + / - speed    R restart countdown    Q / ESC close",(32,750),1130,17)
    return canvas


@dataclass
class _CameraView:
    reference: object
    heading: str
    subtitle: str
    message: str
    details: list
    progress: object
    state: str


class _CameraSession:
    """Own camera resources and mutable capture state for one local window."""

    def __init__(self, config, cv2, status_path, stop_path):
        self.config, self.cv2 = config, cv2
        self.project = ensure_project(config['project'])
        self.mode = config.get('mode', 'capture')
        if self.mode not in ('capture', 'predict'):
            raise ValueError('Choose capture or predict.')
        self.status_path, self.stop_path = status_path, stop_path
        self.title = 'Hand-gesturelab - ' + ('Guided capture' if self.mode == 'capture' else 'Live recognition')
        self.camera = self.detector = self.bundle = self.schedule = None
        self.references = {}
        self.saved, self.last_status, self.flash_until = 0, 0.0, 0
        self.stable_ml = StableLabel(config.get('smoothing', 5))
        self.result = {'status': 'pending'}
        self.last_inference = -math.inf
        self.interval = 1 / max(1, min(30, float(config.get('inference_hz', 10))))
        self.live_paused = False
        self.ml = self.ml_stable = None

    def publish(self, state, **extra):
        if self.status_path:
            atomic_json(self.status_path, {
                'state': state, 'mode': self.mode, 'saved': self.saved,
                'session': self.config.get('session'), **extra,
            })

    def prepare(self):
        if self.mode == 'capture':
            self._prepare_capture()
        else:
            if self.config.get('model'):
                self.bundle = joblib.load(self.config['model'])
                validate_model_input(self.bundle)
            if self.bundle is None:
                raise ValueError('Select a trained model for live recognition.')
            if self.bundle.get('config', {}).get('model') == 'rule_based':
                raise ValueError('Select a trained ML model for live recognition.')

    def _prepare_capture(self):
        config = self.config
        self.schedule = CaptureSchedule(config['labels'], int(config.get('photos_per_label', 3)),
                                        float(config.get('interval', 3)), float(config.get('change_delay', 5)))
        config.setdefault('session', new_session_name())
        self.participant = config.get('participant', 'participant1').strip()
        if not self.participant:
            raise ValueError('Enter a participant.')
        if any(s['participant'] == self.participant and s['session'] == config['session']
               for s in read_manifest(self.project)['samples']):
            raise ValueError('This session already exists. Start a new recording round with a new session name.')
        for label in self.schedule.labels:
            reference = load_reference(self.project, label)
            if reference is None:
                raise ValueError(f'Choose a reference photo first for {label}.')
            self.references[label] = reference

    def open(self):
        cv2 = self.cv2
        self.detector = Detector()
        self.camera = cv2.VideoCapture(int(self.config.get('camera_index', 0)))
        if not self.camera.isOpened():
            raise RuntimeError('Webcam unavailable. Close other camera apps, try camera 1, or grant Terminal camera access in system settings.')
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1200, 800)

    def _read_frame(self):
        ok, bgr = self.camera.read()
        if not ok:
            raise RuntimeError('No camera frame received. Check the camera and restart.')
        return Image.fromarray(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))

    def _analyze_frame(self, frame, now):
        if self.mode != 'capture' and (self.live_paused or now - self.last_inference < self.interval):
            return
        self.result = self.detector.analyze(frame)
        self.last_inference = time.monotonic()
        if self.mode == 'predict':
            self.ml = None
            if self.result['status'] == 'ok':
                sample = {**self.result, 'width': frame.width, 'height': frame.height}
                try:
                    values = model_features(self.bundle, sample)
                except ValueError as error:
                    self.result.update(status='invalid_features', error=str(error))
                else:
                    self.ml = str(predict_bundle(self.bundle, [values])[0])
            self.ml_stable = self.stable_ml.update(self.ml)

    def _save_photo(self, frame, label):
        _, created = save_camera_photo(
            self.project, frame, self.result, label, self.config['session'], self.participant,
            self.config.get('condition', 'baseline'), self.config.get('role', 'development'),
        )
        if created:
            self.saved += 1
            self.schedule.commit()
            self.flash_until = time.monotonic() + .7
            return 'Photo saved! Reset your hand pose.'
        self.schedule.deadline = None
        return 'Identical image; move your hand and try again.'

    def _capture_view(self, frame, now):
        schedule, config = self.schedule, self.config
        label, shot = schedule.label, schedule.shot
        reference = self.references[label]
        ready, message = schedule.tick(now, self.result['status'] == 'ok')
        if ready:
            message = self._save_photo(frame, label)
        if time.monotonic() < self.flash_until:
            message = 'Photo saved! Reset your hand pose.'
        if schedule.done:
            message = f'Done! {self.saved} photos saved. Close with Q.'
        heading = f'Copy this pose: {label}'
        subtitle = f"Photo {min(shot, schedule.photos_per_label)}/{schedule.photos_per_label}  |  {config['session']}"
        details = [f'Saved: {self.saved}/{schedule.total}', f'Between photos: {schedule.interval:.1f} s',
                   'Reference and camera mirrored' if config.get('mirror', True) else 'Reference and camera unmirrored']
        state = 'complete' if schedule.done else ('paused' if schedule.paused else 'running')
        return _CameraView(reference, heading, subtitle, message, details, self.saved / schedule.total, state)

    def _prediction_view(self):
        raw = self.ml or 'no hand'
        subtitle = 'PRACTICE MODEL - not evaluated' if self.bundle.get('preview') else 'Predictions on new camera images'
        if self.bundle.get('synthetic'):
            subtitle += ' | SYNTHETIC MODEL'
        details = [f"Prediction: {self.ml_stable or 'changing'}", f'Raw prediction: {raw}']
        if self.live_paused:
            message = 'Paused; SPACE resumes recognition'
        elif self.result['status'] != 'ok':
            message = {'no_hand': 'No hand found', 'multiple_hands': 'Show one hand'}.get(self.result['status'], 'Unusable hand landmarks')
            details = ['No prediction']
        else:
            message = f"Prediction: {self.ml_stable or 'changing'}"
        return _CameraView(None, 'Live recognition', subtitle, message, details, None,
                           'paused' if self.live_paused else 'running')

    def _display(self, frame, view, now):
        canvas = camera_canvas(frame, self.result, view.reference, view.heading, view.subtitle,
                               view.message, view.details, self.config.get('mirror', True), view.progress)
        self.cv2.imshow(self.title, self.cv2.cvtColor(np.asarray(canvas), self.cv2.COLOR_RGB2BGR))
        if now - self.last_status > .5:
            label = self.schedule.label if self.schedule else self.ml_stable
            self.publish(view.state, message=view.message, label=label)
            self.last_status = now

    def _handle_key(self, key):
        if key == ord(' '):
            if self.schedule:
                self.schedule.toggle_pause()
            else:
                self.live_paused = not self.live_paused
                if self.live_paused:
                    self.result = {'status': 'pending'}
                    self.ml = self.ml_stable = None
                    self.stable_ml.update(None)
        if self.schedule and key in (ord('+'), ord('='), ord('-')):
            interval = self.schedule.interval + (.5 if key in (ord('+'), ord('=')) else -.5)
            self.schedule.interval = max(.5, min(120, interval))
            self.schedule.deadline = None
        if self.schedule and key in (ord('r'), ord('R')):
            self.schedule.deadline = None

    def run(self):
        while not (self.stop_path and Path(self.stop_path).exists()):
            frame = self._read_frame()
            self._analyze_frame(frame, time.monotonic())
            now = time.monotonic()
            view = self._capture_view(frame, now) if self.schedule else self._prediction_view()
            self._display(frame, view, now)
            key = self.cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or self.cv2.getWindowProperty(self.title, self.cv2.WND_PROP_VISIBLE) < 1:
                break
            self._handle_key(key)

    def close(self):
        if self.camera is not None:
            self.camera.release()
        if self.detector is not None:
            self.detector.close()
        self.cv2.destroyAllWindows()


def run_camera(config, status_path=None, stop_path=None):
    """Run a local camera session and always release its hardware resources."""
    import cv2
    session = _CameraSession(config, cv2, status_path, stop_path)
    try:
        session.publish('starting', message='Starting camera and hand model...')
        session.prepare()
        session.open()
        session.publish('running', message='Window opened. Click inside the camera window to use the keyboard controls.')
        session.run()
        message = (f'Camera closed. {session.saved} photos saved.' if session.schedule
                   else 'Camera closed; no test images saved.')
        session.publish('finished', message=message)
    except Exception as error:
        session.publish('error', message=str(error))
        raise
    finally:
        session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--status")
    parser.add_argument("--stop")
    args=parser.parse_args()
    try:
        run_camera(json.loads(Path(args.config).read_text(encoding="utf-8")),args.status,args.stop)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
