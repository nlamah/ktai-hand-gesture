"""Native webcam surfaces for protocol capture and synchronized A/B development."""
import time
import numpy as np

from PIL import Image, ImageDraw, ImageOps

from gesturelab.camera.runtime import camera_canvas, fit_text
from gesturelab.datasets.data import Detector, overlay
from gesturelab.research.live import LivePair
from gesturelab.research.models import ModelHandle
from gesturelab.research.recording import CaptureSchedule, detector_configuration, save_attempt, validate_capture, resume_schedule
from gesturelab.validation.report import sha256


def _camera(cv2, index, title):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError('Webcam unavailable; check the camera index and OS camera permission.')
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 800)
    return camera


def _frame(cv2, camera):
    good, bgr = camera.read()
    if not good:
        raise RuntimeError('No current camera frame received; no old frame is saved as a new attempt.')
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _reference(protocol, label):
    spec = protocol['references'][label]
    if spec['alphabet'] != protocol['alphabet'] or sha256(spec['file']) != spec['sha256']:
        raise ValueError('Reference alphabet/source bytes differ from the saved protocol.')
    with Image.open(spec['file']) as source:
        return (source.crop(spec['crop']) if spec.get('crop') else source.copy()).convert('RGB')


def _capture_view(frame, detection, protocol, schedule, now, show_landmarks):
    trial = schedule.current
    if trial is None:
        return camera_canvas(frame, detection if show_landmarks else {}, title='Protocol complete',
            message='All planned attempts stored. R: explicit retake of last attempt. Q: stop.', progress=1, mirror=False)
    left = _reference(protocol, trial['label'])
    timing = max(0, schedule.deadline - now) if schedule.deadline is not None and not schedule.paused else None
    status = ('Paused: adjust the nominal condition; SPACE starts.' if schedule.paused
              else f'Capture in {timing:.1f}s. Relax your hand and form the gesture again.')
    title = f"{protocol['alphabet']} | {protocol['role']} capture | letter {trial['label']}"
    subtitle = f"{trial['condition']} | nominal {trial['nominal_angle']}° | repetition {trial['repetition']} | {protocol['angle_method']}"
    canvas = camera_canvas(frame, detection if show_landmarks else {}, left, title, subtitle, '',
        [f"Planned slots: {schedule.index}/{len(protocol['trials'])}", 'Predictions are hidden in protocol capture.',
         'Unmirrored view; nominal angle is not measured.'], mirror=False, progress=schedule.index / len(protocol['trials']))
    draw = ImageDraw.Draw(canvas)
    fit_text(draw, status, (32, 650), 1130, 22, fill='#6ce4c7')
    help_text = 'No hand detected. Keep your full hand visible and ensure sufficient light.' if detection.get('status') == 'no_hand' else (
        'Every planned attempt is saved, including detection failures. Relax and reform the gesture.' if detection.get('status') == 'ok' else 'No usable single-hand detection. The planned attempt will still be saved.')
    fit_text(draw, help_text, (32, 683), 1130, 19, fill='#a9bfd8')
    draw.rectangle((20, 738, 1180, 795), fill='#101e32')
    fit_text(draw, 'SPACE start/pause | R explicit retake | L landmarks | Q stop | failures are saved', (32, 750), 1130, 17)
    return canvas


def run_protocol_capture(protocol_file, project, seal=None, resume=False):
    import cv2
    protocol, digest = validate_capture(protocol_file, project, seal, resume)
    for label in protocol['letters']:
        _reference(protocol, label)
    schedule, detector, camera = CaptureSchedule(protocol), None, None
    if resume: resume_schedule(schedule, project)
    show_landmarks, title = True, 'Research protocol capture'
    try:
        detector = Detector()
        camera = _camera(cv2, protocol['camera'], title)
        while True:
            frame, now = _frame(cv2, camera), time.monotonic()
            detection = detector.analyze(frame)
            if schedule.due(now):
                sample = save_attempt(project, frame, detection, protocol, schedule.current, digest,
                                      detector_configuration(detector), schedule.retake_of)
                schedule.saved(sample['id'], now)
            canvas = _capture_view(frame, detection, protocol, schedule, now, show_landmarks)
            cv2.imshow(title, cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord(' '): schedule.toggle(now)
            if key == ord('r') and schedule.last_id: schedule.retake(now)
            if key == ord('l'): show_landmarks = not show_landmarks
    finally:
        if camera is not None: camera.release()
        if detector is not None: detector.close()
        cv2.destroyAllWindows()


def _model_panel(draw, pair, index, model, result):
    x = 25 + index * 590
    fit_text(draw, f"{chr(65 + index)}: {model.metadata['name']}", (x, 580), 560, 24)
    text = result['prediction'] if result['status'] == 'ok' else ('No usable hand detection' if result['status'] == 'no_usable_detection' else 'Model-specific error')
    fit_text(draw, str(text), (x, 620), 560, 32)
    fit_text(draw, f"{model.metadata['classifier']} | {model.metadata['variant']}", (x, 668), 560, 16)
    fit_text(draw, 'Experiment: ' + str(model.metadata.get('experiment', 'unknown')).split('/')[-1], (x, 692), 560, 14)
    if pair.observation['smoothed_predictions'] is not None:
        fit_text(draw, f"Separate smoothed ({pair.smoothing}): {pair.observation['smoothed_predictions'][index]}", (x, 712), 560, 15)


def _live_view(pair, show_landmarks, mirror, message):
    observation = pair.observation
    image = overlay(pair.frame, observation['detection']) if show_landmarks else pair.frame.copy()
    if mirror: image = ImageOps.mirror(image)
    canvas = Image.new('RGB', (1200, 850), '#101e32')
    preview = ImageOps.contain(image, (1160, 500))
    canvas.paste(preview, ((1200 - preview.width) // 2, 60))
    draw = ImageDraw.Draw(canvas)
    fit_text(draw, 'A/B development view — same frame ' + observation['frame_id'][:10], (25, 18), 1150, 24)
    for index, (model, result) in enumerate(zip(pair.models, observation['results'])):
        _model_panel(draw, pair, index, model, result)
    fit_text(draw, observation['category'] + ' | intended: ' + str(observation['intended_label']), (25, 735), 1150, 22)
    fit_text(draw, message, (25, 772), 900, 17)
    draw.rounded_rectangle((955, 765, 1175, 800), 8, fill='#235c70')
    fit_text(draw, 'Save frame (S)', (975, 773), 190, 20)
    fit_text(draw, 'F freeze/resume | S select/save frame | L landmarks | Q stop | raw predictions by default', (25, 810), 1150, 17)
    return canvas


def _save_live(pair, out):
    try:
        return 'Saved development frame: ' + str(pair.save(out))
    except FileExistsError:
        return 'This exact frame was already saved; resume to choose a new one.'


def _save_mouse(cv2, title, queue):
    def click(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and 955 <= x <= 1175 and 765 <= y <= 800:
            queue.append(ord('s'))
    cv2.setMouseCallback(title, click)


def _live_loop(cv2, camera, detector, pair, title, out, countdown, mirror):
    show_landmarks, deadline, message = True, None, 'Agreement does not establish correctness without an intended label.'
    clicks = []
    _save_mouse(cv2, title, clicks)
    while True:
        if not pair.frozen or pair.observation is None:
            pair.process(_frame(cv2, camera), detector)
        if deadline is not None and time.monotonic() >= deadline:
            message = _save_live(pair, out)
            deadline = None
        canvas = _live_view(pair, show_landmarks, mirror, message)
        cv2.imshow(title, cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if clicks: key = clicks.pop(0)
        if key in (27, ord('q')): break
        if key == ord('f'): pair.frozen = not pair.frozen
        if key == ord('l'): show_landmarks = not show_landmarks
        if key == ord('s'):
            if countdown == 0:
                message = _save_live(pair, out)
            else:
                deadline = time.monotonic() + countdown
                message = f'Selected capture in {countdown}s. Keep the previously intended gesture stable.'


def run_live_ab(model_a, model_b, out, camera=0, intended_label=None, countdown=0, smoothing=1, mirror=True):
    import cv2
    if countdown < 0 or smoothing < 1:
        raise ValueError('Countdown must be nonnegative and smoothing at least one.')
    pair = LivePair(ModelHandle.load(model_a), ModelHandle.load(model_b), smoothing)
    print(pair.compatibility['limitation'])
    pair.set_intended(intended_label)
    detector, source, title = None, None, 'Research live A/B — development only'
    try:
        detector = Detector()
        source = _camera(cv2, camera, title)
        _live_loop(cv2, source, detector, pair, title, out, countdown, mirror)
    finally:
        if source is not None: source.release()
        if detector is not None: detector.close()
        cv2.destroyAllWindows()
