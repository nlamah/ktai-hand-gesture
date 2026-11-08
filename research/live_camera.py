"""Minimal native webcam preview: camera image and the model's current letter."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import time
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from research.capture import DEFAULT_CAPTURE_DIR, save_capture
from research.data import Detector
from research.protocol import ProtocolSession
from research.live_model import DEFAULT_MODEL, load_bundle, predict_letter
from research.model_choices import (MODEL_CHOICES, MODEL_BUTTONS, PRIMARY_KEYS,
                                    describe_model,model_shortcut,visible_model_buttons)

WINDOW = 'ASL letter - Space: capture | Q / Esc: close'
BUTTON = (12, 488, 628, 532)
PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 480
MODEL_ROW_COUNT = (len(PRIMARY_KEYS) + 2) // 3
ROUND_BUTTON = (12, 576 + MODEL_ROW_COUNT * 48, 208, 620 + MODEL_ROW_COUNT * 48)
HAND_CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                    (15,16),(13,17),(0,17),(17,18),(18,19),(19,20))


def new_round(name=None, number=1):
    now=datetime.now(timezone.utc)
    return {'id':now.strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8],
            'name':name or now.astimezone().strftime('%d-%m %H:%M'),
            'number':number,'started_at_utc':now.isoformat()}



def render_preview(frame, letter, detection=None, mirror=True):
    """Fixed-size preview; mirroring affects display only, never model inputs."""
    height, width = frame.shape[:2]
    scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    if mirror:resized = cv2.flip(resized, 1)
    canvas = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
    y = (PREVIEW_HEIGHT - resized.shape[0]) // 2
    x = (PREVIEW_WIDTH - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    if detection and detection.get('status')=='ok':
        points=np.asarray(detection.get('image_landmarks',[]),dtype=float)
        if points.shape==(21,3) and np.isfinite(points).all():
            # Normalized detector points, projected into the letterboxed mirrored preview.
            coords=np.rint(np.c_[x+((1-points[:,0]) if mirror else points[:,0])*(resized.shape[1]-1),
                                  y+points[:,1]*(resized.shape[0]-1)]).astype(int)
            for a,b in HAND_CONNECTIONS:
                cv2.line(canvas,tuple(coords[a]),tuple(coords[b]),(70,220,70),2,cv2.LINE_AA)
            for point in coords:
                cv2.circle(canvas,tuple(point),4,(0,200,255),-1,cv2.LINE_AA)
    cv2.rectangle(canvas, (12, 12), (120, 114), (255, 255, 255), -1)
    cv2.putText(canvas, letter or '-', (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                2.7, (0, 0, 0), 5, cv2.LINE_AA)
    return canvas


def camera_panel(preview, prompt=None, notice=None, model_key=None, locked=False, round_info=None, archive=False):
    """Fixed-size footer with a clickable snapshot button, also in freeze mode."""
    buttons=visible_model_buttons(archive)
    rows=(len(buttons)+2)//3
    round_button=(12,576+rows*48,208,620+rows*48)
    panel = np.zeros((round_button[3] + 36, PREVIEW_WIDTH, 3), dtype=np.uint8)
    panel[:PREVIEW_HEIGHT] = preview
    cv2.rectangle(panel, BUTTON[:2], BUTTON[2:], (235, 235, 235), -1)
    text = prompt or 'Save example [Space]'
    cv2.putText(panel, text, (24, 516), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, notice or 'Click the button or press Space. Q/Esc: close.',
                (12, 559), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    for key,label,path,rect in buttons:
        color=(120,120,120) if locked else (170,225,170) if key==model_key else (220,220,220)
        cv2.rectangle(panel,rect[:2],rect[2:],color,-1)
        shortcut = model_shortcut(key)
        cv2.putText(panel,f'{label} [{shortcut}]',(rect[0]+8,rect[1]+28),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,0,0),1,cv2.LINE_AA)
    cv2.rectangle(panel,round_button[:2],round_button[2:],(120,120,120) if locked else (220,220,220),-1)
    cv2.putText(panel,'New round [N]',(24,round_button[1]+28),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
    cv2.rectangle(panel,(222,round_button[1]),(418,round_button[3]),(120,120,120) if locked else (220,220,220),-1)
    cv2.putText(panel,'Hide archive [A]' if archive else 'More models [A]',(230,round_button[1]+28),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
    label=f"Round: {round_info['name'][:36]} / {round_info['number']}" if round_info else 'L: toggle landmarks'
    cv2.putText(panel,label,(12,round_button[3]+22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    if round_info:
        cv2.putText(panel,'L: landmarks',(442,round_button[1]+28),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    return panel


def confirm_expected(preview, initial=None, fixed=False, panel_renderer=None):
    """Freeze the chosen frame while the user labels it; Esc discards it."""
    expected = initial
    while True:
        prompt = f'Intended letter: {expected or "?"}  | Enter: save'
        note='Enter: confirm planned capture | Esc: retry' if fixed else 'A-Z: choose letter | Backspace: unknown | Esc: cancel'
        view=panel_renderer(preview,prompt,note) if panel_renderer else camera_panel(preview,prompt,note,locked=True)
        cv2.imshow(WINDOW,view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            return False, None
        if key in (10, 13):
            return True, expected
        if not fixed and key in (8, 127):
            expected = None
        elif not fixed and (ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z')):
            expected = chr(key).upper()
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False, None
        except cv2.error:
            return False, None


def snapshot_mouse(event, x, y, flags, state):
    if state.get('annotating') or event != cv2.EVENT_LBUTTONDOWN:
        return
    if not state.get('model_locked'):
        for key,label,path,rect in visible_model_buttons(state.get('archive',False)):
            if rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]:
                state['model_request']=key
                return
    rows=(len(visible_model_buttons(state.get('archive',False)))+2)//3
    round_y=576+rows*48
    if not state.get('model_locked') and round_y<=y<=round_y+44:
        if 12<=x<=208:state['round_request']=True;return
        if 222<=x<=418:state['archive']=not state.get('archive',False);return
    if BUTTON[0] <= x <= BUTTON[2] and BUTTON[1] <= y <= BUTTON[3]:
        state['requested'] = state.get('shown')


def run_camera(model_path=DEFAULT_MODEL, camera=0, output=DEFAULT_CAPTURE_DIR, guided=False, role="development", resume=None, round_name=None, protocol_session=None):
    bundle = load_bundle(model_path)
    model_sha = sha256(Path(model_path).read_bytes()).hexdigest() if Path(model_path).is_file() else None
    model_key,model_label=describe_model(model_path,bundle)
    model_cache={str(Path(model_path).resolve()):(bundle,model_sha)}
    protocol = protocol_session
    if guided or protocol_session:
        protocol = protocol_session or (ProtocolSession(resume, model_sha) if resume else ProtocolSession.create(output, role, model_sha))
        output = protocol.directory
        print(f'Capture session: {protocol.directory}', flush=True)
        print('Save all planned attempts, including errors. Predictions are hidden to limit selection.', flush=True)
        if protocol.current is None:
            print('Capture session is already complete.', flush=True)
            return
    capture = cv2.VideoCapture(camera)
    detector = None
    window_open = False
    state = {"shown": None, "requested": None, "model_request":None, "model_locked":bool(protocol),
             "archive":bool(model_key and model_key not in PRIMARY_KEYS)}
    round_info = new_round(round_name)
    show_landmarks = not getattr(protocol,'hide_landmarks',False)
    capture_ready_at = 0.
    print(f"Capture round: {round_info['name']} / {round_info['number']} ({round_info['id']})",flush=True)
    expected = None
    notice = None
    notice_until = 0.
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps and check camera permission. Try --camera 1 if needed.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        detector = Detector()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        window_open = True
        cv2.setMouseCallback(WINDOW, snapshot_mouse, state)
        print('Camera started. [1] Reference, [2] World joints, [9] G/H y180; [7/8/0] left/right controls. A: model archive. N: new round. L: landmarks. Space: save example (correct or incorrect). Q/Esc: close.', flush=True)
        while True:
            if protocol and (protocol.directory/'withdrawn.json').exists():
                raise ValueError('This capture round was withdrawn. Close the window and start a new session.')
            if state.pop('round_request',False) and not protocol:
                round_info=new_round(round_name,round_info['number']+1)
                state['shown']=None;state['requested']=None
                print(f"Capture round: {round_info['name']} / {round_info['number']} ({round_info['id']})",flush=True)
            requested_model=state.pop('model_request',None)
            if requested_model and not protocol:
                key,label,path=next(c for c in MODEL_CHOICES if c[0]==requested_model)
                try:
                    cache_key=str(path.resolve())
                    if cache_key not in model_cache:
                        model_cache[cache_key]=(load_bundle(path),sha256(path.read_bytes()).hexdigest())
                    bundle,model_sha=model_cache[cache_key]
                except (OSError,ValueError,ImportError) as error:
                    notice='Model loading failed; see terminal.'
                    notice_until=time.monotonic()+4
                    print(f'{label} could not be loaded: {error}',flush=True)
                else:
                    model_path=path;model_key=key;model_label=label
                    if key not in PRIMARY_KEYS:state['archive']=True
                    state['shown']=None;state['requested']=None
                    print(f'Selected model: {label}',flush=True)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('No camera frame received. Check the camera and camera permission.')
            captured_at = datetime.now(timezone.utc).isoformat()
            # Keep capture and inference unmirrored and preserve their aspect ratio.
            scale = min(1., PREVIEW_WIDTH / max(frame.shape[:2]))
            model_frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale))) if scale < 1 else frame
            image = Image.fromarray(cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB))
            detection = {**detector.analyze(image), 'image_size':[image.width,image.height]}
            prediction = predict_letter(bundle, detection)
            letter = prediction['letter'] if prediction else None
            packet = {'frame': frame.copy(), 'model_frame': model_frame.copy(), 'detection': detection,
                      'prediction': prediction, 'captured_at_utc': captured_at, 'camera': camera, 'session_id': round_info['id'], 'capture_round':dict(round_info), 'landmarks_visible':show_landmarks}
            if protocol:
                packet['protocol'] = protocol.metadata()
                packet['session_id'] = protocol.plan['id']
                packet['capture_round'] = {'id':protocol.plan['id'],'name':protocol.plan['id'],'number':1,'kind':'guided'}
            preview_mirrored=getattr(protocol,'mirror_preview',True)
            packet['preview_mirrored']=preview_mirrored
            preview = render_preview(frame, '?' if protocol else letter, detection if show_landmarks else None,
                **({'mirror':False} if not preview_mirrored else {}))
            state['shown'] = (packet, preview)
            participant_pending=bool(protocol and hasattr(protocol,'participant_confirmed') and protocol.current['participant']!=protocol.participant_confirmed)
            if hasattr(protocol,'render_panel'):
                cv2.imshow(WINDOW,protocol.render_panel(preview,
                    prompt=f"{getattr(protocol,'participant_label',protocol.current['participant'])}: press Enter to begin" if participant_pending else None,
                    notice='Check participant and hand. Enter starts the round.' if participant_pending else notice if time.monotonic()<notice_until else None))
            else:
                cv2.imshow(WINDOW, camera_panel(preview, prompt='Take planned photo [Space]' if protocol else None, notice=protocol.label() if protocol else notice if time.monotonic() < notice_until else f'Model: {model_label} | Space: capture | Q/Esc: close', model_key=model_key, locked=bool(protocol), round_info=packet['capture_round'],archive=state['archive']))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if participant_pending:
                state['requested']=None
                if key in (10,13):protocol.participant_confirmed=protocol.current['participant']
                continue
            if ord('1')<=key<=ord('9') and not protocol:
                state['model_request']=MODEL_CHOICES[key-ord('1')][0]
            if key==ord('0') and len(MODEL_CHOICES)>=10 and not protocol:
                state['model_request']=MODEL_CHOICES[9][0]
            if key in (ord('n'),ord('N')) and not protocol:
                state['round_request']=True
            if key in (ord('a'),ord('A')) and not protocol:
                state['archive']=not state['archive']
            if key in (ord('h'),ord('H')) and hasattr(protocol,'toggle_guidance'):
                protocol.toggle_guidance()
            if key in (ord('l'),ord('L')):
                show_landmarks=not show_landmarks
            if key == ord(' '):
                state['requested'] = state['shown']
            if state['requested'] is not None and time.monotonic()<capture_ready_at:
                state['requested']=None
            if state['requested'] is not None:
                chosen, frozen_preview = state['requested']
                state['requested'] = None
                state['annotating'] = True
                try:
                    if protocol and getattr(protocol,'capture_immediately',False):
                        confirmed, annotation=True,chosen['protocol']['task']['letter']
                    elif protocol:
                        confirmed, annotation = confirm_expected(frozen_preview, chosen['protocol']['task']['letter'], fixed=True,
                            **({'panel_renderer':protocol.render_panel} if hasattr(protocol,'render_panel') else {}))
                    else:
                        confirmed, annotation = confirm_expected(frozen_preview, expected,
                            panel_renderer=lambda p,prompt,note:camera_panel(p,prompt,note,
                                model_key=model_key,locked=True,round_info=chosen['capture_round'],archive=state['archive']))
                finally:
                    state['annotating'] = False
                    state['requested'] = None
                if confirmed:
                    try:
                        path = save_capture(chosen, annotation, bundle, model_path, model_sha, frozen_preview, output)
                        expected = annotation
                        notice = 'Photo saved. Release the space bar.' if getattr(protocol,'capture_immediately',False) else f'Saved: {path.name}'
                        capture_ready_at=time.monotonic()+getattr(protocol,'capture_cooldown',0.)
                        print(f'Capture saved: {path}', flush=True)
                        if protocol:
                            protocol.refresh()
                    except (OSError, ValueError) as error:
                        notice = 'Saving failed; see terminal.'
                        print(f'Saving failed: {error}', flush=True)
                else:
                    notice = 'Capture cancelled.'
                notice_until = time.monotonic() + 3.
                if protocol and protocol.current is None:
                    print(f'Capture session complete: {len(protocol.completed)}/{len(protocol.plan["tasks"])} captures saved.', flush=True)
                    break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break  # Window was closed between rendering and the visibility check.
    finally:
        capture.release()
        if detector is not None:
            detector.close()
        if window_open:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0, help='Camera index; default 0.')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=None, help='Directory for manually saved confusions.')
    parser.add_argument('--round-name',help='Optional label stored with each capture, e.g. GH-back-of-hand.')
    parser.add_argument('--guided', action='store_true', help='Guided C/D/O tilt protocol, 27 planned captures.')
    parser.add_argument('--role', choices=('development','evaluation'), default=None)
    parser.add_argument('--resume', type=Path, help='Resume an existing guided session directory.')
    args = parser.parse_args()
    if (args.resume or args.role) and not args.guided:
        parser.error('--resume and --role require --guided')
    output = args.output or (DEFAULT_CAPTURE_DIR.parent / 'guided' if args.guided else DEFAULT_CAPTURE_DIR)
    if not args.model.is_file():
        parser.error('Live model missing. Train it first with: python -m research.live_model')
    try:
        run_camera(args.model, args.camera, output, args.guided, args.role or "development", args.resume, args.round_name)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, ImportError, cv2.error) as error:
        parser.exit(1, f'Live recognition stopped: {error}\n')


if __name__ == '__main__':
    main()
