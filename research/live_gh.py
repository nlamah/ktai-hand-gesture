"""Native live diagnostic restricted to G versus H (not all 26 letters)."""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
import sklearn

from research.data import Detector
from research.geometry import measure_hand_features
from research.image_angles import projected_angles
from research.inputs import encode_angles
from research.live_camera import render_preview,PREVIEW_WIDTH,PREVIEW_HEIGHT
from research.studies.compare_gh_classifier import select_inputs

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('3d','2d','both','middle')
NAMES={'3d':'Alle 3D-hoeken','2d':'Alle 2D-hoeken','both':'Alle 2D + 3D-hoeken','middle':'Alleen middelvingerbuiging'}
WINDOW='G/H-diagnose - uitsluitend G en H | Q / Esc: sluiten'
BUTTONS=tuple((v,(12+i*156,536,164+i*156,574)) for i,v in enumerate(VARIANTS))


def load_model(variant):
    if variant not in VARIANTS:raise ValueError('Kies 3d, 2d, both of middle.')
    bundle=joblib.load(ROOT/'models/gh_diagnostic'/f'gh_{variant}.joblib')
    expected='gh-middle-diagnostic-v1' if variant=='middle' else 'gh-angle-diagnostic-v1'
    width={'3d':15,'2d':14,'both':29,'middle':2}[variant]
    model=bundle.get('estimator')
    if bundle.get('version')!=expected or bundle.get('sklearn_version')!=sklearn.__version__:
        raise ValueError('Onjuiste G/H-modelversie of scikit-learn-versie.')
    if getattr(model,'n_features_in_',None)!=width or list(getattr(model,'classes_',()))!=['G','H']:
        raise ValueError('Onjuist G/H-featureschema.')
    return bundle


def inputs(detection,variant):
    world=encode_angles(measure_hand_features(detection['world_landmarks'],detection.get('handedness'))).astype(np.float32)
    pixel=projected_angles(detection['image_landmarks'],detection['image_size'])
    row=np.array([world[6],pixel[6]],np.float32) if variant=='middle' else select_inputs(world,pixel,variant)
    return row.reshape(1,-1)


def predict(bundle,detection,variant):
    if detection.get('status')!='ok':return None
    return str(bundle['estimator'].predict(inputs(detection,variant))[0])


def panel(preview,variant):
    result=np.zeros((596,640,3),np.uint8);result[:480]=preview
    cv2.putText(result,'Alleen G/H-test: andere letters worden ook G of H.',(12,499),cv2.FONT_HERSHEY_SIMPLEX,.54,(0,220,255),1,cv2.LINE_AA)
    cv2.putText(result,NAMES[variant],(12,523),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    for i,(v,rect) in enumerate(BUTTONS,1):
        cv2.rectangle(result,rect[:2],rect[2:],(170,225,170) if v==variant else (220,220,220),-1)
        label={'3d':'3D','2d':'2D','both':'2D + 3D','middle':'Middel'}[v]
        cv2.putText(result,f'{label} [{i}]',(rect[0]+10,561),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(result,'L: landmarks aan/uit | Q/Esc: sluiten',(12,590),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    return result


def run_camera(variant='middle',camera=0):
    bundle=load_model(variant);cache={variant:bundle};state={};overlay=True
    capture=cv2.VideoCapture(camera);detector=None;opened=False
    def mouse(event,x,y,flags,unused):
        if event==cv2.EVENT_LBUTTONDOWN:
            for v,rect in BUTTONS:
                if rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]:state['variant']=v
    try:
        if not capture.isOpened():raise RuntimeError('Camera kan niet worden geopend. Sluit andere camera-apps.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,PREVIEW_WIDTH);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,PREVIEW_HEIGHT)
        detector=Detector();cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE);opened=True
        cv2.setMouseCallback(WINDOW,mouse)
        while True:
            requested=state.pop('variant',None)
            if requested:
                if requested not in cache:cache[requested]=load_model(requested)
                variant=requested;bundle=cache[requested]
            ok,frame=capture.read()
            if not ok or frame is None:raise RuntimeError('Geen camerabeeld ontvangen.')
            scale=min(1.,640/max(frame.shape[:2]))
            raw=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale))) if scale<1 else frame
            image=Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
            detection={**detector.analyze(image),'image_size':[image.width,image.height]}
            letter=predict(bundle,detection,variant)
            cv2.imshow(WINDOW,panel(render_preview(frame,letter,detection if overlay else None),variant))
            key=cv2.waitKey(1)&255
            if key in (27,ord('q'),ord('Q')):break
            if ord('1')<=key<=ord('4'):state['variant']=VARIANTS[key-ord('1')]
            if key in (ord('l'),ord('L')):overlay=not overlay
            try:
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
            except cv2.error:break
    finally:
        capture.release()
        if detector is not None:detector.close()
        if opened:cv2.destroyAllWindows()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default='middle')
    parser.add_argument('--camera',type=int,default=0)
    args=parser.parse_args()
    try:run_camera(args.variant,args.camera)
    except KeyboardInterrupt:pass
    except (OSError,ValueError,RuntimeError,ImportError,cv2.error) as error:parser.exit(1,f'G/H-test gestopt: {error}\n')


if __name__=='__main__':main()
