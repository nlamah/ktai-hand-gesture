"""Exploratory G/H volume rendering and occlusion-targeted landmark noise.

This is a robustness proxy, not a validated model of MediaPipe occlusion errors.
No withdrawn captures, test augmentation, or deployment model replacement.
"""
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

from research.data import atomic_json
from research.gh_vertical import rotate_vertical
from research.inputs import LABELS, write_csv
from research.live_model import detection_inputs
from research.studies.gh_rotation_standalone import inputs
from research.studies.mirror_augmentation import metrics
from research.synthetic_hand import PARAMETERS, VERSION, render_hand

OUT=Path('results/gh_occlusion_01')
WORK=Path('runs/gh_occlusion_01')
PREVIOUS=Path('runs/rotation_GH_y180_01')
LEVELS={'noise002':.02,'noise005':.05}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    protocol={
        'version':'gh-occlusion-proxy-v1','renderer':VERSION,'renderer_parameters':PARAMETERS,
        'render_screen':'64 fixed accepted G/H sources, 32 per class, lowest SHA256(id+screen-v1); original RGB, volume 0 and volume 180 degrees',
        'screen_detector':'MediaPipe 0.10.35 IMAGE; detection and presence thresholds 0.5; no threshold tuning',
        'proxy':'Rotate image/world landmarks around wrist by diag(-1,1,-1); add deterministic Gaussian xyz noise only to finger joints hidden behind another component in volume render',
        'hidden_candidates':[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20],
        'noise_sigma_palm_length':LEVELS,
        'noise_coupling':'Same standard-normal vector for image/world, each scaled by its own wrist-to-middle-MCP length; source-ID seed; same vector across both strengths',
        'visibility_assumptions':'Fixed approximate finger radii and palm thickness; source palm-facing assumption; no anatomical fit or measured detector error distribution',
        'training':'Only original fitting sources; original GH 0.5 plus one noisy rotated copy 0.5; other letters 1; original-only scaling, gamma and class weights',
        'evaluation':'Same five frozen filename-block folds; same 60251 accepted original images, 26 letters, pooled macro precision/F1; no test augmentation',
        'controls':['original','gh_y180'],
        'selection':'Two fixed exploratory strengths; no best-arm deployment or claim of confirmatory testing; this dataset has been inspected before',
        'withdrawn_capture_sources':0,'new_real_dorsal_test_sources':0,
        'claim_limit':'Original-view regression/robustness only; no evidence of improved real dorsal-camera recognition. Synthetic render detector failures are reported, never replaced with supplied landmarks.',
        'inputs':{p:digest(p) for p in ['data/asl_kaggle/manifest.json','data/asl_kaggle/research/inputs.npz','data/asl_kaggle/research/inputs_metadata.json','data/asl_kaggle/research/summary.json']},
        'source_sha256':{p:digest(p) for p in ['research/synthetic_hand.py',__file__,'research/live_model.py','research/geometry.py']}}
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol changed: use a new run directory.')
    else:atomic_json(path,protocol)
    return protocol


def perturb(points,hidden,noise,sigma):
    """Return new coordinates; preserve wrist, MCPs and all visible candidates."""
    p=np.asarray(points,float);result=(p-p[0])*np.array([-1.,1.,-1.])+p[0]
    result[np.asarray(hidden,dtype=int)]+=noise[np.asarray(hidden,dtype=int)]*sigma*np.linalg.norm(p[9]-p[0])
    return result


def transformed(sample,hidden,sigma):
    seed=int.from_bytes(sha256((sample['id']+'occlusion-noise-v1').encode()).digest()[:8],'little')
    noise=np.random.default_rng(seed).normal(size=(21,3))
    return {'status':'ok','handedness':sample['handedness'],
            **{k:perturb(sample[k],hidden,noise,sigma) for k in ('image_landmarks','world_landmarks')}}


def sources():
    meta=json.loads(Path('data/asl_kaggle/research/inputs_metadata.json').read_text())
    wanted={s['id']:i for i,s in enumerate(meta['samples']) if s['label'] in ('G','H')}
    manifest=json.loads(Path('data/asl_kaggle/manifest.json').read_text())
    rows=sorted((s for s in manifest['samples'] if s['id'] in wanted),key=lambda s:wanted[s['id']])
    assert len(rows)==4801 and len({s['id'] for s in rows})==4801
    return rows,wanted


def prepare():
    path=WORK/'features.npz'
    if path.exists():return
    X,y,ids,_=inputs();rows,wanted=sources();outputs={a:[] for a in LEVELS};audit=[]
    for n,s in enumerate(rows):
        i=wanted[s['id']]
        np.testing.assert_allclose(detection_inputs(s)[0],X[i],rtol=0,atol=1e-5,equal_nan=True)
        _,info=render_hand(s['image_landmarks'],180,(s['width'],s['height']))
        hidden=info['hidden_finger_joints']
        zero=detection_inputs(transformed(s,hidden,0))[0]
        np.testing.assert_allclose(zero,rotate_vertical(X[i:i+1])[0],rtol=0,atol=1e-5,equal_nan=True)
        for arm,sigma in LEVELS.items():outputs[arm].append(detection_inputs(transformed(s,hidden,sigma))[0])
        audit.append({'id':s['id'],'label':s['label'],'row':i,'hidden_joints':hidden,
                      'component_occlusion':info['component_occlusion']})
        if n%500==0:print(f'prepared {n}/{len(rows)}',flush=True)
    arrays={a:np.asarray(v,np.float32) for a,v in outputs.items()}
    assert all(np.isfinite(v[:,:820]).all() and not np.isinf(v).any() for v in arrays.values())
    np.savez_compressed(path,rows=np.array([wanted[s['id']] for s in rows]),ids=np.array([s['id'] for s in rows]),**arrays)
    atomic_json(OUT/'geometry.json',{'sources':len(rows),'zero_noise_rotation_check':'all 4801, absolute tolerance 1e-5',
                'hidden_joint_counts':dict(sorted(Counter(len(s['hidden_joints']) for s in audit).items())),
                'samples':audit})
    print('feature preparation complete',flush=True)


def screen():
    from research.data import Detector
    rows,_=sources()
    selected=[]
    for label in ('G','H'):
        selected+=sorted((r for r in rows if r['label']==label),key=lambda s:sha256((s['id']+'screen-v1').encode()).hexdigest())[:32]
    path=OUT/'render_screen.json'
    if path.exists():return
    detector=Detector();results=[]
    try:
        for n,s in enumerate(selected):
            with Image.open(Path('data/asl_kaggle')/s['image_path']) as raw:images={'original':raw.convert('RGB')}
            for degrees in (0,180):images[f'volume{degrees}']=render_hand(s['image_landmarks'],degrees,(s['width'],s['height']))[0]
            row={'id':s['id'],'label':s['label'],'filename':s['filename'],'conditions':{}}
            for condition,image in images.items():
                d=detector.analyze(image)
                row['conditions'][condition]=d
            results.append(row)
            if n%16==0:print(f'screened {n}/{len(selected)}',flush=True)
    finally:detector.close()
    counts={c:dict(Counter(r['conditions'][c]['status'] for r in results)) for c in ('original','volume0','volume180')}
    atomic_json(path,{'sources':len(results),'counts':counts,'detector_thresholds':detector.thresholds,'samples':results})
    print(json.dumps(counts),flush=True)


def train(arm):
    X,y,ids,splits=inputs()
    with np.load(WORK/'features.npz') as data:ghrows=data['rows'];copies=data[arm]
    for fold,tr,te in splits:
        path=WORK/f'{arm}_fold{fold}.joblib';checkpoint=WORK/f'{arm}_fold{fold}.npz'
        if checkpoint.exists():continue
        started=time.monotonic();baseline=joblib.load(PREVIOUS/f'original_fold{fold}.joblib')
        if path.exists():model=joblib.load(path)
        else:
            keep=np.isin(ghrows,tr);assert not np.isin(ghrows[keep],te).any()
            rows=np.r_[X[tr],copies[keep]];targets=np.r_[y[tr],y[ghrows[keep]]]
            weights=np.r_[np.where(np.isin(y[tr],['G','H']),.5,1),np.full(keep.sum(),.5)]
            assert np.isclose(weights.sum(),len(tr))
            scaler=deepcopy(baseline[0]);svm=clone(baseline[-1]);classes=np.unique(y[tr])
            svm.set_params(gamma=baseline[-1]._gamma,class_weight=dict(zip(classes,compute_class_weight('balanced',classes=classes,y=y[tr]))))
            svm.fit(scaler.transform(rows),targets,sample_weight=weights)
            model=make_pipeline(scaler,svm);joblib.dump(model,path,compress=3)
        pred=model.predict(X[te]);np.savez_compressed(checkpoint,ids=ids[te],prediction=pred)
        print(json.dumps({'arm':arm,'fold':fold,'seconds':time.monotonic()-started}),flush=True)


def report():
    X,y,ids,splits=inputs();del X
    arms=('original','gh_y180',*LEVELS);pred={a:np.full(len(y),'',dtype='<U1') for a in arms};assignment=np.zeros(len(y),int)
    for fold,_,te in splits:
        assignment[te]=fold
        for arm in arms:
            path=(PREVIOUS if arm in arms[:2] else WORK)/f'{arm}_fold{fold}.npz'
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['ids'],ids[te]);pred[arm][te]=saved['prediction']
    rows=[];letters=[];paired=[]
    for fold in ('pooled',1,2,3,4,5):
        keep=np.ones(len(y),bool) if fold=='pooled' else assignment==fold
        for arm in arms:
            score,detail=metrics(y[keep],pred[arm][keep]);rows.append({'model':arm,'fold':fold,**score})
            for j,label in enumerate(LABELS):letters.append({'model':arm,'fold':fold,'letter':label,'n':int(detail[3][j]),'precision':float(detail[0][j]),'recall':float(detail[1][j]),'f1':float(detail[2][j])})
            for reference in ('original','gh_y180'):
                a,b=pred[reference][keep],pred[arm][keep];truth=y[keep]
                fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
                paired.append({'model':arm,'reference':reference,'fold':fold,'fixed':fixed,'lost':lost,'net':fixed-lost})
    write_csv(OUT/'metrics.csv',rows);write_csv(OUT/'per_letter.csv',letters);write_csv(OUT/'paired.csv',paired)
    write_csv(OUT/'predictions.csv',[{'id':ids[i],'label':y[i],'fold':int(assignment[i]),**{a:pred[a][i] for a in arms}} for i in range(len(y))])
    atomic_json(OUT/'artifacts.json',{'protocol_sha256':digest(OUT/'protocol.json'),
                'results_sha256':{p.name:digest(p) for p in OUT.glob('*.csv')},
                'new_checkpoints_sha256':{p.name:digest(p) for p in WORK.glob('*fold*')},
                'source_features_sha256':digest(WORK/'features.npz')})
    print(json.dumps([r for r in rows if r['fold']=='pooled'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','screen','train','report'])
    parser.add_argument('--arm',choices=list(LEVELS));args=parser.parse_args();freeze()
    if args.stage=='train':
        if not args.arm:parser.error('--arm required for train')
        train(args.arm)
    else:globals()[args.stage]()


if __name__=='__main__':main()
