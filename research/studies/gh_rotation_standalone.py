"""Isolate G/H camera-y180 training augmentation from mirroring and calibration."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

from research.data import atomic_json
from research.gh_vertical import rotate_vertical
from research.inputs import ANGLE_COLUMNS, LABELS, write_csv
from research.live_model import BUNDLE_VERSION, SHIN_VERSION, DEFAULT_MODEL, load_bundle, predict_letter
from research.studies.compare_angles import classifier, variant_inputs
from research.studies.compare_models import load_inputs
from research.studies.mirror_augmentation import metrics

OUT=Path('results/rotation_GH_y180_01')
WORK=Path('runs/rotation_GH_y180_01')
CACHE=Path('data/asl_kaggle/research')
MODEL=Path('models/asl_shin_angles_GH_y180.joblib')
ARMS=('original','gh_y180')


def augmented_rows(X,y):
    """One original per source; an additional proper y180 turn for G/H only."""
    y=np.asarray(y);gh=np.isin(y,['G','H'])
    return (np.concatenate((X,rotate_vertical(X[gh]))),np.r_[y,y[gh]],
            np.r_[np.where(gh,.5,1.),np.full(gh.sum(),.5)])


def fit_rotated(baseline,X,y):
    rows,targets,weights=augmented_rows(X,y)
    preprocessing=deepcopy(baseline[0]);svm=clone(baseline[-1])
    classes=np.unique(y)
    svm.set_params(gamma=baseline[-1]._gamma,
                   class_weight=dict(zip(classes,compute_class_weight('balanced',classes=classes,y=y))))
    svm.fit(preprocessing.transform(rows),targets,sample_weight=weights)
    return make_pipeline(preprocessing,svm)


def frozen_protocol():
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    return {'version':'gh-y180-standalone-v1','features':835,'learner':'26-class RBF SVM, C=10',
            'rotation_axis':'camera y (vertical)','rotation_degrees':180,'matrix_diagonal':[-1,1,-1],
            'augmented_letters':['G','H'],'source_weight':'G/H original and turn: 0.5 each; other sources: 1',
            'mirror_augmentation':False,'camera_calibration_sources':0,'routing_gate':False,
            'preprocessing':'Fit only on original fitting rows; share scaling, resolved gamma and balanced class weights',
            'evaluation':'Original accepted images only; same historical five filename-block folds; no test augmentation',
            'selection':'User-requested isolated y180 recipe after earlier exploratory rotation studies; no parameter search in this run',
            'limitation':'Normalized landmark-feature rotation does not render new RGB views, occlusions or detector errors',
            'historical_model':str(DEFAULT_MODEL.relative_to(Path.cwd())),
            'historical_model_sha256':sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),
            'input_hashes':meta['hashes'],
            'frozen_inputs_sha256':{str(CACHE/name):sha256((CACHE/name).read_bytes()).hexdigest()
                                    for name in ('inputs.npz','inputs_metadata.json','summary.json')},
            'source_sha256':{p:sha256(Path(p).read_bytes()).hexdigest() for p in
                            ('research/gh_vertical.py','research/studies/gh_rotation_standalone.py',
                             'research/studies/compare_angles.py','research/live_model.py')}}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    protocol=frozen_protocol();path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('Frozen study changed; use a new directory.')
    else:atomic_json(path,protocol)
    return protocol


def inputs():
    raw,y,ids,splits=load_inputs(CACHE)
    X=variant_inputs(raw,'shin_angles')
    assert X.shape==(60251,835) and len(set(ids))==len(y) and np.isfinite(X).all()
    return X,y,np.asarray(ids),splits


def cv(protocol):
    import pandas as pd
    X,y,ids,splits=inputs()
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    groups=np.array([r['group'] for r in meta['samples']])
    old=pd.read_csv('results/angles/oof_predictions_svm.csv',keep_default_na=False)
    old=old[old.variant=='shin_angles'].set_index('id').loc[ids]
    np.testing.assert_array_equal(old.label,y)
    predictions={arm:np.full(len(y),'',dtype='<U1') for arm in ARMS}
    assignment=np.zeros(len(y),int);fitting=[]
    for fold,train,test in splits:
        assert not set(groups[train])&set(groups[test])
        np.testing.assert_array_equal(old.fold.to_numpy()[test],np.full(len(test),fold))
        assignment[test]=fold;started=time.monotonic()
        paths={arm:WORK/f'{arm}_fold{fold}.joblib' for arm in ARMS}
        baseline=joblib.load(paths['original']) if paths['original'].exists() else classifier('svm','shin_angles',len(train)).fit(X[train],y[train])
        if not paths['original'].exists():joblib.dump(baseline,paths['original'],compress=3)
        candidate=joblib.load(paths['gh_y180']) if paths['gh_y180'].exists() else fit_rotated(baseline,X[train],y[train])
        if not paths['gh_y180'].exists():joblib.dump(candidate,paths['gh_y180'],compress=3)
        for arm,model in zip(ARMS,(baseline,candidate)):
            checkpoint=WORK/f'{arm}_fold{fold}.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as saved:
                    np.testing.assert_array_equal(saved['ids'],ids[test]);pred=saved['prediction']
            else:
                pred=model.predict(X[test]);np.savez_compressed(checkpoint,ids=ids[test],prediction=pred)
            predictions[arm][test]=pred
        np.testing.assert_array_equal(predictions['original'][test],old.prediction.to_numpy()[test])
        n_gh=int(np.isin(y[train],['G','H']).sum())
        fitting.append({'fold':fold,'original_sources':len(train),'GH_sources':n_gh,
                        'rotated_fitting_rows':len(train)+n_gh,'held_out_originals':len(test),
                        'seconds':time.monotonic()-started})
        print(json.dumps(fitting[-1]),flush=True)
    assert (assignment>0).all() and all((p!='').all() for p in predictions.values())
    rows=[];letters=[];paired=[]
    for fold in ('pooled',*sorted(set(assignment))):
        keep=np.ones(len(y),bool) if fold=='pooled' else assignment==fold
        for arm in ARMS:
            score,detail=metrics(y[keep],predictions[arm][keep])
            rows.append({'model':arm,'fold':fold,**score})
            for j,letter in enumerate(LABELS):
                letters.append({'model':arm,'fold':fold,'letter':letter,'images':int(detail[3][j]),
                                'precision':float(detail[0][j]),'recall':float(detail[1][j]),'f1':float(detail[2][j])})
        a,b=predictions['original'][keep],predictions['gh_y180'][keep];truth=y[keep]
        fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
        paired.append({'fold':fold,'fixed':fixed,'lost':lost,'net':fixed-lost,
                       'wrong_to_wrong':int(((a!=truth)&(b!=truth)&(a!=b)).sum())})
    write_csv(OUT/'metrics.csv',rows);write_csv(OUT/'per_letter.csv',letters)
    write_csv(OUT/'paired_changes.csv',paired);write_csv(OUT/'fitting.csv',fitting)
    write_csv(OUT/'predictions.csv',[{'id':ids[i],'label':y[i],'group':groups[i],'fold':int(assignment[i]),
              **{arm:predictions[arm][i] for arm in ARMS}} for i in range(len(y))])
    print('Standalone G/H CV complete; no mirrors, captures or routing rules used.',flush=True)


def train(protocol):
    X,y,ids,splits=inputs();path=WORK/'original_full.joblib'
    baseline=joblib.load(path) if path.exists() else classifier('svm','shin_angles',len(y)).fit(X,y)
    if not path.exists():joblib.dump(baseline,path,compress=3)
    historical=load_bundle(DEFAULT_MODEL)
    np.testing.assert_equal(baseline[-1]._gamma,historical['estimator'][-1]._gamma)
    sample=np.linspace(0,len(y)-1,260,dtype=int)
    np.testing.assert_array_equal(baseline.predict(X[sample]),historical['estimator'].predict(X[sample]))
    if MODEL.exists():
        bundle=load_bundle(MODEL)
        if bundle.get('augmentation')!=protocol:raise ValueError('Existing model uses a different protocol.')
    else:
        started=time.monotonic();model=fit_rotated(baseline,X,y)
        bundle={'version':BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':list(ANGLE_COLUMNS),
                'sklearn_version':sklearn.__version__,'estimator':model,'training_images':len(y),
                'training_rows':len(y)+int(np.isin(y,['G','H']).sum()),'augmentation':protocol,
                'source_hashes':protocol['input_hashes'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'fit_seconds':time.monotonic()-started,
                'scope':'Original-only source fit plus G/H y180 copies; no RGB mirrors, webcam calibration or routing gate'}
        temporary=MODEL.with_suffix('.partial.joblib');joblib.dump(bundle,temporary,compress=3)
        load_bundle(temporary);temporary.replace(MODEL)
        atomic_json(MODEL.with_suffix('.json'),{k:v for k,v in bundle.items() if k!='estimator'})
    capture_rows=[]
    for path in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(path.read_text())
        if r['detection']['status']!='ok':continue
        for name,info in r['files'].items():assert sha256((path.parent/name).read_bytes()).hexdigest()==info['sha256']
        a=predict_letter(historical,r['detection'])['letter'];b=predict_letter(bundle,r['detection'])['letter']
        capture_rows.append({'id':r['id'],'intended':r['expected_letter'],'original':a,'gh_y180':b,
                             'role':'previously_inspected_development_control_not_used_for_fitting'})
    write_csv(OUT/'capture_replay.csv',capture_rows)
    atomic_json(OUT/'model.json',{'candidate_sha256':sha256(MODEL.read_bytes()).hexdigest(),
               'fresh_original_sha256':sha256((WORK/'original_full.joblib').read_bytes()).hexdigest(),
               'training_sources':len(y),'GH_sources':int(np.isin(y,['G','H']).sum()),
               'training_rows':bundle['training_rows'],'calibration_sources':0})
    print(json.dumps(capture_rows,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','cv','train'),required=True)
    args=parser.parse_args();protocol=freeze()
    if args.stage!='freeze':{'cv':cv,'train':train}[args.stage](protocol)


if __name__=='__main__':main()
