"""Controlled camera-y180 alternative to the index-axis G/H adaptation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
import sklearn

from research.data import Detector, atomic_json
from research.gh_view_adapter import GHViewGuard
from research.gh_vertical import VERSION, adaptation_rows
from research.inputs import ANGLE_COLUMNS, LABELS, write_csv
from research.live_model import BUNDLE_VERSION, SHIN_VERSION, detection_inputs, load_bundle, predict_letter
from research.mirroring import mirror_image, pixel_hash
from research.studies.mirror_augmentation import load_extracted, assignments_for, balanced_views, metrics, VIEWS

OUT = Path('results/gh_vertical_01')
WORK = Path('runs/gh_vertical_01')
MODEL = Path('models/asl_gh_vertical_adapter.joblib')
# Visually inspected back-of-hand frames. The last, palm-facing H is a diagnostic control only.
BACK_CAPTURES = ('20261031T152108010733Z_9233886f', '20261031T152116143793Z_11756e66',
                 '20261031T181839806179Z_9ef071cc')


def prepare():
    WORK.mkdir(parents=True, exist_ok=True);OUT.mkdir(parents=True, exist_ok=True)
    cache = WORK/'calibration.npz'
    if cache.exists():
        print('Calibration cache exists; existing pixels/settings remain frozen.', flush=True)
        return
    X=np.full((len(BACK_CAPTURES),2,835),np.nan,np.float32)
    statuses=np.full((len(BACK_CAPTURES),2),'',dtype='<U24');labels=[];records=[]
    detector=Detector(.5,.5)
    try:
        for i,id_ in enumerate(BACK_CAPTURES):
            folder=Path('captures/confusions')/id_;record=json.loads((folder/'capture.json').read_text())
            assert record['expected_letter'] in ('G','H')
            photo=folder/'model_input.png'
            assert sha256(photo.read_bytes()).hexdigest()==record['files']['model_input.png']['sha256']
            with Image.open(photo) as im:original=im.convert('RGB')
            views=[]
            for view,image in enumerate((original,mirror_image(original))):
                d=detector.analyze(image);statuses[i,view]=d['status']
                if d['status']=='ok':X[i,view]=detection_inputs(d)[0]
                views.append({'pixel_sha256':pixel_hash(image),'detection':{k:v for k,v in d.items() if k!='features'}})
            if statuses[i,0]!='ok' or not np.isfinite(X[i,0]).all():
                raise ValueError('A selected calibration source lacks valid original geometry.')
            labels.append(record['expected_letter'])
            records.append({'id':id_,'label':labels[-1],'role':'calibration_training',
                'view':'back_of_hand_visually_reviewed','capture_json_sha256':sha256((folder/'capture.json').read_bytes()).hexdigest(),
                'image_sha256':sha256(photo.read_bytes()).hexdigest(),'views':views})
    finally:detector.close()
    np.savez_compressed(cache,features=X,status=statuses,labels=labels,ids=BACK_CAPTURES)
    atomic_json(WORK/'calibration_records.json',records)
    atomic_json(OUT/'calibration_sources.json',[{k:v for k,v in r.items() if k!='views'} for r in records])
    print('Prepared three real back-of-hand calibration sources; not validation data.',flush=True)


def protocol():
    records=json.loads((OUT/'calibration_sources.json').read_text())
    return {'version':VERSION,'base_model':'models/asl_shin_angles_mirrored.joblib',
        'base_sha256':sha256(Path('models/asl_shin_angles_mirrored.joblib').read_bytes()).hexdigest(),
        'mirror_protocol_sha256':sha256(Path('results/mirror_01/protocol.json').read_bytes()).hexdigest(),
        'roll':'180 degrees around the vertical camera y axis: (dx,dy,dz) -> (-dx,dy,-dz); G/H only',
        'roll_limit':'Synthetic landmark rotation; no new visibility, occlusion or detector-error simulation',
        'calibration_weight_per_source':20.,'source_weight_policy':'Kaggle originals/mirrors/rolls together retain total weight one per original source',
        'calibration_sources':records,'calibration_role':'All three are training; replay is not an independent test',
        'calibration_permission':'User explicitly requested reusing existing hand data for training; raw capture records are not changed',
        'gate':{'minimum_horizontal':.8,'minimum_projection':.15,'allowed_new_labels':['G','H']},
        'preprocessing':'Reuse original-fit angle scaling and SVM gamma/class weights from the matched mirror baseline',
        'evaluation':'Same frozen source folds and random mix as mirror study; calibration sources are extra fitting data, never held-out evidence',
        'selection':'Axis correction requested by user after the index-axis development recipe; not selected by source-test score',
        'source_sha256':{p:sha256(Path(p).read_bytes()).hexdigest() for p in ('research/gh_view_adapter.py','research/gh_vertical.py','research/studies/gh_vertical_adaptation.py')}}


def fit(baseline,X,y,status,calibration):
    rows,targets,weights=adaptation_rows(X[:,0],X[:,1],y,status[:,1]=='ok',calibration)
    preprocessing=deepcopy(baseline[0]);learner=clone(baseline[-1])
    learner.fit(preprocessing.transform(rows),targets,sample_weight=weights)
    return make_pipeline(preprocessing,GHViewGuard(baseline[-1],learner)),len(rows)


def train():
    p=protocol();path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=p:raise ValueError('Protocol changed; use a new study directory.')
    atomic_json(path,p)
    with np.load(WORK/'calibration.npz') as d:cal={k:d[k] for k in ('features','status','labels')}
    X,y,ids,groups,status,hands=load_extracted(Path('runs/mirror_01'),Path('results/mirror_01'))
    accepted=status[:,0]=='ok';base=load_bundle(Path(p['base_model']))
    if MODEL.exists():
        bundle=load_bundle(MODEL)
        if bundle.get('gh_view_adaptation')!=p:raise ValueError('Existing adaptation model has a different protocol.')
    else:
        started=time.monotonic();model,n_rows=fit(base['estimator'],X[accepted],y[accepted],status[accepted],cal)
        bundle={**{k:v for k,v in base.items() if k!='estimator'},'estimator':model,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'gh_view_adaptation':p,
            'training_rows':n_rows,'calibration_sources':len(cal['labels']),
            'scope':'Personal G/H view adaptation. Known back-of-hand examples are fitting data, not independent validation.',
            'fit_seconds':time.monotonic()-started}
        temporary=MODEL.with_suffix('.partial.joblib');joblib.dump(bundle,temporary,compress=3)
        load_bundle(temporary);temporary.replace(MODEL)
        atomic_json(MODEL.with_suffix('.json'),{k:v for k,v in bundle.items() if k!='estimator'})
    captures=[]
    for path in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(path.read_text())
        if r['detection']['status']!='ok':continue
        d=r['detection'];a=predict_letter(base,d);b=predict_letter(bundle,d)
        captures.append({'id':r['id'],'intended':r['expected_letter'],'baseline':a['letter'],'adapted':b['letter'],
            'role':'calibration_training' if r['id'] in BACK_CAPTURES else 'previously_inspected_development_control'})
    write_csv(OUT/'capture_replay.csv',captures)
    atomic_json(OUT/'model.json',{'model_sha256':sha256(MODEL.read_bytes()).hexdigest(),
        'fit_seconds':bundle.get('fit_seconds'),'training_rows':bundle['training_rows'],'calibration_sources':3})
    print(json.dumps(captures,indent=2),flush=True)


def cv():
    import pandas as pd
    p=protocol();assert json.loads((OUT/'protocol.json').read_text())==p
    with np.load(WORK/'calibration.npz') as d:cal={k:d[k] for k in ('features','status','labels')}
    X,y,ids,groups,status,hands=load_extracted(Path('runs/mirror_01'),Path('results/mirror_01'))
    folds=json.loads(Path('data/asl_kaggle/research/summary.json').read_text())['folds']
    assignment=assignments_for(groups,folds);selected=balanced_views(y,assignment)
    reference=pd.read_csv('results/mirror_01/predictions.csv',keep_default_na=False)
    baseline=np.full(status.shape,'?',dtype='<U1');predicted=baseline.copy()
    for view,name in enumerate(VIEWS):
        rows=reference[(reference.model=='svm')&(reference.view==name)].set_index('id').loc[ids]
        assert rows.label.tolist()==y.tolist() and rows.fold.tolist()==assignment.tolist()
        baseline[:,view]=rows.mirror_training.to_numpy()
    fitting=[]
    for fold in sorted(set(assignment)):
        train=(assignment!=fold)&(status[:,0]=='ok');test=assignment==fold
        assert not set(groups[train])&set(groups[test])
        base=joblib.load(Path('runs/mirror_01')/f'svm_mirrored_fold{fold}.joblib')
        path=WORK/f'fold{fold}.joblib';started=time.monotonic()
        if path.exists():model=joblib.load(path);n_rows=None
        else:
            model,n_rows=fit(base,X[train],y[train],status[train],cal);joblib.dump(model,path)
        for view in range(2):
            ix=np.flatnonzero(test&(status[:,view]=='ok'))
            transformed=model[0].transform(X[ix,view])
            predicted[ix,view]=model[-1].adapt_predictions(transformed,baseline[ix,view])
            sample=ix[np.linspace(0,len(ix)-1,26,dtype=int)]
            np.testing.assert_array_equal(model.predict(X[sample,view]),predicted[sample,view])
        fitting.append({'fold':int(fold),'training_sources':int(train.sum()),'rows':n_rows,'seconds':time.monotonic()-started})
        print(json.dumps(fitting[-1]),flush=True)
    out_rows=[];letters=[];paired=[]
    for presentation in (*VIEWS,'balanced_mix'):
        view=np.full(len(y),VIEWS.index(presentation)) if presentation in VIEWS else selected
        index=np.arange(len(y));a=baseline[index,view];b=predicted[index,view];accepted=status[index,view]=='ok'
        for scope,mask in [('all_attempts',np.ones(len(y),bool)),('accepted_only',accepted)]:
            for fold in ('pooled',*sorted(set(assignment))):
                ix=mask if fold=='pooled' else mask&(assignment==fold)
                for label,pr in [('baseline',a),('adapted',b)]:
                    values,detail=metrics(y[ix],pr[ix])
                    out_rows.append({'model':label,'presentation':presentation,'scope':scope,'fold':fold,**values})
                    if fold=='pooled':
                        letters.extend({'model':label,'presentation':presentation,'scope':scope,'letter':letter,
                            'images':int(detail[3][j]),'precision':float(detail[0][j]),'recall':float(detail[1][j]),'f1':float(detail[2][j])} for j,letter in enumerate(LABELS))
                paired.append({'presentation':presentation,'scope':scope,'fold':fold,
                    'fixed':int(((a[ix]!=y[ix])&(b[ix]==y[ix])).sum()),'lost':int(((a[ix]==y[ix])&(b[ix]!=y[ix])).sum()),
                    'net':int((b[ix]==y[ix]).sum()-(a[ix]==y[ix]).sum())})
    write_csv(OUT/'metrics.csv',out_rows);write_csv(OUT/'per_letter.csv',letters);write_csv(OUT/'paired_changes.csv',paired)
    write_csv(OUT/'fitting.csv',fitting)
    np.savez_compressed(WORK/'predictions.npz',ids=ids,labels=y,fold=assignment,status=status,baseline=baseline,adapted=predicted,selected_view=selected)
    print('Cross-validation complete; selected camera images remain calibration data.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('prepare','train','cv'),required=True)
    args=parser.parse_args();{'prepare':prepare,'train':train,'cv':cv}[args.stage]()


if __name__=='__main__':main()
