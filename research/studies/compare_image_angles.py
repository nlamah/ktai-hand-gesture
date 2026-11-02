"""One fixed ablation: 835 Shin+3D inputs versus 849 inputs including 2D angles."""
import argparse
import csv
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
import sklearn
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from research.data import atomic_json
from research.image_angles import projected_angles,projected_angles_batch,PIXEL_ANGLE_COLUMNS,VERSION
from research.inputs import ANGLE_COLUMNS,LABELS,angle_estimator,write_csv
from research.live_model import (IMAGE_ANGLE_BUNDLE_VERSION,SHIN_VERSION,load_bundle,model_inputs)
from research.studies.compare_models import load_inputs,model_for

OUT=Path('results/angles_2d_01')
CACHE=Path('data/asl_kaggle/research')
PROJECT=CACHE.parent
MODEL=Path('models/asl_shin_angles_2d.joblib')
BASELINE=Path('models/asl_shin_angles.joblib')


def estimator(n):
    return make_pipeline(angle_estimator(820,849).steps[0][1],model_for('svm',False,n))


def metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    return {'images':len(y),'correct':int((y==pred).sum()),'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    manifest_sha=sha256((PROJECT/'manifest.json').read_bytes()).hexdigest()
    assert manifest_sha==meta['hashes']['manifest'],'Frozen manifest changed'
    config={'variant':'shin_angles_2d','features':849,'extra_columns':list(PIXEL_ANGLE_COLUMNS),
            'pixel_angle_version':VERSION,'input_hashes':meta['hashes'],
            'input_cache_sha256':sha256((CACHE/'inputs.npz').read_bytes()).hexdigest(),
            'folds_sha256':sha256((CACHE/'summary.json').read_bytes()).hexdigest(),
            'baseline_model_sha256':sha256(BASELINE.read_bytes()).hexdigest(),
            'feature_code_sha256':sha256(Path('research/image_angles.py').read_bytes()).hexdigest(),
            'recipe':'Original 835 inputs plus 14 pixel-aspect-correct projected internal joint angles; no augmentation or contact additions.',
            'model':'RBF SVM C=10 gamma=scale balanced; passthrough 820 Shin inputs; train-only median imputation and standard scaling of all 29 angles.',
            'policy':'One fixed candidate, same frozen five group folds. Captures never used to fit or tune; selected captures are development diagnostics only.'}
    file=OUT/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==config,'Protocol changed; use a fresh output directory'
    else:atomic_json(file,config)
    target=OUT/'pixel_angles.npz'
    if not target.exists():
        print('Building 14 projected angles from stored training landmarks; no detector rerun.',flush=True)
        manifest=json.loads((PROJECT/'manifest.json').read_text())
        samples={s['id']:s for s in manifest['samples'] if s['status']=='ok'}
        ordered=[samples[s['id']] for s in meta['samples']]
        assert all(a['label']==b['label'] for a,b in zip(ordered,meta['samples']))
        points=np.array([s['image_landmarks'] for s in ordered])
        sizes=np.array([[s['width'],s['height']] for s in ordered])
        values=projected_angles_batch(points,sizes)
        # Audit both dimensional provenance and exact same train/live calculation.
        for i in np.linspace(0,len(ordered)-1,26,dtype=int):
            with Image.open(PROJECT/ordered[i]['image_path']) as image:assert list(image.size)==sizes[i].tolist()
            np.testing.assert_array_equal(values[i],projected_angles(points[i],sizes[i]))
        np.savez_compressed(target,angles=values,ids=np.array([s['id'] for s in meta['samples']]))
        print(f'Cached {values.shape}; missing projected angles: {int(np.isnan(values).sum())}.',flush=True)
    values=np.load(target);assert values['ids'].tolist()==[s['id'] for s in meta['samples']]
    arrays=np.load(CACHE/'inputs.npz')
    X=np.c_[arrays['X'][:,:820],arrays['angles'],values['angles']].astype(np.float32)
    y=np.array([s['label'] for s in meta['samples']])
    assert X.shape==(60251,849) and not np.isinf(X).any()
    return X,y,meta,config


def train(X,y,meta,config,resume=False):
    if MODEL.exists():
        if not resume:raise FileExistsError('Candidate exists; refusing overwrite')
        assert load_bundle(MODEL)['experiment_protocol']==config
        return
    print('Training full-data 849-input deployment model.',flush=True);start=time.monotonic()
    model=estimator(len(y)).fit(X,y)
    info={'version':IMAGE_ANGLE_BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':list(ANGLE_COLUMNS),
          'pixel_angle_columns':list(PIXEL_ANGLE_COLUMNS),'pixel_angle_version':VERSION,
          'sklearn_version':sklearn.__version__,'training_images':len(y),'source_hashes':meta['hashes'],
          'experiment_protocol':config,'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'fit_seconds':time.monotonic()-start,'scope':'Full accepted Kaggle deployment fit. No webcam captures used for training. Independent final webcam accuracy unknown.'}
    temporary=MODEL.with_suffix('.partial.joblib')
    try:
        joblib.dump({**info,'estimator':model},temporary,compress=3);load_bundle(temporary);temporary.replace(MODEL)
    finally:temporary.unlink(missing_ok=True)
    atomic_json(MODEL.with_suffix('.json'),info)
    print(f'Saved {MODEL} in {info["fit_seconds"]:.1f}s.',flush=True)


def cv(X,y,meta):
    _,yy,ids,splits=load_inputs(CACHE);np.testing.assert_array_equal(y,yy)
    assert ids==[s['id'] for s in meta['samples']]
    assert json.loads(Path('results/angles/inputs.json').read_text())['input_hashes']==meta['hashes']
    with Path('results/angles/oof_predictions_svm.csv').open(encoding='utf-8-sig') as f:
        frozen={r['id']:r for r in csv.DictReader(f) if r['variant']=='shin_angles'}
    baseline=np.array([frozen[i]['prediction'] for i in ids]);pred=np.full(len(y),'',dtype='<U1');folds=[]
    for fold,fit,test in splits:
        assert all(frozen[ids[i]]['label']==y[i] and int(frozen[ids[i]]['fold'])==fold for i in test)
        checkpoint=OUT/f'fold_{fold}.npz'
        if checkpoint.exists():
            saved=np.load(checkpoint);assert saved['ids'].tolist()==[ids[i] for i in test];pr=saved['prediction']
        else:
            print(f'CV {fold}/5: fit {len(fit)}, validate {len(test)} unchanged images.',flush=True)
            model=estimator(len(fit)).fit(X[fit],y[fit]);pr=model.predict(X[test])
            np.savez_compressed(checkpoint,ids=np.array(ids)[test],prediction=pr)
        pred[test]=pr
        folds.append({'fold':fold,'baseline':metrics(y[test],baseline[test]),'image_angles':metrics(y[test],pr)})
        print(json.dumps(folds[-1]),flush=True)
    assert not np.any(pred=='')
    result={'baseline':metrics(y,baseline),'image_angles':metrics(y,pred),'folds':folds,
            'fixed':int(((baseline!=y)&(pred==y)).sum()),'lost':int(((baseline==y)&(pred!=y)).sum())}
    for key in ['accuracy','macro_f1']:
        result[f'fold_{key}_mean_std']={name:{'mean':float(np.mean([f[name][key] for f in folds])),
            'sample_std':float(np.std([f[name][key] for f in folds],ddof=1))} for name in ['baseline','image_angles']}
    atomic_json(OUT/'cv_summary.json',result)
    write_csv(OUT/'cv_predictions.csv',[{'id':ids[i],'label':str(y[i]),'baseline':str(baseline[i]),'image_angles':str(pred[i])} for i in range(len(y))])
    details=[]
    for name,pp in [('baseline',baseline),('image_angles',pred)]:
        precision,recall,f1,support=precision_recall_fscore_support(y,pp,labels=LABELS,zero_division=0)
        details.extend({'model':name,'letter':l,'images':int(support[i]),'correct':int(((y==l)&(pp==l)).sum()),
            'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,l in enumerate(LABELS))
    write_csv(OUT/'cv_per_letter.csv',details)
    print(json.dumps(result),flush=True)


def evaluate():
    models={'baseline':load_bundle(BASELINE),'image_angles':load_bundle(MODEL)}
    rows=[]
    for path in sorted(Path('captures').rglob('capture.json')):
        r=json.loads(path.read_text())
        if r['detection']['status']!='ok':continue
        for name,info in r['files'].items():assert sha256((path.parent/name).read_bytes()).hexdigest()==info['sha256']
        with Image.open(path.parent/'model_input.png') as image:size=list(image.size)
        detection={**r['detection'],'image_size':size}
        predictions={name:str(b['estimator'].predict(model_inputs(b,detection))[0]) for name,b in models.items()}
        rows.append({'id':r['id'],'session_id':r.get('session_id'),'intended':r['expected_letter'],
                     'path':str(path),'saved_prediction':r['predicted_letter'],**predictions})
    write_csv(OUT/'capture_predictions.csv',rows)
    latest=json.loads(Path('results/GH_capture_review_20261031/analysis.json').read_text())
    session=latest['session_id'];selected=[r for r in rows if r['session_id']==session]
    atomic_json(OUT/'capture_summary.json',{'latest_GH_round':{'session_id':session,'images':len(selected),
        'correct':{name:sum(r[name]==r['intended'] for r in selected) for name in models},'predictions':selected},
        'note':'Known selected errors, not an independent webcam test. All saved valid captures replayed without redetection.'})
    print(json.dumps({'latest_GH_round':selected}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','prepare','train','cv','evaluate'],default='all')
    args=p.parse_args()
    if args.stage=='evaluate':evaluate();return
    X,y,meta,config=prepare()
    if args.stage in ['all','train']:train(X,y,meta,config,resume=args.stage=='all')
    if args.stage in ['all','cv']:cv(X,y,meta)
    if args.stage=='all':evaluate()


if __name__=='__main__':main()
