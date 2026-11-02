"""Fixed camera-axis direction augmentation, retaining 835 inputs (default x +/-15)."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import sklearn

from research.inputs import ANGLE_COLUMNS, LABELS, write_csv
from research.live_model import BUNDLE_VERSION, DEFAULT_MODEL, SHIN_VERSION, load_bundle
from research.protocol import ProtocolSession
from research.studies.compare_angles import classifier
from research.studies.compare_models import load_inputs

ANGLES = (0., -15., 15.)
DEFAULT_CANDIDATE = Path('models/asl_shin_angles_tilt15.joblib')


def rotate_directions(X, degrees, axis="x"):
    """Rotate the 210 image-space unit vectors; keep distances/joint angles exact."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 835 or not np.isfinite(X).all():
        raise ValueError('Expected finite N x 835 inputs.')
    a = np.deg2rad(degrees)
    c,s = np.cos(a),np.sin(a)
    matrices = {"x": [[1,0,0],[0,c,-s],[0,s,c]],
                "y": [[c,0,s],[0,1,0],[-s,0,c]],
                "z": [[c,-s,0],[s,c,0],[0,0,1]]}
    if axis not in matrices or not np.isfinite(degrees):
        raise ValueError("Expected a finite angle and axis x, y or z.")
    rotation = np.array(matrices[axis], dtype=X.dtype)
    result = X.copy()
    result[:,190:820] = (X[:,190:820].reshape(-1,210,3) @ rotation.T).reshape(-1,630)
    return result


def augment(X, y, letters=None, axis="x", angles=ANGLES):
    y = np.asarray(y)
    angles = tuple(angles)
    if len(angles)<2 or angles[0]!=0 or not np.isfinite(angles).all() or len(set(angles))!=len(angles):
        raise ValueError("Use distinct finite angles beginning with zero.")
    if len(X)!=len(y):raise ValueError("Input and label counts differ.")
    count = len(angles)
    if letters is None:
        return (np.concatenate([rotate_directions(X,a,axis) for a in angles]),
                np.tile(y,count), np.full(len(y)*count,1./count))
    selected = tuple(letters)
    if not selected or set(selected)-set(LABELS):
        raise ValueError('Choose one or more letters A-Z.')
    mask = np.isin(y,selected)
    return (np.concatenate([X.copy()]+[rotate_directions(X[mask],a,axis) for a in angles[1:]]),
            np.concatenate([y]+[y[mask] for _ in angles[1:]]),
            np.r_[np.where(mask,1./count,1.),np.full((count-1)*mask.sum(),1./count)])


def fit_candidate(X,y,letters=None,axis="x",angles=ANGLES):
    aug, target, weights = augment(X,y,letters,axis,angles)
    model = classifier('svm','shin_angles',len(y))
    if letters is None:
        model.fit(aug,target,svc__sample_weight=weights)
    else:
        # Selective duplication must not change class priors, angle scaling or gamma.
        # Learn these on original fitting rows only, then augment the SVM fit.
        from sklearn.utils.class_weight import compute_class_weight
        preprocessing,svc = model.steps[0][1],model.steps[-1][1]
        original = preprocessing.fit_transform(X,y)
        gamma = 1./(original.shape[1]*np.asarray(original,dtype=np.float64).var())
        classes = np.unique(y)
        class_weights = dict(zip(classes,compute_class_weight('balanced',classes=classes,y=y)))
        svc.set_params(class_weight=class_weights,gamma=gamma)
        svc.fit(preprocessing.transform(aug),target,sample_weight=weights)
    return model


def metrics(y,pred,labels):
    return {'images':len(y),'correct':int((np.asarray(y)==np.asarray(pred)).sum()),
            'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--letters',nargs='+',choices=LABELS,help='Augment selected letters only; default all letters.')
    parser.add_argument('--axis',choices=('x','y','z'),default='x')
    parser.add_argument('--degrees',type=float,nargs='+',default=list(ANGLES),help='Distinct rotations including original zero first.')
    parser.add_argument('--stage',choices=('train','cv','evaluate'),required=True)
    parser.add_argument('--cache',type=Path,default=Path('data/asl_kaggle/research'))
    parser.add_argument('--baseline',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--candidate',type=Path,default=DEFAULT_CANDIDATE)
    parser.add_argument('--out',type=Path,default=Path('results/tilt_augmentation_01'))
    parser.add_argument('--development',type=Path,default=Path('captures/guided/development_20261031T160814Z_3dd178e6'))
    parser.add_argument('--evaluation',type=Path,default=Path('captures/guided/evaluation_20261031T162427Z_bddd0d80'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if len(args.degrees)<2 or args.degrees[0]!=0 or not np.isfinite(args.degrees).all() or len(set(args.degrees))!=len(args.degrees):
        parser.error('--degrees must contain distinct finite angles beginning with zero')
    baseline_sha=sha256(args.baseline.read_bytes()).hexdigest()
    meta=json.loads((args.cache/'inputs_metadata.json').read_text())
    if load_bundle(args.baseline).get('source_hashes') != meta['hashes']:
        raise ValueError('Baseline and candidate must use the same frozen inputs')
    config={'recipe':f'camera-{args.axis} direction rotation','angles_deg':args.degrees,'variants_tried':1,
            'features':835,'all_26_training_classes_augmented':True,'weight_per_row':1./len(args.degrees),
            'svm':'C=10 RBF gamma=scale balanced','baseline_sha256':baseline_sha,
            'input_hashes':meta['hashes'],'cache_sha256':sha256((args.cache/'inputs.npz').read_bytes()).hexdigest(),
            'development':str(args.development.resolve()),'evaluation':str(args.evaluation.resolve()),
            'policy':'One fixed candidate; no hyperparameter search or selection using evaluation outcomes. Captures never enter training.',
            'limitation':'Rotation in normalized landmark-feature space is an approximation, not a physical wrist or detector simulation.'}
    if args.letters:
        config.update(all_26_training_classes_augmented=False,augmented_letters=sorted(set(args.letters)),
                      weight_per_row={'selected_class':1./len(args.degrees),'other_classes':1.},
                      preprocessing_policy='Fit scaling, balanced class weights and gamma=scale on original fitting rows before augmentation')
    config_path=args.out/'protocol.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:raise ValueError('Experiment configuration changed')
    else:
        config_path.write_text(json.dumps(config,indent=2)+'\n')
    arrays=np.load(args.cache/'inputs.npz');X=np.c_[arrays['X'][:,:820],arrays['angles']]
    y=np.array([s['label'] for s in meta['samples']]);ids=[s['id'] for s in meta['samples']]
    assert X.shape==(60251,835) and len(set(ids))==len(ids)
    if args.stage=='train':
        if args.candidate.exists():raise FileExistsError('Candidate already exists; use another experiment path to avoid overwriting')
        n_augmented=len(y) if args.letters is None else int(np.isin(y,args.letters).sum())
        n_rows=len(y)+(len(args.degrees)-1)*n_augmented
        start=time.monotonic();print(f'Training fixed candidate: {len(y)} originals; augment {n_augmented} images; {n_rows} weighted rows.',flush=True)
        model=fit_candidate(X,y,args.letters,args.axis,args.degrees)
        metadata={'version':BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':list(ANGLE_COLUMNS),
                  'sklearn_version':sklearn.__version__,'training_images':len(y),'training_rows':n_rows,
                  'source_hashes':meta['hashes'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
                  'augmentation':config,'fit_seconds':time.monotonic()-start,
                  'scope':'Final deployment fit on original Kaggle training data and synthetic direction rotations; no captures used for fitting.'}
        args.candidate.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.candidate.with_suffix('.partial.joblib')
        try:
            joblib.dump({**metadata,'estimator':model},temporary,compress=3);load_bundle(temporary);temporary.replace(args.candidate)
        finally:temporary.unlink(missing_ok=True)
        args.candidate.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(f'Candidate saved in {metadata["fit_seconds"]:.1f}s: {args.candidate}',flush=True)
    elif args.stage=='cv':
        # The cache supplies the original disjoint group folds; augment only fitting rows.
        from research.studies.compare_models import load_inputs
        _,yy,ii,splits=load_inputs(args.cache);assert np.array_equal(yy,y) and ii==ids
        import csv
        frozen_config=json.loads(Path('results/angles/inputs.json').read_text())
        assert frozen_config['input_hashes']==meta['hashes']
        with Path('results/angles/oof_predictions_svm.csv').open(encoding='utf-8-sig') as handle:
            frozen={r['id']:r for r in csv.DictReader(handle) if r['variant']=='shin_angles'}
        groups=np.array([s['group'] for s in meta['samples']]);predicted=np.full(len(y),'',dtype='<U1');baseline=np.array([frozen[i]['prediction'] for i in ids]);fold_rows=[]
        for fold,fit,test in splits:
            assert not set(groups[fit])&set(groups[test])
            assert all(int(frozen[ids[i]]['fold'])==fold and frozen[ids[i]]['label']==y[i] for i in test)
            checkpoint=args.out/f'fold_{fold}.npz'
            if checkpoint.exists():
                saved=np.load(checkpoint);assert saved['ids'].tolist()==[ids[i] for i in test];pred=saved['prediction']
            else:
                print(f'CV fold {fold}/5: augmenting {len(fit)} fitting images only.',flush=True)
                model=fit_candidate(X[fit],y[fit],args.letters,args.axis,args.degrees);pred=model.predict(X[test]);np.savez_compressed(checkpoint,ids=np.array(ids)[test],prediction=pred)
            predicted[test]=pred
            row={'fold':fold,**metrics(y[test],pred,LABELS),'baseline_accuracy':float(accuracy_score(y[test],baseline[test]))};fold_rows.append(row);print(json.dumps(row),flush=True)
        assert not np.any(predicted=='')
        result={'baseline':metrics(y,baseline,LABELS),'tilt15':metrics(y,predicted,LABELS),'folds':fold_rows,
                'fixed':int(((baseline!=y)&(predicted==y)).sum()),'lost':int(((baseline==y)&(predicted!=y)).sum()),
                'note':'Pooled 26-letter out-of-fold scores. Original fold validation images are never augmented. Candidate key tilt15 is historical; see protocol.json for actual rotations.'}
        (args.out/'cv_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        write_csv(args.out/'cv_predictions.csv',[{'id':ids[i],'label':str(y[i]),'baseline':str(baseline[i]),'tilt15':str(predicted[i])} for i in range(len(y))])
        print(json.dumps(result),flush=True)
    else:
        models={'baseline':load_bundle(args.baseline)['estimator'],'tilt15':load_bundle(args.candidate)['estimator']}
        from research.live_model import detection_inputs
        result={};rows=[]
        for role,path in [('development',args.development),('evaluation',args.evaluation)]:
            session=ProtocolSession(path,baseline_sha)
            if session.plan['role']!=role:raise ValueError('Protocol session role does not match')
            if session.current is not None:
                if role=='development':raise ValueError('Development session is incomplete')
                result[role]={'status':'incomplete','completed':len(session.completed),'planned':27,'scores_not_computed':True}
                continue
            samples=[];truth=[];raw=[]
            for task in session.plan['tasks']:
                p=Path(session.completed[task['id']]);r=json.loads((p/'capture.json').read_text())
                for name,info in r['files'].items():assert sha256((p/name).read_bytes()).hexdigest()==info['sha256']
                truth.append(task['letter']);samples.append((p,r,task))
                row=detection_inputs(r['detection'])[0] if r['detection']['status']=='ok' else None
                if row is not None:np.testing.assert_array_equal(row,np.array(r['raw_model_features'],np.float32))
                raw.append(row)
            truth=np.array(truth);result[role]={}
            for name,model in models.items():
                pred=np.array([str(model.predict(row.reshape(1,-1))[0]) if row is not None else '?' for row in raw])
                if name=='baseline':assert all(p==str(r['predicted_letter'] or '?') for p,(_,r,_) in zip(pred,samples))
                entry=metrics(truth,pred,list('CDO'));entry['unknown_predictions']=int((pred=='?').sum());entry['confusion_CDO']=confusion_matrix(truth,pred,labels=list('CDO')).tolist();entry['by_letter']={}
                precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=list('CDO'),zero_division=0)
                for j,letter in enumerate('CDO'):entry['by_letter'][letter]={'images':int(support[j]),'precision':float(precision[j]),'recall':float(recall[j]),'f1':float(f1[j])}
                result[role][name]=entry
                for i,(p,r,t) in enumerate(samples):rows.append({'role':role,'model':name,'id':r['id'],'task':t['id'],'letter':t['letter'],'tilt':t['tilt'],'repeat':t['repeat'],'prediction':str(pred[i]),'correct':bool(pred[i]==truth[i]),'capture_json_sha256':sha256((p/'capture.json').read_bytes()).hexdigest()})
        (args.out/'capture_summary.json').write_text(json.dumps(result,indent=2)+'\n');write_csv(args.out/'capture_predictions.csv',rows)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
