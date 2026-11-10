"""Frozen detector-Right canonicalization, compared with both existing mirror arms."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from research.data import atomic_json
from research.handedness import VERSION, fit_canonical, right_hand_detection, right_hand_inputs
from research.inputs import ANGLE_COLUMNS, LABELS, write_csv
from research.live_model import BUNDLE_VERSION, SHIN_VERSION, detection_inputs, load_bundle
from research.studies.mirror_augmentation import assignments_for, balanced_views, load_extracted, metrics

OUT = Path('results/handedness_01')
WORK = Path('runs/handedness_01')
MIRROR = Path('results/mirror_01')
MIRROR_WORK = Path('runs/mirror_01')
FOLDS = Path('data/asl_kaggle/research/summary.json')
MODEL = Path('models/asl_shin_angles_right_hand.joblib')
ARMS = ('original_training', 'mirror_training', 'right_normalized')
VIEWS = ('original', 'mirrored_rgb', 'balanced_mix')


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True);WORK.mkdir(parents=True, exist_ok=True)
    paths = sorted((MIRROR_WORK/'detections').glob('chunk_*.npz'))
    assert len(paths) == 78
    source_files = ('research/handedness.py', 'research/live_model.py',
                    'research/studies/handedness_normalization.py', 'research/shin.py',
                    'research/geometry.py', 'research/inputs.py')
    protocol = {
        'version': VERSION, 'models': ['svm','lightgbm'], 'features':835,
        'rule':'Detector Left: image x=1-x, world x=-x, handedness Right; y/z and anatomical IDs unchanged. Detector Right: unchanged.',
        'training':'Exactly one canonical row per accepted original source; no mirrored RGB rows added; weight 1.',
        'test':'Canonicalize each independently detected held-out original or RGB mirror using its own detector handedness.',
        'controls':'Existing original-only and RGB-mirror-augmented arms on identical sources/folds/views.',
        'settings':'Reuse original-only train-fold preprocessing, resolved SVM gamma and original balanced class priors. No tuning.',
        'unknown_handedness':'Reject; never infer from true class or select orientation by prediction.',
        'seed':20261106, 'failures':'No prediction (?) counts as false negative in all-attempt metrics.',
        'scope':'Exploratory filename-block CV on existing data; detector handedness is not verified human handedness; synthetic views are not independent people.',
        'folds_sha256':digest(FOLDS),
        'extraction_complete_sha256':digest(MIRROR/'extraction_complete.json'),
        'extraction_protocol_sha256':digest(MIRROR/'extraction_protocol.json'),
        'historical_predictions_sha256':digest(MIRROR/'predictions.csv'),
        'detection_chunks_sha256':{str(p):digest(p) for p in paths},
        'source_sha256':{p:digest(p) for p in source_files},
    }
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Frozen normalization study changed; use fresh output paths.')
    atomic_json(path,protocol)
    return protocol


def inputs():
    x,y,ids,groups,status,hands=load_extracted(MIRROR_WORK,MIRROR)
    assignments=assignments_for(groups,json.loads(FOLDS.read_text())['folds'])
    selected=balanced_views(y,assignments)
    canonical=x.copy()
    accepted=status=='ok'
    canonical[accepted]=right_hand_inputs(x[accepted],hands[accepted])
    return x,canonical,y,ids,groups,status,hands,assignments,selected


def historical(name, ids, y, groups, status, hands, assignments, selected):
    data=pd.read_csv(MIRROR/'predictions.csv',keep_default_na=False)
    data=data[data.model==name]
    result={arm:np.full(status.shape,'?',dtype='<U1') for arm in ARMS[:2]}
    for view,title in enumerate(VIEWS[:2]):
        part=data[data.view==title].set_index('id').loc[ids]
        for field,expected in [('label',y),('group',groups),('fold',assignments),
                               ('status',status[:,view]),('detected_handedness',hands[:,view]),
                               ('balanced_selected',(selected==view).astype(int))]:
            np.testing.assert_array_equal(part[field].to_numpy(),expected)
        for arm in ARMS[:2]:result[arm][:,view]=part[arm].to_numpy()
    return result


def geometry_check():
    checked=0;max_difference=0.
    for path in sorted((MIRROR_WORK/'detections').glob('chunk_*.npz')):
        with np.load(path) as data:
            for hand in ('Left','Right'):
                indices=np.argwhere((data['status']=='ok')&(data['handedness']==hand))
                for i,view in indices[:2]:
                    d={'status':'ok','handedness':hand,
                       'image_landmarks':data['image_landmarks'][i,view].tolist(),
                       'world_landmarks':data['world_landmarks'][i,view].tolist()}
                    raw=detection_inputs(d)
                    direct=detection_inputs(right_hand_detection(d))
                    shortcut=right_hand_inputs(raw,[hand])
                    np.testing.assert_allclose(shortcut,direct,atol=3e-5,rtol=1e-6)
                    max_difference=max(max_difference,float(np.max(np.abs(shortcut-direct))))
                    checked+=1
    atomic_json(OUT/'geometry_verification.json',{'status':'passed','raw_detections_checked':checked,
        'maximum_feature_difference':max_difference,'scope':'Raw landmark recomputation versus cached-feature mathematical shortcut, both hand labels and both view types.'})


def cv(name):
    x,canonical,y,ids,groups,status,hands,assignment,selected=inputs()
    fitting=[]
    for fold in sorted(set(assignment)):
        train=(assignment!=fold)&(status[:,0]=='ok');test=assignment==fold
        assert not set(groups[train])&set(groups[test])
        path=WORK/f'{name}_canonical_fold{fold}.joblib'
        checkpoint=WORK/f'{name}_fold{fold}.npz'
        started=time.monotonic()
        baseline=joblib.load(MIRROR_WORK/f'{name}_original_fold{fold}.joblib')
        if path.exists():candidate=joblib.load(path)
        else:
            candidate=fit_canonical(baseline,canonical[train,0],y[train])
            joblib.dump(candidate,path,compress=3)
        assert len(candidate.classes_)==26
        np.testing.assert_array_equal(candidate[0].transform(x[train,0]),baseline[0].transform(x[train,0]))
        if name=='svm':
            assert candidate[-1]._gamma==baseline[-1]._gamma
            np.testing.assert_array_equal(candidate[-1].class_weight_,baseline[-1].class_weight_)
        if checkpoint.exists():
            with np.load(checkpoint) as saved:
                np.testing.assert_array_equal(saved['ids'],ids[test]);prediction=saved['prediction']
        else:
            prediction=np.full(status[test].shape,'?',dtype='<U1')
            for view in range(2):
                good=status[test,view]=='ok'
                prediction[good,view]=candidate.predict(canonical[test,view][good])
            np.savez_compressed(checkpoint,ids=ids[test],prediction=prediction)
        fitting.append({'model':name,'fold':int(fold),'training_sources':int(train.sum()),
            'canonical_rows':int(train.sum()),'left_sources_reflected':int((train&(hands[:,0]=='Left')).sum()),
            'held_out_sources':int(test.sum()),'train_ids_sha256':sha256('\n'.join(ids[train]).encode()).hexdigest(),
            'test_ids_sha256':sha256('\n'.join(ids[test]).encode()).hexdigest(),
            'estimator_sha256':digest(path),'seconds':time.monotonic()-started})
        write_csv(OUT/f'fitting_{name}.csv',fitting)
        print(json.dumps(fitting[-1]),flush=True)
    print(f'{name}: all five normalization folds complete.',flush=True)


def export():
    x,canonical,y,ids,groups,status,hands,assignment,selected=inputs()
    rows=[];letters=[];paired=[];hand_rows=[];prediction_rows=[]
    ix=np.arange(len(y))
    for name in ('svm','lightgbm'):
        arms=historical(name,ids,y,groups,status,hands,assignment,selected)
        candidate=np.full(status.shape,'?',dtype='<U1')
        for fold in sorted(set(assignment)):
            test=assignment==fold
            with np.load(WORK/f'{name}_fold{fold}.npz') as saved:
                np.testing.assert_array_equal(saved['ids'],ids[test]);candidate[test]=saved['prediction']
        arms['right_normalized']=candidate
        for presentation in VIEWS:
            view=selected if presentation=='balanced_mix' else np.full(len(y),VIEWS.index(presentation))
            accepted=status[ix,view]=='ok'
            for scope in ('all_attempts','accepted_only','paired_both_accepted'):
                mask=np.ones(len(y),bool) if scope=='all_attempts' else accepted if scope=='accepted_only' else (status=='ok').all(axis=1)
                for fold in ('pooled',*sorted(set(assignment))):
                    keep=mask if fold=='pooled' else mask&(assignment==fold)
                    for arm,p in arms.items():
                        values,detail=metrics(y[keep],p[ix,view][keep])
                        rows.append({'model':name,'arm':arm,'presentation':presentation,'scope':scope,'fold':fold,**values})
                        if fold=='pooled':
                            letters.extend({'model':name,'arm':arm,'presentation':presentation,'scope':scope,
                                'letter':l,'images':int(detail[3][j]),'precision':float(detail[0][j]),
                                'recall':float(detail[1][j]),'f1':float(detail[2][j])} for j,l in enumerate(LABELS))
                    for reference in ARMS[:2]:
                        a=arms[reference][ix,view][keep];b=candidate[ix,view][keep];truth=y[keep]
                        paired.append({'model':name,'reference':reference,'presentation':presentation,'scope':scope,'fold':fold,
                            'images':int(keep.sum()),'fixed':int(((a!=truth)&(b==truth)).sum()),
                            'lost':int(((a==truth)&(b!=truth)).sum()),'net':int((b==truth).sum()-(a==truth).sum())})
            for hand in ('Left','Right'):
                keep=accepted&(hands[ix,view]==hand)
                for arm,p in arms.items():
                    values,_=metrics(y[keep],p[ix,view][keep])
                    hand_rows.append({'model':name,'arm':arm,'presentation':presentation,'detector_hand':hand,
                        'scope':'accepted_only_macro_over_AZ_missing_classes_zero',**values})
        for i in range(len(y)):
            for view in range(2):
                prediction_rows.append({'model':name,'id':ids[i],'label':y[i],'group':groups[i],
                    'fold':int(assignment[i]),'view':VIEWS[view],'status':status[i,view],
                    'detector_hand':hands[i,view],'balanced_selected':int(selected[i]==view),
                    **{arm:pred[i,view] for arm,pred in arms.items()}})
    write_csv(OUT/'metrics.csv',rows);write_csv(OUT/'per_letter.csv',letters)
    write_csv(OUT/'paired_changes.csv',paired);write_csv(OUT/'handedness_metrics.csv',hand_rows)
    write_csv(OUT/'predictions.csv',prediction_rows)
    valid=status[:,0]=='ok'
    atomic_json(OUT/'summary.json',{'training_originals':int(valid.sum()),
        'original_Left':int((valid&(hands[:,0]=='Left')).sum()),'original_Right':int((valid&(hands[:,0]=='Right')).sum()),
        'balanced_original_views':int((selected==0).sum()),'balanced_reflected_views':int((selected==1).sum()),
        'both_detected':int((status=='ok').all(axis=1).sum()),
        'same_handedness_on_both_views':int(((status=='ok').all(axis=1)&(hands[:,0]==hands[:,1])).sum()),
        'metrics':[r for r in rows if r['fold']=='pooled'],
        'paired':[r for r in paired if r['fold']=='pooled']})
    print('Exported all predictions, folds, letters and paired changes for three matched arms.',flush=True)


def train(protocol):
    x,canonical,y,ids,groups,status,hands,assignment,selected=inputs()
    train=status[:,0]=='ok'
    baseline=joblib.load(MIRROR_WORK/'svm_original_full.joblib')
    if MODEL.exists():
        bundle=load_bundle(MODEL)
        assert bundle.get('normalization_protocol_sha256')==digest(OUT/'protocol.json')
    else:
        started=time.monotonic();model=fit_canonical(baseline,canonical[train,0],y[train])
        bundle={'version':BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':list(ANGLE_COLUMNS),
            'sklearn_version':sklearn.__version__,'estimator':model,'hand_normalization':VERSION,
            'normalization_protocol_sha256':digest(OUT/'protocol.json'),
            'training_images':int(train.sum()),'training_rows':int(train.sum()),
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'fit_seconds':time.monotonic()-started,
            'scope':'Experimental canonical detector-Right landmarks before features; all accepted original development sources; no independent live score.'}
        temp=MODEL.with_suffix('.partial.joblib');joblib.dump(bundle,temp,compress=3)
        load_bundle(temp);temp.replace(MODEL)
        atomic_json(MODEL.with_suffix('.json'),{k:v for k,v in bundle.items() if k!='estimator'})
    atomic_json(OUT/'model.json',{'path':str(MODEL),'sha256':digest(MODEL),
        'training_images':int(train.sum()),'normalization':VERSION})
    print(f'Saved canonical live model: {MODEL}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','geometry','cv','export','train'),required=True)
    parser.add_argument('--model',choices=('svm','lightgbm'),default='svm')
    args=parser.parse_args();protocol=freeze()
    if args.stage=='geometry':geometry_check()
    elif args.stage=='cv':cv(args.model)
    elif args.stage=='export':export()
    elif args.stage=='train':train(protocol)


if __name__=='__main__':main()
