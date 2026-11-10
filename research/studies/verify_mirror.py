"""Recompute mirror metrics and replay held-out models and live bundles independently."""
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from research.data import atomic_json
from research.inputs import LABELS
from research.live_model import detection_inputs, load_bundle, predict_letter
from research.studies.mirror_augmentation import load_extracted, assignments_for, balanced_views, ARMS, VIEWS


def main():
    out, work = Path('results/mirror_01'), Path('runs/mirror_01')
    protocol=json.loads((out/'protocol.json').read_text())
    extraction=json.loads((out/'extraction_protocol.json').read_text())
    for config,key in [(extraction,'source_sha256'),(protocol,'training_source_sha256')]:
        for name,digest in config[key].items():
            assert sha256(Path(name).read_bytes()).hexdigest()==digest,name
    folds_path=Path('data/asl_kaggle/research/summary.json')
    assert sha256(folds_path.read_bytes()).hexdigest()==protocol['folds_sha256']
    assert sha256((out/'extraction_protocol.json').read_bytes()).hexdigest()==protocol['extraction_sha256']
    X,y,ids,groups,status,hands=load_extracted(work,out)
    assignment=assignments_for(groups,json.loads(folds_path.read_text())['folds'])
    views=balanced_views(y,assignment);index=np.arange(len(y))
    mix=pd.read_csv(out/'test_mix.csv')
    assert mix.id.tolist()==ids.tolist()
    np.testing.assert_array_equal(mix.view.to_numpy(),np.array(VIEWS)[views])
    np.testing.assert_array_equal(mix.fold.to_numpy(),assignment)
    for fold in sorted(set(assignment)):
        for letter in LABELS:
            ix=(assignment==fold)&(y==letter)
            assert abs(int((views[ix]==0).sum())-int((views[ix]==1).sum()))<=1
    frame=pd.read_csv(out/'predictions.csv',keep_default_na=False)
    scores=pd.read_csv(out/'metrics.csv');letters=pd.read_csv(out/'per_letter.csv')
    paired=pd.read_csv(out/'paired_changes.csv');fitting=pd.read_csv(out/'fitting.csv')
    assert len(frame)==len(y)*2*2
    metric_rows=0;models=0;replayed=0
    for name in ('svm','lightgbm'):
        by_view=[]
        for view,label in enumerate(VIEWS):
            rows=frame[(frame.model==name)&(frame.view==label)]
            assert rows.id.is_unique and set(rows.id)==set(ids)
            rows=rows.set_index('id').loc[ids]
            np.testing.assert_array_equal(rows.label.to_numpy(),y)
            np.testing.assert_array_equal(rows.fold.to_numpy(),assignment)
            np.testing.assert_array_equal(rows.group.to_numpy(),groups)
            np.testing.assert_array_equal(rows.status.to_numpy(),status[:,view])
            np.testing.assert_array_equal(rows.balanced_selected.to_numpy(),(views==view).astype(int))
            for arm in ARMS:
                assert (rows[arm].to_numpy()[status[:,view]!='ok']=='?').all()
            by_view.append(rows)
        predictions={arm:np.column_stack([rows[arm].to_numpy() for rows in by_view]) for arm in ARMS}
        for presentation in (*VIEWS,'balanced_mix'):
            selected=np.full(len(y),VIEWS.index(presentation)) if presentation in VIEWS else views
            accepted=status[index,selected]=='ok'
            a,b=[predictions[arm][index,selected] for arm in ARMS]
            for scope in ('all_attempts','accepted_only','paired_both_accepted'):
                mask=np.ones(len(y),bool) if scope=='all_attempts' else accepted if scope=='accepted_only' else (status=='ok').all(axis=1)
                for fold in ('pooled',*sorted(set(assignment))):
                    ix=mask if fold=='pooled' else mask&(assignment==fold)
                    if not ix.any():continue
                    for arm,pred in zip(ARMS,(a,b)):
                        row=scores[(scores.model==name)&(scores.training==arm)&(scores.presentation==presentation)&(scores.scope==scope)&(scores.fold==str(fold))]
                        assert len(row)==1
                        row=row.iloc[0]
                        assert row.images==int(ix.sum()) and row.correct==int((y[ix]==pred[ix]).sum())
                        assert abs(row.accuracy-accuracy_score(y[ix],pred[ix]))<1e-12
                        detail=precision_recall_fscore_support(y[ix],pred[ix],labels=LABELS,zero_division=0)
                        for field,expected in zip(('macro_precision','macro_recall','macro_f1'),detail[:3]):
                            assert abs(row[field]-expected.mean())<1e-12
                        metric_rows+=1
                        if fold=='pooled':
                            per=letters[(letters.model==name)&(letters.training==arm)&(letters.presentation==presentation)&(letters.scope==scope)].set_index('letter').loc[LABELS]
                            for field,expected in zip(('precision','recall','f1','images'),detail):
                                np.testing.assert_allclose(per[field],expected,rtol=0,atol=1e-12)
                            cm=pd.read_csv(out/f'confusion_{name}_{arm}_{presentation}_{scope}.csv').drop(columns='true_letter')
                            np.testing.assert_array_equal(cm,confusion_matrix(y[ix],pred[ix],labels=[*LABELS,'?']))
                    changes=paired[(paired.model==name)&(paired.presentation==presentation)&(paired.scope==scope)&(paired.fold==str(fold))].iloc[0]
                    assert changes.fixed==int(((a[ix]!=y[ix])&(b[ix]==y[ix])).sum())
                    assert changes.lost==int(((a[ix]==y[ix])&(b[ix]!=y[ix])).sum())
                    assert changes.net==changes.fixed-changes.lost
        for fold in sorted(set(assignment)):
            train=(assignment!=fold)&(status[:,0]=='ok');test=assignment==fold
            assert not set(groups[train])&set(groups[test])
            info=fitting[(fitting.model==name)&(fitting.fold==fold)].iloc[0]
            assert sha256('\n'.join(ids[train]).encode()).hexdigest()==info.training_source_ids_sha256
            assert sha256('\n'.join(ids[test]).encode()).hexdigest()==info.held_out_source_ids_sha256
            assert info.original_sources==train.sum()
            assert info.mirror_rows==int((train&(status[:,1]=='ok')).sum())
            baseline=joblib.load(work/f'{name}_original_fold{fold}.joblib')
            for arm,suffix in zip(ARMS,('original','mirrored')):
                model=baseline if suffix=='original' else joblib.load(work/f'{name}_{suffix}_fold{fold}.joblib')
                assert model.n_features_in_==835 and list(model.classes_)==LABELS
                scaler=model[0].named_transformers_['angles'].named_steps['standardscaler']
                np.testing.assert_allclose(scaler.mean_,X[train,0,820:].mean(axis=0,dtype=np.float64),rtol=0,atol=1e-10)
                original_sample=X[np.flatnonzero(train)[:10],0]
                np.testing.assert_allclose(model[0].transform(original_sample),baseline[0].transform(original_sample),rtol=0,atol=0)
                if name=='svm':
                    assert model[-1]._gamma==baseline[-1]._gamma
                    np.testing.assert_array_equal(model[-1].class_weight_,baseline[-1].class_weight_)
                for view in range(2):
                    candidates=np.flatnonzero(test&(status[:,view]=='ok'))
                    ix=candidates[np.linspace(0,len(candidates)-1,26,dtype=int)]
                    np.testing.assert_array_equal(model.predict(X[ix,view]),predictions[arm][ix,view])
                    replayed+=len(ix)
                models+=1
            del baseline,model
    # Validate the actual per-view live encoder on raw detector checkpoints.
    encoded=0
    for path in sorted((work/'detections').glob('chunk_*.npz'))[::8]:
        with np.load(path) as data:
            cached_status=data['status'];cached_features=data['features']
            cached_image=data['image_landmarks'];cached_world=data['world_landmarks'];cached_hands=data['handedness']
            for view in range(2):
                valid=np.flatnonzero(cached_status[:,view]=='ok')
                for k in valid[np.linspace(0,len(valid)-1,4,dtype=int)]:
                    detection={'status':'ok','image_landmarks':cached_image[k,view],
                        'world_landmarks':cached_world[k,view], 'handedness':str(cached_hands[k,view])}
                    row=detection_inputs(detection)[0]
                    np.testing.assert_allclose(row,cached_features[k,view],rtol=2e-5,atol=5e-4)
                    encoded+=1
    live_models=[]
    for path in [Path('models/asl_shin_angles_mirror_baseline.joblib'),Path('models/asl_shin_angles_mirrored.joblib')]:
        bundle=load_bundle(path)
        assert bundle['experiment_protocol']==protocol
        assert bundle['training_images']==int((status[:,0]=='ok').sum())
        live_models.append({'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()})
    hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    result={'verified':True,'source_images':len(y),'detector_inputs':2*len(y),
        'independent_metric_rows_reconciled':metric_rows,'fold_models_checked':models,
        'held_out_predictions_replayed':replayed,'raw_detection_feature_rows_reencoded':encoded,
        'source_groups_disjoint':True,'balanced_mix_reproduced':True,'train_only_scaling_verified':True,
        'svm_gamma_and_class_weights_preserved':True,'live_models':live_models,
        'result_sha256':hashes,'source_sha256':{'research/studies/verify_mirror.py':sha256(Path(__file__).read_bytes()).hexdigest()},
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','lightgbm','mediapipe')},
        'scope':'Computational consistency; no independent validation of signer handedness or live generalization.'}
    atomic_json(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if 'sha256' not in k},indent=2))


if __name__=='__main__':main()
