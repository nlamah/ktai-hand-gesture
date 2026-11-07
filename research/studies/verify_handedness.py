"""Recompute every score/paired count and replay held-out and live normalization."""
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from research.data import atomic_json
from research.handedness import right_hand_detection, right_hand_inputs
from research.inputs import LABELS
from research.live_model import detection_inputs, load_bundle, model_inputs
from research.studies.handedness_normalization import OUT, WORK, MIRROR_WORK, MODEL, digest


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for key in ('source_sha256','detection_chunks_sha256'):
        for path,expected in protocol[key].items():assert digest(path)==expected,path
    data=pd.read_csv(OUT/'predictions.csv',keep_default_na=False)
    scores=pd.read_csv(OUT/'metrics.csv',keep_default_na=False)
    paired=pd.read_csv(OUT/'paired_changes.csv',keep_default_na=False)
    per_letter=pd.read_csv(OUT/'per_letter.csv',keep_default_na=False)
    assert len(data)==312000
    assert not data.duplicated(['model','id','view']).any()
    metric_checks=letter_checks=paired_checks=0
    for record in scores.to_dict('records'):
        part=data[data.model==record['model']]
        if record['presentation']=='balanced_mix':part=part[part.balanced_selected==1]
        else:part=part[part.view==record['presentation']]
        if record['scope']=='accepted_only':part=part[part.status=='ok']
        elif record['scope']=='paired_both_accepted':
            all_views=data[data.model==record['model']]
            valid=all_views.assign(valid=all_views.status=='ok').groupby('id').valid.all()
            part=part[part.id.isin(valid[valid].index)]
        if record['fold']!='pooled':part=part[part.fold==int(record['fold'])]
        y=part.label.to_numpy();pred=part[record['arm']].to_numpy()
        p,r,f,n=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
        measured={'images':len(y),'correct':int((y==pred).sum()),'accuracy':float((y==pred).mean()),
                  'macro_precision':p.mean(),'macro_recall':r.mean(),'macro_f1':f.mean()}
        for key,value in measured.items():np.testing.assert_allclose(value,record[key],atol=1e-12,rtol=0)
        metric_checks+=1
        if record['fold']=='pooled':
            selected=per_letter[(per_letter.model==record['model'])&(per_letter.arm==record['arm'])&
                (per_letter.presentation==record['presentation'])&(per_letter.scope==record['scope'])].set_index('letter').loc[LABELS]
            for key,value in [('images',n),('precision',p),('recall',r),('f1',f)]:
                np.testing.assert_allclose(selected[key],value,atol=1e-12,rtol=0)
            letter_checks+=len(LABELS)
        if record['arm']=='right_normalized':
            for reference in ('original_training','mirror_training'):
                base=part[reference].to_numpy()
                changes=paired[(paired.model==record['model'])&(paired.reference==reference)&
                    (paired.presentation==record['presentation'])&(paired.scope==record['scope'])&
                    (paired.fold==record['fold'])]
                assert len(changes)==1
                fixed=int(((base!=y)&(pred==y)).sum());lost=int(((base==y)&(pred!=y)).sum())
                assert (changes.iloc[0][['fixed','lost','net']].to_numpy()==[fixed,lost,fixed-lost]).all()
                paired_checks+=1
    # Saved estimator replay uses only its held-out fold and the actual cached view.
    replayed=0
    # Several chunks cover every fold; use first two valid views from each fold encountered.
    checked=set()
    for path in sorted((MIRROR_WORK/'detections').glob('chunk_*.npz')):
        with np.load(path) as cache:
            for name in ('svm','lightgbm'):
                records=data[(data.model==name)&data.id.isin(cache['ids'])]
                for fold in sorted(records.fold.unique()):
                    for hand in ('Left','Right'):
                        key=(name,int(fold),hand)
                        if key in checked:continue
                        chosen=records[(records.fold==fold)&(records.status=='ok')&(records.detector_hand==hand)].head(2)
                        if len(chosen)==0:continue
                        estimator=joblib.load(WORK/f'{name}_canonical_fold{fold}.joblib')
                        for row in chosen.to_dict('records'):
                            i=int(np.flatnonzero(cache['ids']==row['id'])[0]);view=0 if row['view']=='original' else 1
                            x=right_hand_inputs(cache['features'][i,view][None,:],[row['detector_hand']])
                            assert estimator.predict(x)[0]==row['right_normalized']
                            replayed+=1
                        checked.add(key)
        if len(checked)==20:break
    assert len(checked)==20
    bundle=load_bundle(MODEL)
    live_checked=0
    for hand in ('Left','Right'):
        for path in sorted((MIRROR_WORK/'detections').glob('chunk_*.npz')):
            with np.load(path) as cache:
                matches=np.argwhere((cache['status']=='ok')&(cache['handedness']==hand))
                if not len(matches):continue
                for i,view in matches[:8]:
                    d={'status':'ok','handedness':hand,'image_landmarks':cache['image_landmarks'][i,view].tolist(),
                       'world_landmarks':cache['world_landmarks'][i,view].tolist()}
                    a=model_inputs(bundle,d)
                    b=detection_inputs(right_hand_detection(d))
                    np.testing.assert_array_equal(a,b)
                    shortcut=right_hand_inputs(detection_inputs(d),[hand])
                    np.testing.assert_allclose(a,shortcut,atol=3e-5,rtol=1e-6)
                    np.testing.assert_array_equal(bundle['estimator'].predict(a),bundle['estimator'].predict(shortcut))
                    live_checked+=1
                break
    assert live_checked==16
    atomic_json(OUT/'verification.json',{'status':'passed','metric_rows_recomputed':metric_checks,
        'letter_rows_recomputed':letter_checks,'paired_rows_recomputed':paired_checks,
        'held_out_estimator_predictions_replayed':replayed,'live_raw_landmark_cases':live_checked,
        'source_and_detection_hashes_verified':True,
        'model_sha256':digest(MODEL),'predictions_sha256':digest(OUT/'predictions.csv'),
        'scope':'Computational parity and score verification, not independent handedness or live-view validation.'})
    print(json.dumps(json.loads((OUT/'verification.json').read_text()),indent=2),flush=True)


if __name__=='__main__':main()
