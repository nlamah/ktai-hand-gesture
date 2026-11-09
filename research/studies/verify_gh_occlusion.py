"""Independent reconciliation of proxy scores, split membership and fitted controls."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from research.data import atomic_json
from research.gh_vertical import rotate_vertical
from research.inputs import LABELS
from research.studies.gh_occlusion import OUT,WORK,PREVIOUS,digest
from research.studies.gh_rotation_standalone import inputs


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,expected in {**protocol['inputs'],**protocol['source_sha256']}.items():assert digest(path)==expected,path
    extra=json.loads((OUT/'synthetic_test_protocol.json').read_text())
    for path,expected in extra['source_sha256'].items():assert digest(path)==expected,path
    X,y,ids,splits=inputs();checked=0;replayed=0
    for prefix in ('','synthetic_test_'):
        preds=pd.read_csv(OUT/f'{prefix}predictions.csv',keep_default_na=False)
        scores=pd.read_csv(OUT/f'{prefix}metrics.csv',keep_default_na=False)
        letters=pd.read_csv(OUT/('synthetic_test_letters.csv' if prefix else 'per_letter.csv'),keep_default_na=False)
        np.testing.assert_array_equal(preds.id,ids);np.testing.assert_array_equal(preds.label,y)
        for r in scores.itertuples(index=False):
            keep=np.ones(len(y),bool) if r.fold=='pooled' else preds.fold.to_numpy()==int(r.fold)
            truth=y[keep];prediction=preds[r.model].to_numpy()[keep]
            p,rec,f1,n=precision_recall_fscore_support(truth,prediction,labels=LABELS,zero_division=0)
            np.testing.assert_allclose([r.accuracy,r.macro_precision,r.macro_recall,r.macro_f1],
                                      [accuracy_score(truth,prediction),p.mean(),rec.mean(),f1.mean()],rtol=0,atol=1e-12)
            assert r.correct==int((truth==prediction).sum()) and r.images==len(truth)
            detail=letters[(letters.model==r.model)&(letters.fold==r.fold)].set_index('letter').loc[LABELS]
            np.testing.assert_allclose(detail[['precision','recall','f1']].to_numpy(),np.c_[p,rec,f1],rtol=0,atol=1e-12)
            checked+=1
        paired=pd.read_csv(OUT/f'{prefix}paired.csv',keep_default_na=False)
        for r in paired.itertuples(index=False):
            keep=np.ones(len(y),bool) if r.fold=='pooled' else preds.fold.to_numpy()==int(r.fold)
            a=preds[getattr(r,'reference','original')].to_numpy()[keep];b=preds[r.model].to_numpy()[keep];truth=y[keep]
            fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
            assert (r.fixed,r.lost,r.net)==(fixed,lost,fixed-lost)
        for fold,tr,te in splits:
            assert np.all(preds.fold.to_numpy()[te]==fold)
            for arm in ('noise002','noise005'):
                model=joblib.load(WORK/f'{arm}_fold{fold}.joblib');base=joblib.load(PREVIOUS/f'original_fold{fold}.joblib')
                assert model[-1]._gamma==base[-1]._gamma
                classes=np.unique(y[tr])
                assert model[-1].class_weight==dict(zip(classes,compute_class_weight('balanced',classes=classes,y=y[tr])))
                sample=np.array([te[np.flatnonzero(y[te]==letter)[0]] for letter in LABELS])
                rows=X[sample].copy()
                if prefix:
                    gh=np.isin(y[sample],['G','H']);rows[gh]=rotate_vertical(rows[gh])
                np.testing.assert_array_equal(model.predict(rows),preds[arm].to_numpy()[sample]);replayed+=len(sample)
                # J-only imputer/scaler must be identical to the original fit.
                np.testing.assert_array_equal(model[0].transform(X[sample]),base[0].transform(X[sample]))
    screen=json.loads((OUT/'render_screen.json').read_text())
    assert screen['sources']==64 and screen['counts']=={'original':{'ok':64},'volume0':{'no_hand':64},'volume180':{'no_hand':64}}
    np.testing.assert_array_equal(pd.read_csv(OUT/'predictions.csv')['original'],pd.read_csv('results/rotation_GH_y180_01/predictions.csv')['original'])
    atomic_json(OUT/'verification.json',{'status':'passed','metric_rows_recomputed':checked,
            'model_predictions_replayed':replayed,'checks':'Frozen source/input hashes; all pooled/fold macro and letter scores; paired changes; held-out IDs; original-only preprocessing; synthetic-view model replay; fixed render-screen denominators',
            'training_sources':60251,'GH_sources':4801,'folds':5,'new_models':10,
            'real_dorsal_evaluation_sources':0,'withdrawn_capture_sources':0,
            'deployment_model_sha256':digest('models/asl_shin_angles.joblib'),
            'original_rotation_model_sha256':digest('models/asl_shin_angles_GH_y180.joblib'),
            'synthetic_test_results_sha256':{str(p):digest(p) for p in OUT.glob('synthetic_test_*.csv')}})
    print(f'Passed: {checked} metric rows; {replayed} held-out predictions replayed.',flush=True)


if __name__=='__main__':main()
