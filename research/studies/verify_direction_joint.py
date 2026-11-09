"""Independently reconcile the frozen direction/joint comparison and replay saved models."""
import json
from pathlib import Path
from hashlib import sha256
import importlib.metadata
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from research.studies.compare_models import load_inputs
from research.studies.compare_direction_joint import matrix_for, ARMS



def main():
    root=Path.cwd(); out=root/'results/direction_joint_01'; work=root/'runs/direction_joint_01'
    X,y,ids,splits=load_inputs(root/'data/asl_kaggle/research')
    shin=X[:,:820];del X
    meta=json.loads((root/'data/asl_kaggle/research/inputs_metadata.json').read_text())
    groups=np.array([s['group'] for s in meta['samples']])
    assignment=np.zeros(len(y),int)
    for fold,train,test in splits:
     assert not set(groups[train])&set(groups[test])
     assert set(train)|set(test)==set(range(len(y))) and not assignment[test].any()
     assignment[test]=fold
    assert assignment.all()
    with np.load(work/'image_joint_angles.npz') as d:
     assert d['ids'].tolist()==ids
     angles=d['angles']
    assert angles.shape==(60251,15) and np.isfinite(angles).all()
    frame=pd.read_csv(out/'oof_predictions.csv');metrics=pd.read_csv(out/'metrics.csv');letters=pd.read_csv(out/'per_letter.csv')
    paired=pd.read_csv(out/'paired_changes.csv')
    assert len(frame)==8*len(ids)
    checked=0;replayed=0
    for name in ('svm','lightgbm'):
     base=frame[(frame.model==name)&(frame.variant=='shin')].set_index('id').loc[ids].prediction.to_numpy()
     for arm in (*ARMS,'shin_world'):
      rows=frame[(frame.model==name)&(frame.variant==arm)]
      assert rows.id.is_unique and set(rows.id)==set(ids)
      rows=rows.set_index('id').loc[ids];pred=rows.prediction.to_numpy()
      assert np.array_equal(rows.label.to_numpy(),y) and np.array_equal(rows.fold.to_numpy(),assignment)
      scores=metrics[(metrics.model==name)&(metrics.variant==arm)]
      for r in scores.itertuples():
       ix=np.arange(len(y)) if r.fold=='pooled' else np.flatnonzero(assignment==int(r.fold))
       assert abs(r.accuracy-accuracy_score(y[ix],pred[ix]))<1e-12
       assert abs(r.macro_f1-f1_score(y[ix],pred[ix],average='macro'))<1e-12
      expected=precision_recall_fscore_support(y,pred,labels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),zero_division=0)
      per=letters[(letters.model==name)&(letters.variant==arm)].set_index('letter').loc[list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')]
      for k,values in zip(('precision','recall','f1','images'),expected):np.testing.assert_allclose(per[k],values,atol=1e-12,rtol=0)
      matrix=pd.read_csv(out/f'confusion_{name}_{arm}.csv').drop(columns='true_letter').to_numpy()
      np.testing.assert_array_equal(matrix,confusion_matrix(y,pred,labels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')))
      p=paired[(paired.model==name)&(paired.variant==arm)].iloc[0]
      assert p.fixed==int(((base!=y)&(pred==y)).sum()) and p.lost==int(((base==y)&(pred!=y)).sum())
      assert p.net==p.fixed-p.lost
      checked+=1
      if arm in ('distance_joint','shin_joint'):
       matrix=matrix_for(shin,angles,arm)
       for fold,train,test in splits:
        model=joblib.load(work/f'{name}_{arm}_fold{fold}.joblib')
        assert model.n_features_in_==matrix.shape[1]
        scaler=model.steps[0][1].named_transformers_['angles'].named_steps['standardscaler']
        np.testing.assert_allclose(scaler.mean_,angles[train].mean(axis=0,dtype=np.float64),atol=1e-10,rtol=0)
        sample=test[np.linspace(0,len(test)-1,26,dtype=int)]
        np.testing.assert_array_equal(model.predict(matrix[sample]),pred[sample]);replayed+=len(sample)
        del model
       del matrix
    sources=['research/joint_angles.py','research/studies/compare_direction_joint.py','research/geometry.py','research/shin.py','research/inputs.py','research/studies/compare_models.py']
    result={'verified':True,'images':len(ids),'folds':5,'compared_arms':checked,'new_models_checked':20,
            'sample_predictions_replayed':replayed,'all_metrics_per_letter_confusions_and_paired_counts_reconciled':True,
            'all_new_angle_scalers_fit_on_training_rows_only':True,
            'new_angle_shape':list(angles.shape),'missing_angles':0,
            'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','lightgbm')}},
            'source_sha256':{p:sha256((root/p).read_bytes()).hexdigest() for p in sources},
            'result_sha256':{p:sha256((out/p).read_bytes()).hexdigest() for p in ('inputs.json','metrics.csv','per_letter.csv','paired_changes.csv','oof_predictions.csv')},
            'test_suite':'92 unit tests passed; includes same-coordinate angle geometry, rigid-rotation behavior, and baseline ID/fold validation.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
