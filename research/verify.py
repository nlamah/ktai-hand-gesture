"""Audit frozen OOF evidence, groups, input hashes, and optionally every descriptor."""
import argparse,json
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from research.studies.compare_models import load_inputs
from research.inputs import encode_additions,encode_angles
from research.geometry import measure_hand_features
from research.shin import combined_features
from research.references import load_reference_workbook


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('data/asl_kaggle'));p.add_argument('--cache',type=Path,default=Path('data/asl_kaggle/research'));p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--features',action='store_true');p.add_argument('--out',type=Path,default=Path('results/verification_current.json'));args=p.parse_args()
 X,y,ids,splits=load_inputs(args.cache);meta=json.loads((args.cache/'inputs_metadata.json').read_text());report={'images':len(y),'features':X.shape[1]}
 assert X.shape==(60251,931) and len(set(ids))==len(ids) and np.isfinite(X).all()
 assignments=np.zeros(len(y),int)
 for fold,a,b in splits:
  assert not set(a)&set(b) and set(a)|set(b)==set(range(len(y)));assert not assignments[b].any();assignments[b]=fold
 assert assignments.all();report['outer_folds_disjoint_and_complete']=True
 metrics=pd.read_csv(args.results/'angles/metrics.csv');letter=pd.read_csv(args.results/'angles/per_letter.csv');checked=0
 for model in ('svm','lightgbm'):
  frame=pd.read_csv(args.results/f'angles/oof_predictions_{model}.csv')
  for variant in ('shin','shin_angles','shin_excel','shin_angles_excel'):
   f=frame[frame.variant==variant].set_index('id');assert f.index.is_unique and set(f.index)==set(ids);f=f.loc[ids]
   assert np.array_equal(f.label.to_numpy(),y) and np.array_equal(f.fold.to_numpy(),assignments)
   row=metrics[(metrics.model==model)&(metrics.variant==variant)&(metrics.fold=='pooled')].iloc[0]
   assert abs(row.accuracy-accuracy_score(y,f.prediction))<1e-12
   assert abs(row.macro_f1-f1_score(y,f.prediction,average='macro'))<1e-12
   pr,re,f1,n=precision_recall_fscore_support(y,f.prediction,labels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
   detail=letter[(letter.model==model)&(letter.variant==variant)].set_index('letter').loc[list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')]
   for field,expected in [('precision',pr),('recall',re),('f1',f1),('images',n)]:np.testing.assert_allclose(detail[field],expected,rtol=0,atol=1e-12)
   checked+=1
 report['eight_oof_models_and_all_letter_metrics_reconciled']=checked==8
 manifest_raw=(args.project/'manifest.json').read_bytes();assert sha256(manifest_raw).hexdigest()==meta['hashes']['manifest']
 manifest=json.loads(manifest_raw);samples=manifest['samples'];assert len(samples)==78000
 counts=pd.Series([s['status'] for s in samples]).value_counts().to_dict();assert counts=={'ok':60251,'no_hand':17598,'multiple_hands':151};report['status_counts']=counts
 report['manifest_sha256']=sha256(manifest_raw).hexdigest()
 # Compare targeted-weighting predictions and metrics, not just the main eight rows.
 index=pd.read_csv(args.results/'index_weights/oof_predictions.csv');summary=json.loads((args.results/'index_weights/summary.json').read_text())
 for arm in ('unweighted','pip_2','tuned_pip','tuned_index'):
  actual=accuracy_score(index.label,index[arm]);assert abs(actual-summary['metrics'][arm]['accuracy'])<1e-12
 report['index_weight_metrics_reconciled']=True
 if args.features:
  byid={s['id']:s for s in samples};refs=load_reference_workbook(Path('assets/asl_feature_references.xlsx').read_bytes());parameters=json.loads(Path('config/thresholds.json').read_text())
  for i,id_ in enumerate(ids):
   s=byid[id_];m=measure_hand_features(s['world_landmarks'],s.get('handedness'));c,e=encode_additions(m,parameters,refs)
   actual=np.r_[list(combined_features(s['image_landmarks']).values()),c,e,encode_angles(m)].astype(np.float32)
   np.testing.assert_array_equal(actual,X[i])
   if (i+1)%5000==0:print(f'Exact descriptor equality {i+1}/{len(ids)}',flush=True)
  report['all_60251_descriptor_rows_exact_float32_equality']=True
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
