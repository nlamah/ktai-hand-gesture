"""Rebuild the central input cache from frozen landmarks, without redetection."""
import argparse,json
from hashlib import sha256
from pathlib import Path
import numpy as np
from research.inputs import encode_additions,encode_angles
from research.shin import combined_features
from research.geometry import measure_hand_features
from research.references import load_reference_workbook
from sklearn.model_selection import StratifiedGroupKFold

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('data/asl_kaggle'));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 if args.out.exists():raise FileExistsError('Choose a new cache directory')
 args.out.mkdir(parents=True)
 raw=(args.project/'manifest.json').read_bytes();manifest=json.loads(raw)
 samples=[s for s in manifest['samples'] if s.get('included') and s.get('role')=='development' and s.get('status')=='ok']
 refs=load_reference_workbook(Path('assets/asl_feature_references.xlsx').read_bytes());parameters=json.loads(Path('config/thresholds.json').read_text())
 X=np.empty((len(samples),916),np.float32);angles=np.empty((len(samples),15),np.float32)
 for i,s in enumerate(samples):
  m=measure_hand_features(s['world_landmarks'],s.get('handedness'));c,e=encode_additions(m,parameters,refs)
  X[i]=np.r_[list(combined_features(s['image_landmarks']).values()),c,e];angles[i]=encode_angles(m)
  if (i+1)%5000==0:print(f'{i+1}/{len(samples)}',flush=True)
 np.savez_compressed(args.out/'inputs.npz',X=X,angles=angles)
 meta={'hashes':{'manifest':sha256(raw).hexdigest(),'reference':sha256(Path('assets/asl_feature_references.xlsx').read_bytes()).hexdigest(),'thresholds':sha256(Path('config/thresholds.json').read_bytes()).hexdigest(),'proposal_changes':sha256(Path('config/proposal_changes.json').read_bytes()).hexdigest()},'samples':[{'id':s['id'],'label':s['label'],'group':s['session']} for s in samples]}
 (args.out/'inputs_metadata.json').write_text(json.dumps(meta))
 y=np.array([s['label'] for s in samples]);g=np.array([s['session'] for s in samples]);folds=[]
 for fold,(a,b) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,g),1):folds.append({'fold':fold,'train_groups':sorted(set(g[a])),'test_groups':sorted(set(g[b]))})
 (args.out/'summary.json').write_text(json.dumps({'folds':folds},indent=2))
if __name__=='__main__':main()
