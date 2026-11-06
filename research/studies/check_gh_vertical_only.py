"""Check rotation alone on selected existing captures, without camera calibration."""
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np

from research.data import atomic_json
from research.inputs import write_csv
from research.live_model import load_bundle, predict_letter
from research.studies.gh_vertical_adaptation import OUT, WORK, fit
from research.studies.mirror_augmentation import load_extracted


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    X,y,ids,groups,status,hands=load_extracted(Path('runs/mirror_01'),Path('results/mirror_01'))
    base=load_bundle(Path(protocol['base_model']))
    accepted=status[:,0]=='ok'
    path=WORK/'rotation_only.joblib'
    # A fresh fit: no selected camera images or weights enter this learner.
    model,n_rows=fit(base['estimator'],X[accepted],y[accepted],status[accepted],None)
    bundle={**base,'estimator':model,'scope':'Development ablation: y180 G/H turns, no camera calibration',
            'training_rows':n_rows,'calibration_sources':0}
    joblib.dump(bundle,path,compress=3)
    candidate=load_bundle(path)
    rows=[]
    for capture in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(capture.read_text())
        if r['detection']['status']!='ok':continue
        before=predict_letter(base,r['detection'])['letter']
        after=predict_letter(candidate,r['detection'])['letter']
        rows.append({'id':r['id'],'intended':r['expected_letter'],'baseline':before,'rotation_only':after,
                     'role':'previously_inspected_development_control_not_used_for_this_fit'})
    write_csv(OUT/'rotation_only_capture_replay.csv',rows)
    atomic_json(OUT/'rotation_only_check.json',{
        'model_path':str(path),'model_sha256':sha256(path.read_bytes()).hexdigest(),
        'training_rows':n_rows,'calibration_sources':0,'replayed_selected_captures':len(rows),
        'source_sha256':{p:sha256(Path(p).read_bytes()).hexdigest() for p in
            ('research/gh_vertical.py','research/studies/gh_vertical_adaptation.py',__file__)},
        'role':'Ablation on previously inspected development examples; no independent accuracy claim and no cross-validation for this arm'})
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
