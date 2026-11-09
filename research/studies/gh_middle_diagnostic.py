"""Post-hoc diagnostic: G/H using middle-PIP 3D and 2D only; no tuning."""
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import sklearn

from research.data import atomic_json
from research.geometry import FINGER_JOINTS
from research.image_angles import projected_angles
from research.inputs import ANGLE_COLUMNS,write_csv
from research.live_model import detection_inputs
from research.studies.compare_gh_classifier import load_inputs,classifier,measurements

OUT=Path('results/gh_middle_01')
MODEL=Path('models/gh_diagnostic/gh_middle.joblib')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    w,p,y,ids,groups,meta,summary=load_inputs()
    wi=ANGLE_COLUMNS.index('middle_pip_angle_deg');pi=list(FINGER_JOINTS).index('middle_pip')
    X=np.c_[w[:,wi],p[:,pi]]
    config={'kind':'post-hoc targeted feature ablation','features':['middle_pip_angle_deg','image2d_middle_pip_angle_deg'],
        'input_hashes':meta['hashes'],'folds_sha256':sha256(Path('data/asl_kaggle/research/summary.json').read_bytes()).hexdigest(),
        'rationale':'After the three preregistered representations failed on the newest G, isolate the middle-finger flexion signal and omit the other joints. No hyperparameter tuning.',
        'policy':'Exploratory development diagnostic, not independently confirmed. No captures used for fitting.',
        'model':'RBF SVM C=10 gamma=scale balanced; median imputation and standard scaling on fitting data only'}
    file=OUT/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==config
    else:atomic_json(file,config)
    predicted=np.full(len(y),'',dtype='<U1');assigned=np.zeros(len(y),int);folds=[]
    for f in summary['folds']:
        fit=np.flatnonzero(np.isin(groups,f['train_groups']));test=np.flatnonzero(np.isin(groups,f['test_groups']))
        assert not set(groups[fit])&set(groups[test])
        model=classifier().fit(X[fit],y[fit]);pr=model.predict(X[test]);predicted[test]=pr;assigned[test]+=1
        folds.append({'fold':f['fold'],**measurements(y[test],pr)})
    assert np.all(assigned==1)
    scores=measurements(y,predicted);atomic_json(OUT/'cv_summary.json',{'pooled':scores,'folds':folds})
    write_csv(OUT/'oof_predictions.csv',[{'id':str(ids[i]),'label':str(y[i]),'prediction':str(predicted[i])} for i in range(len(y))])
    if MODEL.exists():
        bundle=joblib.load(MODEL);assert bundle['protocol']==config
    else:
        bundle={'version':'gh-middle-diagnostic-v1','protocol':config,'sklearn_version':sklearn.__version__,'training_images':len(y),'estimator':classifier().fit(X,y)}
        joblib.dump(bundle,MODEL,compress=3)
        atomic_json(MODEL.with_suffix('.json'),{k:v for k,v in bundle.items() if k!='estimator'})
    captures=[]
    for path in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(path.read_text())
        if r['expected_letter'] not in ('G','H') or r['detection']['status']!='ok':continue
        for name,info in r['files'].items():assert sha256((path.parent/name).read_bytes()).hexdigest()==info['sha256']
        with Image.open(path.parent/'model_input.png') as image:size=image.size
        cw=detection_inputs(r['detection'])[0,820:835];cp=projected_angles(r['detection']['image_landmarks'],size)
        captures.append({'id':r['id'],'intended':r['expected_letter'],'prediction':str(bundle['estimator'].predict([[cw[wi],cp[pi]]])[0]),
            'middle_PIP_3d':float(cw[wi]),'middle_PIP_2d':float(cp[pi])})
    atomic_json(OUT/'capture_predictions.json',captures)
    (OUT/'report.md').write_text('# Exploratory middle-finger-only G/H diagnostic\n\n'
        'This additional ablation was proposed after inspecting the failed three-representation experiment. It is post-hoc, not an independent confirmation. '
        'Only the middle-PIP angle from the 3D estimate and 2D pixel projection is used; no webcam examples enter fitting. '
        'The two-class model always outputs G or H and cannot recognize other letters.\n\n'
        f"Same frozen five-fold G/H accuracy: {100*scores['accuracy']:.3f}%; macro-F1 {scores['macro_f1']:.5f}.\n\n"
        '| Capture | Intended | Prediction | 3D PIP | 2D PIP |\n|---|---|---|---:|---:|\n'+
        '\n'.join(f"| {r['id']} | {r['intended']} | {r['prediction']} | {r['middle_PIP_3d']:.1f} | {r['middle_PIP_2d']:.1f} |" for r in captures)+
        '\n\nReproduce with `python -m research.studies.gh_middle_diagnostic`. A fresh, balanced webcam round is needed before considering a second-stage recognizer.\n')
    print(json.dumps({'cv':scores,'captures':captures}),flush=True)


if __name__=='__main__':main()
