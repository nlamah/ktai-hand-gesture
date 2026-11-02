"""Reconcile G/H rotation validation and replay existing, selected webcam errors."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from research.inputs import LABELS, write_csv
from research.live_model import load_bundle, detection_inputs

OUT=Path('results/rotation_GH_01')
CANDIDATE=Path('models/asl_shin_angles_GH_rotation.joblib')
BASELINE=Path('models/asl_shin_angles.joblib')


def main():
    config=json.loads((OUT/'protocol.json').read_text())
    assert config['augmented_letters']==['G','H']
    assert config['recipe']=='camera-y direction rotation' and config['angles_deg']==[0,-90,90,180]
    cv=json.loads((OUT/'cv_summary.json').read_text())
    with (OUT/'cv_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==60251 and len({r['id'] for r in rows})==60251
    truth=np.array([r['label'] for r in rows]);details=[]
    for name in ('baseline','tilt15'):
        pred=np.array([r[name] for r in rows])
        assert abs(accuracy_score(truth,pred)-cv[name]['accuracy'])<1e-12
        assert abs(f1_score(truth,pred,average='macro')-cv[name]['macro_f1'])<1e-12
        precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=LABELS,zero_division=0)
        details.extend({'model':name,'letter':l,'images':int(support[i]),'correct':int(((truth==l)&(pred==l)).sum()),
                        'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,l in enumerate(LABELS))
    write_csv(OUT/'cv_per_letter.csv',details)
    bundles={'baseline':load_bundle(BASELINE),'GH_rotation':load_bundle(CANDIDATE)}
    original=bundles['baseline']['estimator'];candidate=bundles['GH_rotation']['estimator']
    assert bundles['GH_rotation']['augmentation']==config
    np.testing.assert_equal(original[-1]._gamma,candidate[-1]._gamma)
    np.testing.assert_array_equal(original[-1].class_weight_,candidate[-1].class_weight_)
    captures=[]
    for path in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(path.read_text())
        if r['detection']['status']!='ok':continue
        for name,info in r['files'].items():
            assert sha256((path.parent/name).read_bytes()).hexdigest()==info['sha256']
        x=detection_inputs(r['detection'])[0].reshape(1,-1)
        captures.append({'id':r['id'],'expected_letter':r['expected_letter'],
                         **{name:str(b['estimator'].predict(x)[0]) for name,b in bundles.items()}})
    (OUT/'exploratory_captures.json').write_text(json.dumps(captures,indent=2)+'\n')
    lines=['# G/H direction-rotation augmentation','',
        'One fixed experiment augments only the accepted G/H training examples with camera-y rotations of -90, +90 and 180 degrees, alongside the original. '
        'The 630 direction features rotate; 190 distances and 15 joint-angle inputs stay unchanged. '
        'This is a landmark-feature approximation, not image augmentation or a simulation of hand-back landmark detection.', '',
        f"There are {bundles['GH_rotation']['training_images']:,} original training images and {bundles['GH_rotation']['training_rows']:,} fitting rows. "
        'Each selected source contributes four rows weighted 0.25; other classes occur once with weight 1. '
        'Scaling, resolved RBF gamma and balanced class weights come from original fitting rows. Captures never enter training.', '',
        '## Frozen five-fold validation','',
        '| Model | Accuracy | Macro-F1 | Correct |','|---|---:|---:|---:|']
    for name,label in [('baseline','With joint angles'),('tilt15','G/H rotation')]:
        m=cv[name];lines.append(f"| {label} | {100*m['accuracy']:.5f}% | {m['macro_f1']:.8f} | {m['correct']}/{m['images']} |")
    lines += ['',f"Paired changes: {cv['fixed']} wrong predictions corrected; {cv['lost']} previously correct predictions lost.", '',
              '| Letter | Baseline precision | G/H precision | Baseline recall | G/H recall | Baseline F1 | G/H F1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for letter in LABELS:
        a=next(r for r in details if r['model']=='baseline' and r['letter']==letter)
        b=next(r for r in details if r['model']=='tilt15' and r['letter']==letter)
        lines.append(f"| {letter} | {100*a['precision']:.2f}% | {100*b['precision']:.2f}% | {100*a['recall']:.2f}% | {100*b['recall']:.2f}% | {a['f1']:.4f} | {b['f1']:.4f} |")
    lines += ['', '## Previously selected webcam errors', '',
              'These are known, selected failures, useful for diagnosis but not an independent or representative evaluation. '
              'The standard guided round contains only C/D/O, not G/H. A new G/H round with different poses and preferably another signer is still needed.', '',
              '| Intended letter | Joint-angle model | G/H rotation |','|---|---|---|']
    for r in captures:lines.append(f"| {r['expected_letter']} | {r['baseline']} | {r['GH_rotation']} |")
    command='--letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib'
    lines += ['', '## Try live', '', '```bash','conda activate ktai-hand-gesture',
              'python -m research.live_camera --model models/asl_shin_angles_GH_rotation.joblib','```','',
              'The initial model caption is G/H rotation. Select it again with button/key 5; All-letter tilt (3) is the separate earlier all-letter model.', '',
              '## Reproduce', '', '```bash']
    lines += [f'python -m research.studies.tilt_augmentation --stage {stage} {command}' for stage in ['train','cv','evaluate']]
    lines += ['python -m research.studies.report_GH_rotation','```','',
              'Training refuses to overwrite an existing model. Use fresh candidate/output paths for a fresh run. '
              'Historical CSV/JSON candidate fields named tilt15 are retained for compatibility; protocol.json specifies the actual larger G/H rotations. '
              'Validation uses original images and frozen filename-block groups, not verified signer groups. '
              'This exploratory result does not establish generalization to unseen people or palm/back views.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    verification={'metric_reconciliation_passed':True,'oof_images_per_variant':60251,
                  'resolved_gamma_matches_baseline':True,'balanced_class_weights_match_baseline':True,
                  'candidate_sha256':sha256(CANDIDATE.read_bytes()).hexdigest(),
                  'baseline_sha256':sha256(BASELINE.read_bytes()).hexdigest()}
    assert verification['baseline_sha256']==config['baseline_sha256']
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(OUT/'report.md')


if __name__=='__main__':main()
