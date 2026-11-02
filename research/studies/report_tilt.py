"""Audit and summarize the fixed tilt experiment without fitting another model."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from research.inputs import LABELS,write_csv

OUT=Path('results/tilt_augmentation_01')


def main():
    cv=json.loads((OUT/'cv_summary.json').read_text());captures=json.loads((OUT/'capture_summary.json').read_text())
    config=json.loads((OUT/'protocol.json').read_text())
    with (OUT/'cv_predictions.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==60251 and len({r['id'] for r in rows})==60251
    truth=np.array([r['label'] for r in rows]);detail=[]
    for name in ('baseline','tilt15'):
        pred=np.array([r[name] for r in rows]);s=cv[name]
        assert abs(s['accuracy']-accuracy_score(truth,pred))<1e-12
        assert abs(s['macro_f1']-f1_score(truth,pred,average='macro'))<1e-12
        assert s['correct']==int((truth==pred).sum())
        pr,re,ff,n=precision_recall_fscore_support(truth,pred,labels=LABELS,zero_division=0)
        detail.extend({'model':name,'letter':l,'images':int(n[i]),'correct':int(((truth==l)&(pred==l)).sum()),'precision':float(pr[i]),'recall':float(re[i]),'f1':float(ff[i])} for i,l in enumerate(LABELS))
    write_csv(OUT/'cv_per_letter.csv',detail)
    with (OUT/'capture_predictions.csv').open(encoding='utf-8-sig') as handle:cap=list(csv.DictReader(handle))
    for role,result in captures.items():
        if result.get('status')=='incomplete':continue
        for name,s in result.items():
            rr=[r for r in cap if r['role']==role and r['model']==name];assert len(rr)==27
            y=[r['letter'] for r in rr];p=[r['prediction'] for r in rr]
            assert abs(s['accuracy']-accuracy_score(y,p))<1e-12
            assert abs(s['macro_f1']-f1_score(y,p,labels=list('CDO'),average='macro'))<1e-12
    before=cv['baseline'];after=cv['tilt15'];a=captures['development']['baseline'];b=captures['development']['tilt15']
    evaluation=captures['evaluation']
    if evaluation.get('status')=='incomplete':
        evaluation_note=f"The evaluation directory contains {evaluation['completed']}/27 finalized records; no evaluation scores were computed."
    else:
        eb,ec=evaluation['baseline'],evaluation['tilt15']
        evaluation_note=f"Reserved evaluation: baseline {eb['correct']}/{eb['images']} accuracy {100*eb['accuracy']:.2f}% macro-F1 {eb['macro_f1']:.6f}; candidate {ec['correct']}/{ec['images']} accuracy {100*ec['accuracy']:.2f}% macro-F1 {ec['macro_f1']:.6f}."
    lines=['# Fixed ±15° direction augmentation experiment','',
        'One predefined candidate: original inputs plus -15°/+15° rotations about the camera-x axis. '
        'Only 630 axis-direction features change; 190 normalized distances and 15 joint-angle values stay fixed. '
        'All 26 Kaggle classes are augmented. Each of the three rows has sample weight 1/3, preserving total fitting weight per original. '
        'The fixed SVM profile is C=10, RBF, gamma=scale, balanced. No webcam captures enter training.', '',
        'This is rotation of normalized landmark-feature coordinates, not a photorealistic wrist, perspective or detector simulation. '
        'No other angle/axis/model variant was searched. Predefined settings and source hashes are in `protocol.json`.', '',
        '## Matched Kaggle five-fold comparison', '',
        '| Model | Accuracy | Macro-F1 | Correct / accepted images |','|---|---:|---:|---:|',
        f"| Baseline Shin + joint angles | {100*before['accuracy']:.5f}% | {before['macro_f1']:.6f} | {before['correct']}/60251 |",
        f"| ±15° augmentation | {100*after['accuracy']:.5f}% | {after['macro_f1']:.6f} | {after['correct']}/60251 |", '',
        f"Net correct change: {after['correct']-before['correct']:+d}; {cv['fixed']} formerly incorrect images recovered, {cv['lost']} formerly correct images lost. "
        f"Accuracy delta: {100*(after['accuracy']-before['accuracy']):+.5f} percentage points. Four folds decline and one improves.", '',
        'The same frozen filename-group partitions and validation IDs are used. Augmentation is applied after the split to fitting images only; '
        'imputation/scaling are fitted on the fitting partition. There are 60,251 OOF predictions per variant, not 180,753 validation observations. '
        'Groups are not verified signers; these remain exploratory detection-conditioned scores, not a paper replication or live accuracy estimate.', '',
        '## Development webcam round: 27 C/D/O images', '',
        '| Model | Correct | Accuracy | Macro-F1 over C/D/O |','|---|---:|---:|---:|',
        f"| Baseline | {a['correct']}/27 | {100*a['accuracy']:.2f}% | {a['macro_f1']:.6f} |",
        f"| ±15° augmentation | {b['correct']}/27 | {100*b['accuracy']:.2f}% | {b['macro_f1']:.6f} |", '',
        '| True letter | Baseline correct | Augmented correct |','|---|---:|---:|',
    ]
    for l in 'CDO':
        aa=sum(r['letter']==l and r['prediction']==l for r in cap if r['role']=='development' and r['model']=='baseline')
        bb=sum(r['letter']==l and r['prediction']==l for r in cap if r['role']=='development' and r['model']=='tilt15')
        lines.append(f'| {l} | {aa}/9 | {bb}/9 |')
    lines+=['','One additional O is recognized; C remains unchanged. One wrong C prediction changes from D to O, '
        'so O precision decreases even though O recall improves. This is a small development-set change, not evidence of general improvement.','',
        '## Other inspected examples and evaluation status','',
        'On the ten previously inspected, manually selected confusions, nine predictions stay unchanged; '
        'the friend’s O changes from D to O. This is descriptive evidence from a selected known example, not an independent test.', '',
        evaluation_note + ' The model and recipe were fixed before checking the evaluation status.', '',
        '## Try live', '', 'Close the existing camera window first. From the repository root:', '', '```bash',
        'conda activate ktai-hand-gesture','python -m research.live_camera --model models/asl_shin_angles_tilt15.joblib','```','',
        'Baseline comparison:', '', '```bash','python -m research.live_camera --model models/asl_shin_angles.joblib','```','',
        'Q/Esc closes the native camera window. Snapshots retain the selected model hash. The default model is still the original baseline. '
        'This candidate is available for testing; results do not justify replacing the baseline as a generally better model.', '',
        '## Reproduce', '', '```bash',
        'python -m research.studies.tilt_augmentation --stage train',
        'python -m research.studies.tilt_augmentation --stage cv',
        'python -m research.studies.tilt_augmentation --stage evaluate',
        'python -m research.studies.report_tilt','```','',
        'The train command refuses to overwrite the candidate. CV checkpoints can be resumed under the same frozen experiment configuration. '
        'To run a separate experiment, provide fresh `--out` and `--candidate` paths. Input and model hashes must match. '
        'After a complete reserved evaluation round is collected, rerunning `--stage evaluate` computes its paired result without retraining.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    verification={'cv_oof_rows_per_model':60251,'all_cv_metrics_recomputed':True,'all_development_metrics_recomputed':True,
                  'cv_augmented_only_in_fitting_partitions':True,'angles_deg':config['angles_deg'],'candidate_variants':1,
                  'baseline_sha256':sha256(Path('models/asl_shin_angles.joblib').read_bytes()).hexdigest(),
                  'candidate_sha256':sha256(Path('models/asl_shin_angles_tilt15.joblib').read_bytes()).hexdigest()}
    assert verification['baseline_sha256']==config['baseline_sha256']
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(OUT/'report.md')


if __name__=='__main__':main()
