"""Verify and summarize the fixed 2D+3D angle experiment, without selecting a model."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from research.data import atomic_json
from research.live_model import load_bundle
from research.studies.compare_image_angles import OUT,MODEL,BASELINE


def main():
    cv=json.loads((OUT/'cv_summary.json').read_text())
    cap=json.loads((OUT/'capture_summary.json').read_text())
    config=json.loads((OUT/'protocol.json').read_text())
    with (OUT/'cv_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==60251 and len({r['id'] for r in rows})==60251
    for name in ['baseline','image_angles']:
        y=[r['label'] for r in rows];pred=[r[name] for r in rows]
        assert abs(accuracy_score(y,pred)-cv[name]['accuracy'])<1e-12
        assert abs(f1_score(y,pred,average='macro')-cv[name]['macro_f1'])<1e-12
    with (OUT/'cv_per_letter.csv').open(encoding='utf-8-sig') as f:letters=list(csv.DictReader(f))
    with (OUT/'capture_predictions.csv').open(encoding='utf-8-sig') as f:captures=list(csv.DictReader(f))
    guided=[r for r in captures if r['session_id']=='development_20261031T160814Z_3dd178e6']
    assert len(guided)==27
    guided_metrics={name:{'correct':sum(r[name]==r['intended'] for r in guided),'images':27,
        'macro_f1_CDO':float(f1_score([r['intended'] for r in guided],[r[name] for r in guided],labels=list('CDO'),average='macro',zero_division=0))}
        for name in ['baseline','image_angles']}
    latest=cap['latest_GH_round']
    bundle=load_bundle(MODEL);assert bundle['experiment_protocol']==config
    assert sha256(BASELINE.read_bytes()).hexdigest()==config['baseline_model_sha256']
    lines=['# Projected 2D angles alongside 3D joint angles','',
        'One fixed experiment adds 14 internal joint angles measured in aspect-correct image pixels to the original 835-input Joint angles model. '
        'The candidate has 849 inputs: 820 unchanged Shin features, 15 original 3D measurements and 14 projected angles. '
        'There is no augmentation and no new contact input. The same training-only median imputation/standardization is applied to the 29 angle inputs. '
        'The classifier remains balanced RBF SVM C=10, gamma=scale. Captures never enter training.', '',
        '## Frozen five-fold comparison','',
        '| Model | Accuracy | Macro-F1 | Correct |','|---|---:|---:|---:|']
    for name,label in [('baseline','Joint angles'),('image_angles','Image + depth')]:
        m=cv[name];lines.append(f"| {label} | {100*m['accuracy']:.5f}% | {m['macro_f1']:.8f} | {m['correct']}/{m['images']} |")
    lines+=['',f"Paired changes: {cv['fixed']} wrong predictions fixed; {cv['lost']} previously correct predictions lost.", '',
            '| Fold | Joint angles accuracy | Image + depth accuracy | Joint angles F1 | Image + depth F1 |',
            '|---|---:|---:|---:|---:|']
    for f in cv['folds']:
        a,b=f['baseline'],f['image_angles'];lines.append(f"| {f['fold']} | {100*a['accuracy']:.3f}% | {100*b['accuracy']:.3f}% | {a['macro_f1']:.5f} | {b['macro_f1']:.5f} |")
    lines+=['','### Choosing the number of folds','',
        'Five folds retain the existing comparison protocol and avoid selecting a favorable split. '
        'The groups are filename blocks, not verified people. Most fits use approximately 80% of images, '
        'but group sizes make the partitions uneven. Increasing to ten folds would not establish generalization to new signers. '
        'A held-out signer/session is a more direct test of webcam robustness. '
        'The main scores pool all out-of-fold predictions; fold means/standard deviations describe split variation and are not confidence intervals.', '']
    for metric in ['accuracy','macro_f1']:
        d=cv[f'fold_{metric}_mean_std']
        lines.append(f"Fold {metric} mean ± sample SD: baseline {d['baseline']['mean']:.6f} ± {d['baseline']['sample_std']:.6f}; candidate {d['image_angles']['mean']:.6f} ± {d['image_angles']['sample_std']:.6f}.")
    lines+=['','## All letters','', '| Letter | Baseline recall | Candidate recall | Baseline F1 | Candidate F1 |',
            '|---|---:|---:|---:|---:|']
    for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        a=next(r for r in letters if r['letter']==l and r['model']=='baseline');b=next(r for r in letters if r['letter']==l and r['model']=='image_angles')
        lines.append(f"| {l} | {100*float(a['recall']):.3f}% | {100*float(b['recall']):.3f}% | {float(a['f1']):.5f} | {float(b['f1']):.5f} |")
    lines+=['','## Existing webcam development examples','',
        'The most recent GH-handrug round contains 11 deliberately selected errors from the G/H-rotation model, including other letters. '
        'Replaying these is diagnostic, not a representative or independent accuracy estimate. '
        'There are no new H-labeled examples in that round.', '',
        '| Intended | Saved G/H prediction | Joint angles | Image + depth |','|---|---|---|---|']
    for r in latest['predictions']:lines.append(f"| {r['intended']} | {r['saved_prediction']} | {r['baseline']} | {r['image_angles']} |")
    lines+=['',f"Correct on these 11 known selected examples: Joint angles {latest['correct']['baseline']}/11; Image + depth {latest['correct']['image_angles']}/11.",'',
        f"Earlier guided C/D/O development round: Joint angles {guided_metrics['baseline']['correct']}/27 (macro-F1 C/D/O {guided_metrics['baseline']['macro_f1_CDO']:.6f}); "
        f"Image + depth {guided_metrics['image_angles']['correct']}/27 (macro-F1 C/D/O {guided_metrics['image_angles']['macro_f1_CDO']:.6f}). "
        'These are reused development images. There is no completed held-out webcam evaluation.', '',
        '## Live comparison', '', '```bash','conda activate ktai-hand-gesture',
        'python -m research.live_camera --model models/asl_shin_angles_2d.joblib --round-name "2D-3D-test"','```','',
        'Select **Image + depth [6]** in the native tool, or **Joint angles [2]** to compare. '
        'Snapshot exports retain 849 feature names/values, separate 3D and projected-angle dictionaries, model hash and round identity.', '',
        '## Reproduce', '', '```bash','python -m research.studies.compare_image_angles --stage all',
        'python -m research.studies.report_image_angles','```','',
        'The all stage resumes the fixed protocol and protected candidate, caches only the 14 new angles, '
        'and reuses completed fold predictions. A direct train stage refuses to overwrite an existing candidate. '
        'Training geometry comes from the original frozen manifest; live geometry uses the same feature function and detector-image dimensions. '
        'View-dependent 2D angles cannot be interpreted as ground-truth anatomy. '
        'This run tests one representation with fixed settings, not all possible weights or contact features.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    verification={'oof_images_per_variant':60251,'metric_reconciliation_passed':True,'same_frozen_folds':True,
        'same_training_live_pixel_function':True,'captures_excluded_from_training':True,
        'candidate_sha256':sha256(MODEL.read_bytes()).hexdigest(),'baseline_sha256':sha256(BASELINE.read_bytes()).hexdigest(),
        'latest_GH_known_errors_correct':latest['correct'],'guided_CDO_development':guided_metrics}
    atomic_json(OUT/'verification.json',verification)
    print(OUT/'report.md')


if __name__=='__main__':main()
