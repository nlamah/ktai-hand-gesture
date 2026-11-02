"""Verify and compare baseline, all-letter tilt, and C-only tilt results."""
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from research.inputs import LABELS,write_csv
from research.live_model import load_bundle

OUT=Path('results/tilt_C_only_01')


def main():
    cv=json.loads((OUT/'cv_summary.json').read_text());cap=json.loads((OUT/'capture_summary.json').read_text())
    config=json.loads((OUT/'protocol.json').read_text());assert config['augmented_letters']==['C'] and not config['all_26_training_classes_augmented']
    with (OUT/'cv_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==60251 and len({r['id'] for r in rows})==60251
    y=np.array([r['label'] for r in rows]);details=[]
    for name in ('baseline','tilt15'):
        pred=np.array([r[name] for r in rows])
        assert abs(accuracy_score(y,pred)-cv[name]['accuracy'])<1e-12
        assert abs(f1_score(y,pred,average='macro')-cv[name]['macro_f1'])<1e-12
        precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
        details.extend({'model':name,'letter':l,'images':int(support[i]),'correct':int(((y==l)&(pred==l)).sum()),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,l in enumerate(LABELS))
    write_csv(OUT/'cv_per_letter.csv',details)
    with (OUT/'capture_predictions.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
    predictions={name:{r['id']:r for r in rr if r['role']=='development' and r['model']==name} for name in ('baseline','tilt15')}
    for name,pp in predictions.items():
        assert len(pp)==27
        yy=[r['letter'] for r in pp.values()];pr=[r['prediction'] for r in pp.values()]
        assert abs(accuracy_score(yy,pr)-cap['development'][name]['accuracy'])<1e-12
        assert abs(f1_score(yy,pr,labels=list('CDO'),average='macro')-cap['development'][name]['macro_f1'])<1e-12
    changed=[r for r in rows if r['baseline']!=r['tilt15']]
    changed_caps=[i for i,r in predictions['baseline'].items() if r['prediction']!=predictions['tilt15'][i]['prediction']]
    baseline=load_bundle('models/asl_shin_angles.joblib');candidate=load_bundle('models/asl_shin_angles_C_tilt15.joblib')
    assert candidate['training_rows']==64075 and candidate['training_images']==60251
    assert candidate['augmentation']==config
    np.testing.assert_equal(baseline['estimator'][-1]._gamma,candidate['estimator'][-1]._gamma)
    np.testing.assert_array_equal(baseline['estimator'][-1].class_weight_,candidate['estimator'][-1].class_weight_)
    meta=json.loads(Path('data/asl_kaggle/research/inputs_metadata.json').read_text());original_y=np.array([r['label'] for r in meta['samples']])
    assert int((original_y=='C').sum())==1912
    all_cv=json.loads(Path('results/tilt_augmentation_01/cv_summary.json').read_text())
    all_cap=json.loads(Path('results/tilt_augmentation_01/capture_summary.json').read_text())
    variants=[('Baseline',cv['baseline'],cap['development']['baseline']),('All letters +/-15 degrees',all_cv['tilt15'],all_cap['development']['tilt15']),('Only C +/-15 degrees',cv['tilt15'],cap['development']['tilt15'])]
    lines=['# Selective C-only tilt augmentation','',
        'Only the 1,912 accepted Kaggle C images receive -15/+15-degree camera-x variants of their 630 direction features. '
        'The other 58,339 images occur once unchanged. Total fitting rows: 64,075 from 60,251 original images. '
        'The 190 distance values and 15 joint angles are unchanged. Captures are not used for training.', '',
        '## Comparison','',
        '| Model | Kaggle 5-fold accuracy | Kaggle macro-F1 | Development correct | Development macro-F1 C/D/O |',
        '|---|---:|---:|---:|---:|']
    for name,cc,dd in variants:lines.append(f"| {name} | {100*cc['accuracy']:.5f}% | {cc['macro_f1']:.8f} | {dd['correct']}/27 | {dd['macro_f1']:.6f} |")
    lines+=['','## Interpretation','',
        'The C-only variant has exactly the same 27 development predictions as the baseline: C 3/9, D 8/9 and O 6/9. '
        'No development improvement is observed. The previously inspected friend\'s O remains D with this variant; the all-letter variant recognized it as O.', '',
        f"Kaggle: {cv['fixed']} incorrect examples recovered and {cv['lost']} correct examples lost. "
        f"{len(changed)} predictions change from one wrong label to another. Accuracy is identical; the macro-F1 difference "
        f"({cv['tilt15']['macro_f1']-cv['baseline']['macro_f1']:+.8f}) is practically negligible and reflects redistribution of false positives, not additional correct images.", '',
        'This is a negative result for this limited C-only +/-15-degree camera-x recipe. It does not show that other axes/ranges, '
        'physical image augmentation or different representations cannot help. No other recipes were searched in this run.', '',
        '## Controls and limitations','',
        'Each C source has three rows weighted 1/3 each; other rows have weight1. Selective duplication would otherwise affect class frequency. '
        'For this run, imputation/scaling, balanced class weights and the resolved gamma=scale value are fitted on original fitting rows. '
        'The SVM is then fitted on the augmented rows. Full-fit class weights and gamma match the baseline exactly. '
        'Each CV fold applies this procedure only to its fitting partition; validation rows remain original.', '',
        'The earlier all-letter run used ordinary gamma=scale on its augmented matrix. Therefore, the C-only run is controlled directly '
        'against the baseline; the three-way table is not a strict one-factor ablation of the earlier all-letter run\'s numeric gamma. '
        'This preprocessing policy is recorded in protocol.json.', '',
        'The angle transformation is a rotation in normalized landmark-feature space, not a complete physical wrist/perspective/detector simulation. '
        'Kaggle groups are filename blocks, not verified people. Development results use only 27 examples of one signer and three letters. '
        f"The evaluation session remains incomplete ({cap['evaluation'].get('completed',0)}/27), so no independent webcam score is available.", '',
        '## Live test','', '```bash','conda activate ktai-hand-gesture',
        'python -m research.live_camera --model models/asl_shin_angles_C_tilt15.joblib','```','',
        'Close the previous camera window first. The default model is unchanged. This candidate is for comparison, not a demonstrated improvement.', '',
        '## Reproduce','', '```bash',
        'python -m research.studies.tilt_augmentation --stage train --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib',
        'python -m research.studies.tilt_augmentation --stage cv --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib',
        'python -m research.studies.tilt_augmentation --stage evaluate --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib',
        'python -m research.studies.report_C_tilt','```','',
        'The existing candidate is protected against overwrite; use fresh --candidate and --out paths for another fit. '
        'Matched OOF predictions, capture predictions and all-letter precision/recall/F1 are retained beside this report.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    verification={'oof_images_per_variant':60251,'development_images_per_variant':27,'metric_reconciliation_passed':True,
                  'C_training_images_augmented':1912,'other_images_not_augmented':58339,'fitting_rows':64075,
                  'resolved_gamma_matches_baseline_exactly':True,'class_weights_match_baseline_exactly':True,
                  'changed_cv_predictions':len(changed),'changed_development_predictions':len(changed_caps),
                  'candidate_sha256':sha256(Path('models/asl_shin_angles_C_tilt15.joblib').read_bytes()).hexdigest(),
                  'baseline_sha256':sha256(Path('models/asl_shin_angles.joblib').read_bytes()).hexdigest()}
    assert verification['baseline_sha256']==config['baseline_sha256']
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(OUT/'report.md')


if __name__=='__main__':main()
