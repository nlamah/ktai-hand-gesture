"""Report the isolated angle/Excel experiment, including paired per-letter effects."""
import csv
import json
import argparse
from pathlib import Path

from research.inputs import write_csv
from research.studies.compare_angles import VARIANTS

NAMES={'shin':'Shin only','shin_angles':'Shin + angles','shin_excel':'Shin + Excel-scores','shin_angles_excel':'Shin + angles + Excel-scores'}


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/angles'))
    out=parser.parse_args().out
    metrics=[];letters=[];summary={};effects=[]
    for model in ('svm','lightgbm'):
        metrics+=read(out/f'metrics_{model}.csv');letters+=read(out/f'per_letter_{model}.csv')
    pooled={(r['model'],r['variant']):r for r in metrics if r['fold']=='pooled'}
    assert len(pooled)==8
    lines=['# Shin: isolated comparison of joint angles and Excel scores','',
           'SVM and LightGBM were compared on the same 60,251 accepted Kaggle ASL Alphabet images, '
           'using exactly the same five grouped folds as previous experiments. '
           'Previously saved Shin baseline predictions were reused after checking input hashes. '
           'All three addition variants were retrained per fold; the 70 standalone one-hot categories are absent.', '',
           '## Overall scores','',
           '| Model | Input | Feature count | Accuracy | Macro-F1 | Accuracy difference versus Shin |',
           '|---|---|---:|---:|---:|---:|']
    for model in ('svm','lightgbm'):
        base=float(pooled[model,'shin']['accuracy']);summary[model]={}
        for variant in VARIANTS:
            row=pooled[model,variant];rr=[r for r in letters if r['model']==model and r['variant']==variant]
            correct=sum(int(r['correct']) for r in rr);assert sum(int(r['images']) for r in rr)==60251
            entry={'accuracy':float(row['accuracy']),'macro_f1':float(row['macro_f1']),'correct':correct,
                   'accuracy_all_78000':correct/78000,'features':int(row['features']),
                   'accuracy_delta_from_shin':float(row['accuracy'])-base}
            assert abs(correct/60251-entry['accuracy'])<1e-12
            summary[model][variant]=entry
            lines.append(f"| {model} | {NAMES[variant]} | {row['features']} | {entry['accuracy']:.3%} | "
                         f"{entry['macro_f1']:.5f} | {100*entry['accuracy_delta_from_shin']:+.3f} pp |")
    svm=summary['svm'];lgbm=summary['lightgbm']
    lines+=['','## What do the angles alone add?','',
            f"For SVM: {svm['shin']['correct']} → {svm['shin_angles']['correct']} correct images "
            f"({svm['shin_angles']['correct']-svm['shin']['correct']:+d}). "
            f"Accuracy {svm['shin']['accuracy']:.3%} → {svm['shin_angles']['accuracy']:.3%}; "
            f"macro-F1 {svm['shin']['macro_f1']:.5f} → {svm['shin_angles']['macro_f1']:.5f}.", '',
            f"For LightGBM: {lgbm['shin']['correct']} → {lgbm['shin_angles']['correct']} correct images "
            f"({lgbm['shin_angles']['correct']-lgbm['shin']['correct']:+d}). "
            f"Accuracy {lgbm['shin']['accuracy']:.3%} → {lgbm['shin_angles']['accuracy']:.3%}.", '',
            '## Per-fold accuracy difference: angles versus Shin','',
            '| Fold | SVM | LightGBM |','|---|---:|---:|']
    for fold in map(str,range(1,6)):
        values=[]
        for model in ('svm','lightgbm'):
            base=next(r for r in metrics if r['model']==model and r['variant']=='shin' and r['fold']==fold)
            angles=next(r for r in metrics if r['model']==model and r['variant']=='shin_angles' and r['fold']==fold)
            values.append(f"{100*(float(angles['accuracy'])-float(base['accuracy'])):+.3f} pp")
        lines.append(f"| {fold} | "+' | '.join(values)+' |')
    for model in ('svm','lightgbm'):
        lines+=['',f'## Recall per letter: {model}','',
                '| Letter | Images | Shin | + angles | + Excel | + both | Angles − Shin |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            rr={r['variant']:r for r in letters if r['model']==model and r['letter']==label}
            delta=float(rr['shin_angles']['recall'])-float(rr['shin']['recall'])
            gain=int(rr['shin_angles']['correct'])-int(rr['shin']['correct'])
            effects.append({'model':model,'letter':label,'images':int(rr['shin']['images']),
                            'baseline_recall':float(rr['shin']['recall']),'angles_recall':float(rr['shin_angles']['recall']),
                            'angles_delta':delta,'extra_correct_angles':gain})
            lines.append(f"| {label} | {rr['shin']['images']} | "+
                         ' | '.join(f"{float(rr[v]['recall']):.2%}" for v in VARIANTS)+f" | {100*delta:+.2f} pp |")
    lines+=['','## Protocol and interpretation','',
            '- The 15 continuous measurements are 14 interior joint angles plus signed thumb MCP deviation, calculated from world landmarks. '
            'Imputation and standardization are learned on the training fold only; Shin remains unchanged.',
            '- Excel scores are 26 compatibility fractions, one per candidate letter, calculated without the true letter label as input. '
            'They indirectly retain categorical information; the angle-only addition uses no category thresholds or Excel references.',
            '- SVM: RBF, C=10, gamma=scale, balanced. LightGBM: fixed project settings (100 rounds, 31 leaves, learning rate 0.1). '
            'No new hyperparameter or weight tuning.',
            '- These scores apply to 60,251 accepted detections. The remaining 17,749 of 78,000 letter images '
            'cannot be processed by a landmark classifier. All-image scores are in summary.json.',
            '- Filename blocks are not known people. This remains repeated development research on the same folds; '
            'small differences do not establish general or independent gains. A new-person/webcam test is still needed.', '',
            '## Repeat locally','',
            '```bash','conda activate ktai-hand-gesture',
            'python -m research.studies.compare_angles --models svm lightgbm',
            'python -m research.studies.report_angles','```','',
            'Models and validation predictions are saved per fold so resumed runs use existing checkpoints. '
            'Previous results are not overwritten. Details: metrics.csv, per_letter.csv, angle_effects_per_letter.csv and model-specific out-of-fold predictions.','']
    write_csv(out/'metrics.csv',metrics);write_csv(out/'per_letter.csv',letters);write_csv(out/'angle_effects_per_letter.csv',effects)
    (out/'summary.json').write_text(json.dumps({'metrics':summary,'images':60251,'all_images':78000,'folds':5,
                    'protocol':'Frozen same folds, fixed hyperparameters, no one-hot categories; baseline OOF reused.',
                    'angle_effects_per_letter':effects},indent=2))
    (out/'report.md').write_text('\n'.join(lines));print(out/'report.md')


if __name__=='__main__':main()
