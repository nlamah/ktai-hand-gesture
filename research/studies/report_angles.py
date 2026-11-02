"""Report the isolated angle/Excel experiment, including paired per-letter effects."""
import csv
import json
import argparse
from pathlib import Path

from research.inputs import write_csv
from research.studies.compare_angles import VARIANTS

NAMES={'shin':'Shin alleen','shin_angles':'Shin + hoeken','shin_excel':'Shin + Excel-scores','shin_angles_excel':'Shin + hoeken + Excel-scores'}


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
    lines=['# Shin: geïsoleerde vergelijking van gewrichtshoeken en Excel-scores','',
           'SVM en LightGBM zijn vergeleken op dezelfde 60.251 bruikbare beelden van Kaggle ASL Alphabet, '
           'met exact dezelfde vijf gegroepeerde folds als de voorgaande experimenten. '
           'De eerder opgeslagen Shin-baselinevoorspellingen zijn hergebruikt en op inputhashes gecontroleerd. '
           'Alle drie toevoegingsvarianten zijn opnieuw per fold getraind; de 70 losse one-hotcategorieën zijn afwezig.', '',
           '## Totaalscores','',
           '| Model | Invoer | Aantal features | Accuracy | Macro-F1 | Verschil accuracy t.o.v. Shin |',
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
    lines+=['','## Wat voegen alleen de hoeken toe?','',
            f"Bij SVM: {svm['shin']['correct']} → {svm['shin_angles']['correct']} juiste beelden "
            f"({svm['shin_angles']['correct']-svm['shin']['correct']:+d}). "
            f"Accuracy {svm['shin']['accuracy']:.3%} → {svm['shin_angles']['accuracy']:.3%}; "
            f"macro-F1 {svm['shin']['macro_f1']:.5f} → {svm['shin_angles']['macro_f1']:.5f}.", '',
            f"Bij LightGBM: {lgbm['shin']['correct']} → {lgbm['shin_angles']['correct']} juiste beelden "
            f"({lgbm['shin_angles']['correct']-lgbm['shin']['correct']:+d}). "
            f"Accuracy {lgbm['shin']['accuracy']:.3%} → {lgbm['shin_angles']['accuracy']:.3%}.", '',
            '## Per fold: verschil accuracy van hoeken t.o.v. Shin','',
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
                '| Letter | Beelden | Shin | + hoeken | + Excel | + beide | Hoeken − Shin |',
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
    lines+=['','## Protocol en interpretatie','',
            '- De 15 continue metingen zijn 14 binnenhoeken van de gewrichten plus de getekende duim-MCP-afwijking, berekend uit wereldlandmarks. '
            'Imputatie en standaardisatie worden alleen op de trainingsfold geleerd; Shin blijft ongewijzigd.',
            '- Excel-scores zijn 26 overeenkomstfracties: één per mogelijke letter. Ze worden voor elk beeld berekend zonder het echte letterlabel als invoer. '
            'Zij behouden indirect categorische informatie; de variant met alleen hoeken gebruikt geen categoriedrempels of Excel-referenties.',
            '- SVM: RBF, C=10, gamma=scale, balanced. LightGBM: vaste projectinstellingen (100 rondes, 31 bladeren, learning rate 0,1). '
            'Geen nieuwe hyperparameter- of gewichtstuning.',
            '- De scores hierboven gelden voor de 60.251 bruikbare detecties. De overige 17.749 van de 78.000 letterbeelden '
            'kunnen niet door een landmarkclassifier worden verwerkt. All-images-scores staan in summary.json.',
            '- Filenameblokken zijn geen bekende personen. Dit blijft herhaald ontwikkelonderzoek op dezelfde folds; '
            'kleine verschillen zijn geen bewijs van algemene of onafhankelijke winst. Een nieuwe persoon-/webcamtest blijft nodig.', '',
            '## Lokaal herhalen','',
            '```bash','conda activate ktai-hand-gesture',
            'python -m research.studies.compare_angles --models svm lightgbm',
            'python -m scripts.report_kaggle_angles_scores','```','',
            'Modellen en validatievoorspellingen worden per fold opgeslagen, zodat hervatten bestaande checkpoints gebruikt. '
            'De vorige resultaten worden niet overschreven. Details: metrics.csv, per_letter.csv, angle_effects_per_letter.csv en de modelspecifieke out-of-foldvoorspellingen.','']
    write_csv(out/'metrics.csv',metrics);write_csv(out/'per_letter.csv',letters);write_csv(out/'angle_effects_per_letter.csv',effects)
    (out/'summary.json').write_text(json.dumps({'metrics':summary,'images':60251,'all_images':78000,'folds':5,
                    'protocol':'Frozen same folds, fixed hyperparameters, no one-hot categories; baseline OOF reused.',
                    'angle_effects_per_letter':effects},indent=2))
    (out/'report.md').write_text('\n'.join(lines));print(out/'report.md')


if __name__=='__main__':main()
