"""Summarize paired Kaggle model, grouped permutation and ablation results."""
import csv
import json
from pathlib import Path


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def merge_parts(folder,stem,parts):
    rows=[]
    for part in parts:rows+=read(folder/f'{stem}_{part}.csv')
    with (folder/f'{stem}.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    out=Path('experiments/kaggle_lightgbm_importance_01')
    metrics=read(out/'model_metrics.csv');letters=read(out/'per_letter.csv')
    permutations=merge_parts(out,'permutation_r5_cap100',('svm','lightgbm'))
    merge_parts(out,'permutation_letters_r5_cap100',('svm','lightgbm'))
    importance=merge_parts(out,'permutation_summary_r5_cap100',('svm','lightgbm'))
    ablation=merge_parts(out,'ablation',('svm','lightgbm'))
    merge_parts(out,'ablation_letters',('svm','lightgbm'))
    pooled={(r['model'],r['variant']):r for r in metrics if r['fold']=='pooled'}
    assert len(pooled)==4
    assert len(permutations)==2*5*9*5
    assert len(ablation)==2*3*5
    lines=['# SVM, LightGBM en de bijdrage van extra features','',
           'Dezelfde 60.251 bruikbare Kaggle ASL Alphabet-beelden en vijf gegroepeerde buitenfolds '
           'als de eerdere experimenten. Shin gebruikt 820 waarden; gecombineerd gebruikt Shin plus '
           '70 categorieën, 26 Excel-overeenkomstscores en 15 gewrichtshoeken. Extra features zijn hier ongewogen. '
           'De 17.749 ontbrekende detecties vallen buiten de classifierscores hieronder.', '',
           '## Modelvergelijking','',
           '| Model | Shin accuracy | Met extra features | Verschil | Shin macro-F1 | Met extra macro-F1 |',
           '|---|---:|---:|---:|---:|---:|']
    for model in ('svm','lightgbm'):
        base=pooled[model,'shin'];extra=pooled[model,'combined']
        lines.append(f"| {model} | {float(base['accuracy']):.3%} | {float(extra['accuracy']):.3%} | "
                     f"{100*(float(extra['accuracy'])-float(base['accuracy'])):+.3f} pp | "
                     f"{float(base['macro_f1']):.5f} | {float(extra['macro_f1']):.5f} |")
    lines+=['','SVM: RBF, C=10, gamma=scale, balanced. LightGBM: 100 boostingrondes, 31 bladeren, '
            'learning rate 0,1, minimaal 20 voorbeelden per blad, seed 42; overige projectinstellingen vast. '
            'LightGBM gebruikt vier rekenthreads bij de resterende fits; deterministic en force_col_wise staan aan. '
            'Er is geen nieuwe hyperparameterzoekactie gedaan. Imputatie en schaling van hoeken worden uitsluitend op training gefit.', '',
            '## Permutation importance','',
            'Vijf herhalingen per groep en fold, maximaal 100 validatiebeelden per letter per fold. '
            'Onderstaande waarden zijn gemiddelde macro-F1-dalingen in procentpunten. Positief betekent slechter na husselen; '
            'negatief betekent beter na husselen. Kolommen binnen een groep worden gezamenlijk verwisseld; '
            'dezelfde permutaties en steekproeven worden voor beide modellen gebruikt. Spreiding staat in de CSV.', '',
            '| Gehusselde groep | SVM: macro-F1-daling | LightGBM: macro-F1-daling |',
            '|---|---:|---:|']
    groups=['all_extra','categories','angles','excel_scores','thumb','finger_flexion','spread','crossing','contact']
    for group in groups:
        values=[]
        for model in ('svm','lightgbm'):
            row=next(r for r in importance if r['model']==model and r['group']==group and r['metric']=='macro_f1_drop')
            values.append(f"{100*float(row['mean_drop']):+.3f} pp")
        lines.append(f"| {group} | "+' | '.join(values)+' |')
    lines+=['','`all_extra` husselt alle extra features samen. De andere groepen overlappen: hun importances zijn niet optelbaar. '
            '`thumb` bevat duimcategorieën en duimhoeken; `finger_flexion` bevat overige buigcategorieën en hoeken. '
            'Excel-scores zijn afgeleid van categorieën. Redundantie kan importance drukken; '
            'onmogelijke combinaties door husselen kunnen haar vergroten. Importance meet modelgebruik, geen zelfstandig bewijs van extra informatie.', '',
            '## Ablatie: opnieuw trainen zonder featuregroep','',
            'Alle validatiebeelden worden gebruikt. Hieronder staat de samengevoegde accuracy na verwijderen. '
            'De delta is ten opzichte van het volledige gecombineerde model: positief betekent verbetering na weglaten.', '',
            '| Weggelaten groep | SVM accuracy | SVM delta | LightGBM accuracy | LightGBM delta |',
            '|---|---:|---:|---:|---:|']
    ablation_summary=[]
    for group in ('angles','categories','excel_scores'):
        values=[]
        for model in ('svm','lightgbm'):
            rows=[r for r in ablation if r['model']==model and r['group_removed']==group]
            count=sum(int(r['images']) for r in rows)
            assert count==60251
            accuracy=sum(float(r['accuracy'])*int(r['images']) for r in rows)/count
            difference=accuracy-float(pooled[model,'combined']['accuracy'])
            ablation_summary.append({'model':model,'removed':group,'accuracy':accuracy,'delta':difference})
            values += [f'{accuracy:.3%}',f'{100*difference:+.3f} pp']
        lines.append(f"| {group} | "+' | '.join(values)+' |')
    lines+=['','De drie groepen worden afzonderlijk weggelaten; alle overige input blijft aanwezig. '
            'Een groep kan dus weinig ablatiewinst opleveren omdat een andere groep dezelfde informatie levert. '
            'Macro-F1 per fold en recallveranderingen per letter staan in de detailbestanden.', '',
            '## Per letter: recall','',
            '| Letter | Beelden | SVM Shin | SVM extra | LightGBM Shin | LightGBM extra | LightGBM winst door extra |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        rows={(r['model'],r['variant']):r for r in letters if r['letter']==label}
        keys=[('svm','shin'),('svm','combined'),('lightgbm','shin'),('lightgbm','combined')]
        values=' | '.join(f"{float(rows[key]['recall']):.2%}" for key in keys)
        delta=float(rows['lightgbm','combined']['recall'])-float(rows['lightgbm','shin']['recall'])
        lines.append(f"| {label} | {rows['svm','shin']['images']} | {values} | {100*delta:+.2f} pp |")
    lines+=['','## Interpretatie en grenzen','',
            'Combineer drie soorten bewijs: winst tegenover Shin, daling bij husselen, en daling na opnieuw trainen zonder de groep. '
            'Alleen hoge permutation importance is onvoldoende om te zeggen dat jouw features werkelijk iets toevoegen boven Shin. '
            'De evaluatie gebruikt zes filenameblokken, niet geverifieerde personen; het is ontwikkelonderzoek. '
            'De referenties en drempels zijn eerder op de aparte Roboflow-set gekozen. '
            'Er is geen onafhankelijke webcam- of persoonstest. De standaardafwijkingen over folds en herhalingen zijn geen onafhankelijke betrouwbaarheidsintervallen.', '',
            'Ruwe resultaten: `model_metrics.csv`, `per_letter.csv`, `permutation_summary_r5_cap100.csv`, '
            '`permutation_letters_r5_cap100.csv`, `ablation.csv`, `ablation_letters.csv`, `oof_predictions.csv`. '
            'De modelspecifieke bestanden blijven behouden. Lokaal herhalen: `docs/KAGGLE_LIGHTGBM_IMPORTANCE.md`.','']
    ablation_letters=read(out/'ablation_letters.csv')
    candidate=[]
    for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        selected=[r for r in ablation_letters if r['model']=='svm'
                  and r['group_removed']=='categories' and r['letter']==label]
        count=sum(int(r['images']) for r in selected)
        correct=sum(round(float(r['ablated_recall'])*int(r['images'])) for r in selected)
        baseline=next(r for r in letters if r['model']=='svm' and r['variant']=='shin' and r['letter']==label)
        before=round(float(baseline['recall'])*count)
        candidate.append({'letter':label,'images':count,'shin_correct':before,
                          'candidate_correct':correct,'shin_recall':float(baseline['recall']),
                          'candidate_recall':correct/count,'net_correct_gain':correct-before})
    with (out/'candidate_without_categories_per_letter.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(candidate[0]));writer.writeheader();writer.writerows(candidate)
    lines+=['## Vervolgtkandidaat: SVM zonder losse categorieën','',
            'Shin + 15 continue hoeken + 26 Excel-scores (861 invoerwaarden) haalt '
            '58.892/60.251 correct: **97,744%**. Dat zijn 65 extra juiste beelden tegenover '
            'Shin/project (+0,108 procentpunt), en 70 tegenover de volledige ongewogen combinatie. '
            'De 70 expliciete one-hotcategorieën worden weggelaten; categorische informatie '
            'blijft indirect aanwezig in de Excel-scores.', '',
            '| Letter | Shin recall | Kandidaat recall | Extra correcte beelden |',
            '|---|---:|---:|---:|']
    for label in ('M','R','S','Z','O','N','U'):
        row=next(r for r in candidate if r['letter']==label)
        lines.append(f"| {label} | {float(row['shin_recall']):.2%} | "
                     f"{float(row['candidate_recall']):.2%} | {int(row['net_correct_gain']):+d} |")
    lines+=['','Deze kandidaat is achteraf tussen de ablatievarianten gekozen en moet '
            'op nieuwe onafhankelijke beelden bevestigd worden. Dit is een accuracyresultaat; '
            'de opgeslagen ablaties bevatten macro-F1 per fold, geen gepoolde macro-F1. '
            'Alle letters staan in `candidate_without_categories_per_letter.csv`.','']
    (out/'report.md').write_text('\n'.join(lines))
    (out/'analysis_summary.json').write_text(json.dumps({'model_metrics':[r for r in metrics if r['fold']=='pooled'],
        'permutation':importance,'ablation':ablation_summary,'rows':60251,'folds':5,'repeats':5,'permutation_max_per_class':100},indent=2))
    print(out/'report.md')


if __name__=='__main__':main()
