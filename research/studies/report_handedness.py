"""Dutch paper extracts and a readable report from verified canonicalization scores."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from research.data import atomic_json
from research.studies.handedness_normalization import OUT

NAMES={'original_training':'Zonder aanpassing','mirror_training':'Spiegeltraining','right_normalized':'Rechtsnormalisatie'}
VIEWS={'original':'Origineel','mirrored_rgb':'Gespiegeld','balanced_mix':'Vast mengsel'}


def number(value, digits=5):return f'{value:.{digits}f}'.replace('.',',')
def integer(value):return f'{int(value):,}'.replace(',','.')
def change(n):return f'{integer(abs(n))} '+('extra' if n>=0 else 'minder')


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['status']=='passed'
    assert verification['predictions_sha256']==sha256((OUT/'predictions.csv').read_bytes()).hexdigest()
    all_metrics=pd.read_csv(OUT/'metrics.csv',keep_default_na=False)
    metrics=all_metrics[all_metrics.fold=='pooled']
    paired=pd.read_csv(OUT/'paired_changes.csv',keep_default_na=False)
    summary=json.loads((OUT/'summary.json').read_text())
    def score(model,view,arm,scope='accepted_only'):
        row=metrics[(metrics.model==model)&(metrics.presentation==view)&(metrics.arm==arm)&(metrics.scope==scope)]
        assert len(row)==1
        return row.iloc[0]
    a=score('svm','original','original_training')
    m=score('svm','original','mirror_training')
    b=score('svm','original','right_normalized')
    mix_a=score('svm','balanced_mix','original_training')
    mix_m=score('svm','balanced_mix','mirror_training')
    mix_b=score('svm','balanced_mix','right_normalized')
    text=("Een andere aanpak spiegelt links gedetecteerde handen vóór oefenen en toetsen naar dezelfde stand. "
          "We vergelijken alleen oorspronkelijke foto's met gevonden handpunten. Voor het SVM-model met extra gewrichtsmetingen "
          f"daalt de precisie van {number(100*a.macro_precision,2)}\\% naar {number(100*b.macro_precision,2)}\\% "
          f"en F1 van {number(a.macro_f1,4)} naar {number(b.macro_f1,4)}: "
          f"{change(b.correct-a.correct)} goede voorspellingen. "
          "Op het aparte mengsel van oorspronkelijke en gespiegelde beelden helpt normaliseren wel, maar spiegeltraining blijft beter (bijlage F).\n")
    Path('paper/handedness_main_nl.tex').write_text(text)
    rows=[]
    for model in ('svm','lightgbm'):
        for view in ('original','mirrored_rgb','balanced_mix'):
            if model=='lightgbm' and view!='original':continue
            for arm in NAMES:
                s=score(model,view,arm)
                rows.append([model.upper() if model=='svm' else 'LightGBM',VIEWS[view],NAMES[arm],integer(s.images),
                    number(100*s.macro_precision,3),number(s.macro_f1,5),integer(s.correct)])
    Path('paper/handedness_rows_nl.tex').write_text('% Gegenereerd uit results/handedness_01/metrics.csv.\n' +
        '\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n')
    statements=[]
    for model in ('svm','lightgbm'):
        for view in ('original','balanced_mix'):
            for reference in ('original_training','mirror_training'):
                r=paired[(paired.model==model)&(paired.presentation==view)&(paired.reference==reference)&
                    (paired.scope=='accepted_only')&(paired.fold=='pooled')].iloc[0]
                statements.append(f"{model}, {VIEWS[view].lower()}, tegenover {NAMES[reference].lower()}: "
                    f"{integer(r.fixed)} hersteld, {integer(r.lost)} verloren, netto {int(r.net):+d}.")
    pair=paired[(paired.model=='svm')&(paired.presentation=='original')&
        (paired.reference=='original_training')&(paired.scope=='accepted_only')&(paired.fold=='pooled')].iloc[0]
    folds=paired[(paired.model=='svm')&(paired.presentation=='original')&
        (paired.reference=='original_training')&(paired.scope=='accepted_only')&(paired.fold!='pooled')]
    fold_text=', '.join(f'{int(n):+d}' for n in folds.net)
    result_text=(f"Op de oorspronkelijke beelden herstelt SVM {integer(pair.fixed)} fouten en verliest het {integer(pair.lost)} goede voorspellingen "
        f"ten opzichte van niets aanpassen; netto {int(pair.net):+d}. De foldverschillen zijn {fold_text}. "
        f"Tegenover spiegeltraining is het netto verschil {int(b.correct-m.correct):+d}. "
        "Normalisatie verbetert de oorspronkelijke classificatie dus niet.\n\n"
        f"Het aparte mengsel heeft {integer(mix_b.images)} geldige detecties. Daar verandert SVM-macro-F1 van "
        f"{number(mix_a.macro_f1,5)} naar {number(mix_b.macro_f1,5)}; spiegeltraining geeft {number(mix_m.macro_f1,5)}. "
        "Dit is een andere toetsverdeling, niet de oorspronkelijke baseline. "
        "Het uitsluiten van detectiemislukkingen verandert hier de precisie niet: die pogingen leveren geen voorspelde letter en dus geen false positive. "
        "F1 verandert wel doordat hun false negatives wegvallen. Alle-pogingenscores blijven apart in de resultaatbestanden staan.\n")
    Path('paper/handedness_results_nl.tex').write_text(result_text)
    counts=(f"Van de {integer(summary['training_originals'])} geaccepteerde originelen heeft de detector "
        f"{integer(summary['original_Left'])} als Left en {integer(summary['original_Right'])} als Right gelabeld. "
        f"Normalisatie voegt geen trainingsrijen toe: elke bron behoudt gewicht \\'e\\'en. "
        f"Bij {integer(summary['both_detected'])} bronnen zijn beide RGB-presentaties geldig; bij "
        f"{integer(summary['same_handedness_on_both_views'])} daarvan wisselt het handlabel niet.\n")
    Path('paper/handedness_counts_nl.tex').write_text(counts)
    md=['# Rechtsnormalisatie vóór de featureberekening','',
        'Verkennende vergelijking op dezelfde bronbeelden, vaste folds en apart gedetecteerde RGB-presentaties als de spiegelproef.',
        'Detector-Left wordt naar detector-Right gespiegeld. Het is geen geverifieerd label van de menselijke hand of voorkeurshand.',
        'Primaire vergelijking: oorspronkelijke beelden met geldige handdetectie, tegen de oorspronkelijke 835-inputreferentie. Mislukte detecties zijn uitgesloten. Het aparte synthetische mengsel is een andere toetsverdeling.',
        '', '## Alleen geaccepteerde detecties (primair)', '',
        '| Model | Toets | Training | N geldig | Macro-precisie (%) | Macro-F1 | Correct |',
        '|---|---|---|---:|---:|---:|---:|']
    for model in ('svm','lightgbm'):
        for view in ('original','mirrored_rgb','balanced_mix'):
            for arm in NAMES:
                r=score(model,view,arm)
                md.append(f'| {model} | {VIEWS[view]} | {NAMES[arm]} | {r.images} | {100*r.macro_precision:.5f} | {r.macro_f1:.8f} | {r.correct} |')
    md+=['','## Alle pogingen (secundair, inclusief detectiemislukkingen)','',
        '| Model | Toets | Training | N aangeboden | Macro-precisie (%) | Macro-F1 | Correct |',
        '|---|---|---|---:|---:|---:|---:|']
    for model in ('svm','lightgbm'):
        for view in ('original','mirrored_rgb','balanced_mix'):
            for arm in NAMES:
                r=score(model,view,arm,'all_attempts')
                md.append(f'| {model} | {VIEWS[view]} | {NAMES[arm]} | {r.images} | {100*r.macro_precision:.5f} | {r.macro_f1:.8f} | {r.correct} |')
    md+=['','## Paarsgewijze veranderingen','',*['- '+s for s in statements],
        '', '## Afbakening','',
        'De primaire scores sluiten mislukte detecties uit, met dezelfde geaccepteerde toetsbeelden voor iedere aanpak binnen een presentatie. De secundaire scores over alle 78.000 pogingen tellen detectiemislukkingen als false negative. Precisie blijft gelijk doordat geen letter is voorspeld; recall en F1 veranderen. De aanpassing verbetert de handdetectie zelf niet.',
        'Training gebruikt één rij per geldige oorspronkelijke bron. Spiegeltraining gebruikt extra RGB-afgeleiden; rechtsnormalisatie niet.',
        'MediaPipe-handschattingen worden gebruikt zonder grondwaarheidscontrole; foutieve handlabels kunnen de verkeerde normalisatie veroorzaken.',
        'Dezelfde filename-folds zijn eerder gebruikt tijdens ontwikkeling. Spiegelbeelden van dezelfde bron zijn geen onafhankelijke deelnemers. Geen claim over ongeziene linkshandige personen.',
        'De camera past de geometrie vóór de features alleen aan als het geselecteerde model de normalisatie in zijn bundle vermeldt. Eerdere modellen blijven ongewijzigd.',
        '', 'Details: `protocol.json`, `metrics.csv`, `per_letter.csv`, `handedness_metrics.csv`, `paired_changes.csv`, `predictions.csv`, `geometry_verification.json`, `verification.json`.']
    (OUT/'report.md').write_text('\n'.join(md)+'\n')
    atomic_json(OUT/'paper_extracts.json',{'status':'passed','primary_scope':'accepted_only','primary_presentation':'original',
        'reporter_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{str(OUT/name):sha256((OUT/name).read_bytes()).hexdigest()
        for name in ('metrics.csv','paired_changes.csv','summary.json','verification.json')},
        'generated_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in
            map(Path,('paper/handedness_main_nl.tex','paper/handedness_rows_nl.tex','paper/handedness_results_nl.tex','paper/handedness_counts_nl.tex'))}})
    print(text,flush=True)


if __name__=='__main__':main()
