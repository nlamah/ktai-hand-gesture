"""English paper extracts and a readable report from verified canonicalization scores."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from research.data import atomic_json
from research.studies.handedness_normalization import OUT

NAMES={'original_training':'No change','mirror_training':'Mirror training','right_normalized':'Right normalization'}
VIEWS={'original':'Original','mirrored_rgb':'Mirrored','balanced_mix':'Fixed mixture'}


def number(value, digits=5):return f'{value:.{digits}f}'
def integer(value):return f'{int(value):,}'
def change(n):return f'{integer(abs(n))} '+('extra' if n>=0 else 'fewer')


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
    text=("Another approach reflects detector-Left hands to a common orientation before fitting and inference. "
          "On accepted originals, the 835-input SVM precision changes "
          f"from {number(100*a.macro_precision,2)}\\% to {number(100*b.macro_precision,2)}\\%, "
          f"and F1 from {number(a.macro_f1,4)} to {number(b.macro_f1,4)}: "
          f"{change(b.correct-a.correct)} correct predictions. "
          "Normalization helps on the separate original/mirrored mixture, but mirror training remains better (Appendix F).\n")
    Path('paper/handedness_main_nl.tex').write_text(text)
    rows=[]
    for model in ('svm','lightgbm'):
        for view in ('original','mirrored_rgb','balanced_mix'):
            if model=='lightgbm' and view!='original':continue
            for arm in NAMES:
                s=score(model,view,arm)
                rows.append([model.upper() if model=='svm' else 'LightGBM',VIEWS[view],NAMES[arm],integer(s.images),
                    number(100*s.macro_precision,3),number(s.macro_f1,5),integer(s.correct)])
    Path('paper/handedness_rows_nl.tex').write_text('% Generated from results/handedness_01/metrics.csv.\n' +
        '\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n')
    statements=[]
    for model in ('svm','lightgbm'):
        for view in ('original','balanced_mix'):
            for reference in ('original_training','mirror_training'):
                r=paired[(paired.model==model)&(paired.presentation==view)&(paired.reference==reference)&
                    (paired.scope=='accepted_only')&(paired.fold=='pooled')].iloc[0]
                statements.append(f"{model}, {VIEWS[view].lower()}, relative to {NAMES[reference].lower()}: "
                    f"{integer(r.fixed)} corrected, {integer(r.lost)} lost, net {int(r.net):+d}.")
    pair=paired[(paired.model=='svm')&(paired.presentation=='original')&
        (paired.reference=='original_training')&(paired.scope=='accepted_only')&(paired.fold=='pooled')].iloc[0]
    folds=paired[(paired.model=='svm')&(paired.presentation=='original')&
        (paired.reference=='original_training')&(paired.scope=='accepted_only')&(paired.fold!='pooled')]
    fold_text=', '.join(f'{int(n):+d}' for n in folds.net)
    result_text=(f"On originals, SVM corrects {integer(pair.fixed)} errors and loses {integer(pair.lost)} correct predictions "
        f"relative to no change: net {int(pair.net):+d}. Fold nets are {fold_text}. "
        f"Relative to mirror training, the net difference is {int(b.correct-m.correct):+d}. "
        "Normalization does not improve original-image classification.\n\n"
        f"The separate mixture has {integer(mix_b.images)} accepted detections. Its SVM macro-F1 changes from "
        f"{number(mix_a.macro_f1,5)} to {number(mix_b.macro_f1,5)}; mirror training gives {number(mix_m.macro_f1,5)}. "
        "This is a different test distribution. Excluding failed detections leaves precision unchanged here, "
        "because those attempts predict no letter and create no false positive. F1 changes as their false negatives are excluded. "
        "All-attempt scores remain available separately.\n")
    Path('paper/handedness_results_nl.tex').write_text(result_text)
    counts=(f"Of {integer(summary['training_originals'])} accepted originals, the detector labels "
        f"{integer(summary['original_Left'])} Left and {integer(summary['original_Right'])} Right. "
        "Normalization adds no fitting rows; each source keeps weight one. "
        f"Both RGB presentations are accepted for {integer(summary['both_detected'])} sources; "
        f"{integer(summary['same_handedness_on_both_views'])} retain the same hand label across views.\n")
    Path('paper/handedness_counts_nl.tex').write_text(counts)
    md=['# Right-hand normalization before feature extraction', '',
        'Exploratory comparison using the same sources, folds and independently detected RGB views as the mirror study. Detector-Left is reflected to detector-Right; these are not independently verified hand labels.',
        'Primary comparison: accepted original images versus the original 835-input reference. The artificial mixture is a separate distribution.',
        '', '| Model | Test view | Training/input | Accepted N | Macro-precision (%) | Macro-F1 | Correct |',
        '|---|---|---|---:|---:|---:|---:|']
    for model in ('svm','lightgbm'):
        for view in ('original','mirrored_rgb','balanced_mix'):
            for arm in NAMES:
                r=score(model,view,arm)
                md.append(f'| {model} | {VIEWS[view]} | {NAMES[arm]} | {r.images} | {100*r.macro_precision:.5f} | {r.macro_f1:.8f} | {r.correct} |')
    md += ['', '## All attempts, including failed detections', '',
        '| Model | Test view | Training/input | Attempt N | Macro-precision (%) | Macro-F1 | Correct |',
        '|---|---|---|---:|---:|---:|---:|']
    for model in ('svm','lightgbm'):
        for view in ('original','mirrored_rgb','balanced_mix'):
            for arm in NAMES:
                r=score(model,view,arm,'all_attempts')
                md.append(f'| {model} | {VIEWS[view]} | {NAMES[arm]} | {r.images} | {100*r.macro_precision:.5f} | {r.macro_f1:.8f} | {r.correct} |')
    md += ['', '## Paired changes', '', *['- '+text for text in statements], '', '## Scope', '',
        'Primary scores share accepted test images within each presentation. Secondary scores count all 78,000 attempts: failed detections add false negatives, leaving precision unchanged but changing recall/F1. Normalization does not improve detection itself.',
        'Normalization fits one row per accepted original source; mirror training adds RGB derivatives. Detector hand labels are unverified. Reused filename folds and mirror pairs are not independent new participants.',
        'Live normalization applies before feature extraction only when the selected model bundle requests it. Deployment fits are not independent tests.',
        '', 'Evidence: `protocol.json`, `metrics.csv`, `per_letter.csv`, `handedness_metrics.csv`, `paired_changes.csv`, `predictions.csv`, `geometry_verification.json`, `verification.json`.']
    (OUT/'report.md').write_text('\n'.join(md)+'\n')
    atomic_json(OUT/'paper_extracts.json',{'status':'passed','primary_scope':'accepted_only','primary_presentation':'original',
        'reporter_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{str(OUT/name):sha256((OUT/name).read_bytes()).hexdigest()
        for name in ('metrics.csv','paired_changes.csv','summary.json','verification.json')},
        'generated_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in
            map(Path,('paper/handedness_main_nl.tex','paper/handedness_rows_nl.tex','paper/handedness_results_nl.tex','paper/handedness_counts_nl.tex'))}})
    print(text,flush=True)


if __name__=='__main__':main()
