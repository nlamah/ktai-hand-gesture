"""Verified eight-arm tables, contrasts and grouped-importance figures for the paper."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from research.data import atomic_json
from research.feature_study import VARIANTS,source_columns
from research.inputs import LABELS,write_csv
from research.studies.feature_factorial import OUT,WORK,references,digest

PAPER=Path('paper');FIG=PAPER/'figures'
NAMES={'DS':'Distances + directions','D':'Distances only',
       'DSE':'DS + own categories','DE':'D + own categories',
       'DSJ':'DS + world joints','DJ':'D + world joints',
       'DSEJ':'DS + categories + joints','DEJ':'D + categories + joints'}


def fmt(value,digits=5):return f'{float(value):.{digits}f}'
def integer(value):return f'{int(value):,}'
def delta(value):return f'{integer(abs(value))} '+('more' if value>=0 else 'fewer')
def latex(name,rows):
    (PAPER/name).write_text('% From verified results; report_feature_factorial.py.\n'+
        '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n')


def feature_name(key):
    if key=='thumb_mcp_shape':return 'Thumb MCP shape'
    if key=='flexion_thumb_ip':return 'Thumb IP flexion'
    if key.startswith('flexion_'):
        _,finger,joint=key.split('_')
        return {'index':'Index','middle':'Middle','ring':'Ring','pinky':'Little'}[finger]+'-'+joint.upper()+' flexion'
    return {'spread_index_middle':'Index–middle spreading','spread_middle_ring':'Middle–ring spreading',
            'crossing_index_middle':'Index–middle crossing','thumb_tip_index_proximity':'Thumb–index proximity',
            'thumb_tip_middle_proximity':'Thumb–middle proximity'}[key]


def prose_feature(key):
    return feature_name(key)


def heatmap(ax,values,rows,columns,title,limit):
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#edf0f2')
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,
                 norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    ax.set_xticks(range(len(columns)),columns,fontsize=10.5)
    ax.set_yticks(range(len(rows)),rows,fontsize=10.5)
    ax.set_title(title,fontsize=12,weight='bold',pad=12)
    for (i,j),value in np.ndenumerate(values):
        label='--' if np.isnan(value) else f'{0. if abs(value)<.005 else value:.2f}'
        ax.text(j,i,label,ha='center',va='center',fontsize=10,
                color='white' if np.isfinite(value) and abs(value)>.6*limit else '#172b3a')
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(length=0)
    return im


def figures(per,reference):
    def value(model,variant,group):
        rows=per[(per.model==model)&(per.variant==variant)&(per.group==group)]
        return np.nan if rows.empty else 100*rows.iloc[0].macro_f1_mean_drop
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    all_values=[]
    for model in ['svm','lightgbm']:
        all_values.append(np.array([[value(model,v,'block_'+b) for b in 'DSEJ'] for v in VARIANTS]))
    limit=max(1.,float(np.nanmax(np.abs(all_values))))
    for ax,model,values in zip(axes,['svm','lightgbm'],all_values):
        im=heatmap(ax,values,list(VARIANTS),['D','S','E','J'],model.upper(),limit)
    fig.colorbar(im,ax=axes,shrink=.8,label='Mean macro-F1 loss (percentage points)')
    fig.savefig(FIG/'factorial_block_importance.pdf',bbox_inches='tight');plt.close(fig)
    configurations=[('svm','DSE'),('svm','DE'),('lightgbm','DSE'),('lightgbm','DE')]
    values=np.array([[value(m,v,'check_'+k) for m,v in configurations] for k in reference])
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    im=heatmap(ax,values,[feature_name(k) for k in reference],['SVM\nDSE','SVM\nDE','LightGBM\nDSE','LightGBM\nDE'],
               'The 19 own checks, without extra world-joint inputs',max(.01,float(np.max(np.abs(values)))))
    fig.colorbar(im,ax=ax,shrink=.75,label='Macro-F1 loss (percentage points)')
    fig.savefig(FIG/'factorial_category_importance.pdf',bbox_inches='tight');plt.close(fig)
    family_keys=['family_E_'+k for k in ['thumb','finger_flexion','spread','crossing','contact']]+[
        'family_J_'+k for k in ['thumb','index','middle','ring','pinky']]
    labels=['E: thumb shape / flexion','E: other finger flexion','E: spreading','E: crossing','E: contact',
            'J: thumb','J: index','J: middle','J: ring','J: little']
    configurations=[('svm','DSEJ'),('svm','DEJ'),('lightgbm','DSEJ'),('lightgbm','DEJ')]
    values=np.array([[value(m,v,k) for m,v in configurations] for k in family_keys])
    fig,ax=plt.subplots(figsize=(9,3.7),layout='constrained')
    im=heatmap(ax,values,labels,['SVM\nDSEJ','SVM\nDEJ','LightGBM\nDSEJ','LightGBM\nDEJ'],
               'Families with both categories and world-joint inputs',max(.01,float(np.max(np.abs(values)))))
    fig.colorbar(im,ax=ax,shrink=.75,label='Macro-F1 loss (percentage points)')
    fig.savefig(FIG/'factorial_family_importance.pdf',bbox_inches='tight');plt.close(fig)


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    for path,expected in verification['results_sha256'].items():assert digest(OUT/path)==expected,path
    m=pd.read_csv(OUT/'metrics.csv',keep_default_na=False);pooled=m[m.fold=='pooled']
    paired=pd.read_csv(OUT/'paired_changes.csv',keep_default_na=False)
    per=pd.read_csv(OUT/'permutation_summary.csv',keep_default_na=False)
    predictions=pd.read_csv(OUT/'predictions.csv',keep_default_na=False)
    def score(model,variant):return pooled[(pooled.model==model)&(pooled.variant==variant)].iloc[0]
    overall=[];main_rows=[];changes=[]
    for variant in VARIANTS:
        a,b=score('svm',variant),score('lightgbm',variant)
        overall.append([variant,len(source_columns(variant)),fmt(100*a.macro_precision,3),fmt(a.macro_f1,6),
                        fmt(100*b.macro_precision,3),fmt(b.macro_f1,6)])
        main_rows.append([NAMES[variant],fmt(100*a.macro_precision,2),fmt(a.macro_f1,4),
                          fmt(100*b.macro_precision,2),fmt(b.macro_f1,4)])
        row=[variant]
        for model in ('svm','lightgbm'):
            p=paired[(paired.model==model)&(paired.variant==variant)&(paired.fold=='pooled')].iloc[0]
            row.extend((integer(p.fixed),integer(p.lost),f'{int(p.net):+d}'))
        changes.append(row)
    latex('factorial_summary_rows_nl.tex',overall);latex('factorial_main_rows_nl.tex',main_rows)
    latex('factorial_paired_rows_nl.tex',changes)
    # Marginal effects retain all four contexts rather than hiding interactions in one average.
    effects=[]
    for name in ['svm','lightgbm']:
        for factor,pairs in {'E':[('D','DE'),('DS','DSE'),('DJ','DEJ'),('DSJ','DSEJ')],
                             'J':[('D','DJ'),('DS','DSJ'),('DE','DEJ'),('DSE','DSEJ')],
                             'S':[('D','DS'),('DE','DSE'),('DJ','DSJ'),('DEJ','DSEJ')]}.items():
            for before,after in pairs:
                a,b=score(name,before),score(name,after)
                effects.append({'model':name,'factor_added':factor,'before':before,'after':after,
                    'macro_precision_change':b.macro_precision-a.macro_precision,'macro_f1_change':b.macro_f1-a.macro_f1,
                    'correct_change':int(b.correct-a.correct)})
    write_csv(OUT/'factor_effects.csv',effects)
    controls=[];control_rows=[]
    old=pd.read_csv('results/direction_joint_01/oof_predictions.csv',keep_default_na=False)
    for variant,label in [('distance_joint','D + image joints'),('shin_joint','DS + image joints')]:
        values=[]
        for name in ['svm','lightgbm']:
            part=old[(old.model==name)&(old.variant==variant)]
            current=predictions[(predictions.model==name)&(predictions.variant=='DS')].set_index('id').loc[part.id]
            np.testing.assert_array_equal(part.label,current.label)
            np.testing.assert_array_equal(part.fold,current.fold)
            p,r,f,n=precision_recall_fscore_support(part.label,part.prediction,labels=LABELS,zero_division=0)
            controls.append({'model':name,'variant':variant,'macro_precision':p.mean(),'macro_f1':f.mean(),
                             'correct':int((part.label==part.prediction).sum())})
            values.extend((fmt(100*p.mean(),3),fmt(f.mean(),6)))
        control_rows.append([label,*values])
    latex('factorial_controls_nl.tex',control_rows);write_csv(OUT/'same_coordinate_controls.csv',controls)
    base=score('svm','DS');own=score('svm','DSE');joint=score('svm','DSJ');full=score('svm','DSEJ')
    distance=score('svm','D');without=score('svm','DEJ')
    lbase=score('lightgbm','DS');lown=score('lightgbm','DSE')
    main_text=(f"Relative to Shin DS, own categories give {delta(own.correct-base.correct)} correct SVM predictions. "
        f"World-joint measurements give {delta(joint.correct-base.correct)}; both additions together give {delta(full.correct-base.correct)}. "
        f"For LightGBM, adding categories changes F1 from {fmt(lbase.macro_f1,4)} to {fmt(lown.macro_f1,4)}. "
        f"Without directions, categories and world joints give {delta(without.correct-distance.correct)} correct predictions than distances alone. "
        "Appendix B gives detailed scores, paired changes and the same-coordinate control.\n")
    (PAPER/'factorial_main_results_nl.tex').write_text(main_text)
    top_names=[]
    for name in ['svm','lightgbm']:
        part=per[(per.model==name)&(per.variant=='DSE')&per.group.str.startswith('check_')].sort_values('macro_f1_mean_drop',ascending=False)
        r=part.iloc[0];top_names.append((prose_feature(r.group.removeprefix('check_')),100*r.macro_f1_mean_drop))
    perm_text=(f"For SVM with DS and own categories, shuffling {top_names[0][0].lower()} gives the largest mean F1 loss "
        f"({fmt(top_names[0][1],2)} percentage points). For LightGBM, it is {top_names[1][0].lower()} "
        f"({fmt(top_names[1][1],2)} percentage points). These are dependencies of the fitted models. "
        "A small loss does not establish that a feature is useless: other features may supply the same information. See Appendix B.\n")
    (PAPER/'factorial_main_permutation_nl.tex').write_text(perm_text)
    figures(per,references()['A'])
    lines=['# Eight feature representations and grouped permutation', '',
        'D: 190 distances; S: 630 Shin directions; E: 70 one-hot values for 19 own checks; J: 15 world-joint measurements.',
        'Eight representations, two learners and five shared filename-block folds on 60,251 accepted original sources. Workbook-match scores, augmentation and hand normalization are excluded.',
        '', '| Variant | Inputs | SVM precision (%) | SVM F1 | LightGBM precision (%) | LightGBM F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        a,b=score('svm',variant),score('lightgbm',variant)
        lines.append(f'| {variant}: {NAMES[variant]} | {len(source_columns(variant))} | {100*a.macro_precision:.5f} | {a.macro_f1:.8f} | {100*b.macro_precision:.5f} | {b.macro_f1:.8f} |')
    lines += ['', '## Adding/removing information', '',
        '`factor_effects.csv` retains the effect of adding S/E/J in all four contexts; `paired_changes.csv` counts corrections/losses relative to DS.',
        '', '## Permutation after fitting', '',
        'Classification uses all 60,251 sources. Permutation uses 100 held-out sources per letter/fold: 13,000 shared IDs, five fixed shuffles per group/fold. Labels and models remain fixed. All columns in a group move together.',
        'D/S/E/J blocks are tested where present. Individual E checks are tested in DSE/DE; five E families and five J families are tested in DSEJ/DEJ. CSVs retain precision/F1 losses. Figure values are percentage points. Fold/repeat spread is descriptive, not an independent confidence interval; negative losses are retained.',
        '', '## Interpretation and limitations', '',
        'Permutation measures fitted-model dependence, not marginal gains from refitting. Related features may substitute, and shuffling may create unusual combinations. The eight training variants answer the complementary adding/removing question.',
        'D/S use image points; E/J use world points. Same-coordinate J controls are reported separately. Coordinate source, scaling and fixed profiles limit interpretation. No independent anatomical ground truth is implied.',
        'Folds were inspected during development and are not verified people. Scores condition on accepted detection. There is no untouched final test or demonstrated webcam gain.',
        '', '## Evidence', '',
        '`protocol.json`, `input_verification.json`, `metrics.csv`, `per_letter.csv`, `paired_changes.csv`, `predictions.csv`, `confusions.csv`, `factor_effects.csv`, `same_coordinate_controls.csv`, `permutation_selection.csv`, `permutation.csv`, `permutation_summary.csv`, `verification.json`. Fold models/checkpoints: `runs/feature_factorial_01/`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    generated=[*PAPER.glob('factorial_*_nl.tex'),*FIG.glob('factorial_*_importance.pdf')]
    atomic_json(OUT/'paper_extracts.json',{'status':'passed','source_sha256':verification['results_sha256'],
        'reporter_sha256':digest(Path(__file__)),'generated_sha256':{str(p):digest(p) for p in generated},
        'control_source_sha256':digest('results/direction_joint_01/oof_predictions.csv')})
    print(main_text+'\n'+perm_text,flush=True)


if __name__=='__main__':main()
