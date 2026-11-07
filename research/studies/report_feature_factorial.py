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
NAMES={'DS':'Basis: afstanden en richtingen','D':'Alleen afstanden',
       'DSE':'Basis + eigen categorieën','DE':'Afstanden + eigen categorieën',
       'DSJ':'Basis + wereldgewrichtshoeken','DJ':'Afstanden + wereldgewrichtshoeken',
       'DSEJ':'Basis + categorieën + wereldhoeken','DEJ':'Afstanden + categorieën + wereldhoeken'}


def fmt(value,digits=5):return f'{float(value):.{digits}f}'.replace('.',',')
def integer(value):return f'{int(value):,}'.replace(',','.')
def delta(value):return f'{integer(abs(value))} '+('meer' if value>=0 else 'minder')
def latex(name,rows):
    (PAPER/name).write_text('% Uit geverifieerde resultaten; report_feature_factorial.py.\n'+
        '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n')


def feature_name(key):
    if key=='thumb_mcp_shape':return 'Duimstand (MCP)'
    if key=='flexion_thumb_ip':return 'Duim-IP: buiging'
    if key.startswith('flexion_'):
        _,finger,joint=key.split('_')
        return {'index':'Wijs','middle':'Middel','ring':'Ring','pinky':'Pink'}[finger]+'-'+joint.upper()+': buiging'
    return {'spread_index_middle':'Spreiding wijs-middel','spread_middle_ring':'Spreiding middel-ring',
            'crossing_index_middle':'Kruising wijs-middel','thumb_tip_index_proximity':'Duim-wijs: contact',
            'thumb_tip_middle_proximity':'Duim-middel: contact'}[key]


def prose_feature(key):
    if key=='thumb_mcp_shape':return 'de duimstand'
    if key=='flexion_thumb_ip':return 'de buiging van de duim'
    if key.startswith('flexion_'):
        _,finger,joint=key.split('_')
        location={'mcp':'de knokkel','pip':'het middelste gewricht','dip':'het eindgewricht'}[joint]
        name={'index':'de wijsvinger','middle':'de middelvinger','ring':'de ringvinger','pinky':'de pink'}[finger]
        return f'de buiging bij {location} van {name}'
    return {'spread_index_middle':'de spreiding tussen wijs- en middelvinger',
            'spread_middle_ring':'de spreiding tussen middel- en ringvinger',
            'crossing_index_middle':'de kruising van wijs- en middelvinger',
            'thumb_tip_index_proximity':'het duimcontact met de wijsvinger',
            'thumb_tip_middle_proximity':'het duimcontact met de middelvinger'}[key]


def heatmap(ax,values,rows,columns,title,limit):
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#edf0f2')
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,
                 norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    ax.set_xticks(range(len(columns)),columns,fontsize=10.5)
    ax.set_yticks(range(len(rows)),rows,fontsize=10.5)
    ax.set_title(title,fontsize=12,weight='bold',pad=12)
    for (i,j),value in np.ndenumerate(values):
        label='--' if np.isnan(value) else f'{0. if abs(value)<.005 else value:.2f}'.replace('.',',')
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
    fig.colorbar(im,ax=axes,shrink=.8,label='Gemiddelde macro-F1-daling (procentpunt)')
    fig.savefig(FIG/'factorial_block_importance.pdf',bbox_inches='tight');plt.close(fig)
    configurations=[('svm','DSE'),('svm','DE'),('lightgbm','DSE'),('lightgbm','DE')]
    values=np.array([[value(m,v,'check_'+k) for m,v in configurations] for k in reference])
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    im=heatmap(ax,values,[feature_name(k) for k in reference],['SVM\nDSE','SVM\nDE','LightGBM\nDSE','LightGBM\nDE'],
               'De 19 eigen categorieën, zonder extra wereldhoeken',max(.01,float(np.max(np.abs(values)))))
    fig.colorbar(im,ax=ax,shrink=.75,label='Macro-F1-daling (procentpunt)')
    fig.savefig(FIG/'factorial_category_importance.pdf',bbox_inches='tight');plt.close(fig)
    family_keys=['family_E_'+k for k in ['thumb','finger_flexion','spread','crossing','contact']]+[
        'family_J_'+k for k in ['thumb','index','middle','ring','pinky']]
    labels=['E: duimstand / duimbuiging','E: overige vingerbuiging','E: spreiding','E: kruising','E: contact',
            'J: duim','J: wijsvinger','J: middelvinger','J: ringvinger','J: pink']
    configurations=[('svm','DSEJ'),('svm','DEJ'),('lightgbm','DSEJ'),('lightgbm','DEJ')]
    values=np.array([[value(m,v,k) for m,v in configurations] for k in family_keys])
    fig,ax=plt.subplots(figsize=(9,3.7),layout='constrained')
    im=heatmap(ax,values,labels,['SVM\nDSEJ','SVM\nDEJ','LightGBM\nDSEJ','LightGBM\nDEJ'],
               'Families wanneer categorieën en wereldhoeken beide aanwezig zijn',max(.01,float(np.max(np.abs(values)))))
    fig.colorbar(im,ax=ax,shrink=.75,label='Macro-F1-daling (procentpunt)')
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
    for variant,label in [('distance_joint','D + J uit beeldpunten'),('shin_joint','DS + J uit beeldpunten')]:
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
    main_text=(f"Met Shin als basis geven de eigen categorieën bij SVM {delta(own.correct-base.correct)} correcte voorspellingen. "
        f"Wereldgewrichtshoeken geven er {delta(joint.correct-base.correct)}; beide toevoegingen samen {delta(full.correct-base.correct)}. "
        f"Bij LightGBM verandert F1 door alleen de eigen categorieën van {fmt(lbase.macro_f1,4)} naar {fmt(lown.macro_f1,4)}. "
        f"Zonder de richtingen geven categorieën en wereldhoeken samen {delta(without.correct-distance.correct)} correcte voorspellingen dan alleen afstanden. "
        "De precieze scores, paarsgewijze verschillen en controle met hoeken uit dezelfde punten staan in bijlage B.\n")
    (PAPER/'factorial_main_results_nl.tex').write_text(main_text)
    top_names=[]
    for name in ['svm','lightgbm']:
        part=per[(per.model==name)&(per.variant=='DSE')&per.group.str.startswith('check_')].sort_values('macro_f1_mean_drop',ascending=False)
        r=part.iloc[0];top_names.append((prose_feature(r.group.removeprefix('check_')),100*r.macro_f1_mean_drop))
    perm_text=(f"Bij SVM met Shin en eigen categorieën geeft het husselen van {top_names[0][0].lower()} de grootste gemiddelde F1-daling "
        f"({fmt(top_names[0][1],2)} procentpunt). Bij LightGBM is dat {top_names[1][0].lower()} "
        f"({fmt(top_names[1][1],2)} procentpunt). Dit laat zien welke informatie deze modellen gebruiken. "
        "Een kleine daling bewijst niet dat een kenmerk nutteloos is: andere kenmerken kunnen dezelfde informatie leveren. De detailresultaten staan in bijlage B.\n")
    (PAPER/'factorial_main_permutation_nl.tex').write_text(perm_text)
    figures(per,references()['A'])
    lines=['# Acht varianten van handfeatures en gegroepeerde permutatie','',
        'D = 190 afstanden; S = 630 Shin-richtingen; E = 70 categorische waarden voor 19 eigen controles; J = 15 wereldgewrichtsmetingen.',
        'Alle acht varianten zijn opnieuw getraind met SVM en LightGBM op dezelfde 60.251 geaccepteerde oorspronkelijke beelden en vijf vaste filename-folds. Geen 26 werkboekovereenkomstscores, spiegelingen of handnormalisatie in deze hoofdvergelijking.',
        '', '| Variant | Invoer | SVM precisie (%) | SVM F1 | LightGBM precisie (%) | LightGBM F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        a,b=score('svm',variant),score('lightgbm',variant)
        lines.append(f'| {variant}: {NAMES[variant]} | {len(source_columns(variant))} | {100*a.macro_precision:.5f} | {a.macro_f1:.8f} | {100*b.macro_precision:.5f} | {b.macro_f1:.8f} |')
    lines+=['','## Bijdrage hangt af van de overige invoer','',
        '`factor_effects.csv` geeft de toevoeging van S, E of J in alle vier bijbehorende contexten. `paired_changes.csv` telt herstel en verlies tegenover DS. De oude 931-invoerproef bevatte ook werkboekscores en is daarom niet dezelfde variant als de nieuwe DSEJ (905 invoerwaarden).',
        '', '## Permutatie na het trainen','',
        'De volledige classificatiescores hierboven gebruiken alle 60.251 beelden. Voor permutatie zijn vooraf 100 toetsbeelden per letter per fold gekozen: 13.000 unieke beelden. Iedere variant en leermethode krijgt exact dezelfde ID’s. Vijf vaste husselingen per groep/fold; de labels blijven staan. Het model wordt niet opnieuw getraind. Alle kolommen van een groep gebruiken dezelfde rijpermutatie.',
        'D/S/E/J worden in elke variant getest voor zover aanwezig. De 19 afzonderlijke categorieën worden in DSE en DE onderzocht. In DSEJ en DEJ worden vijf categorische families en vijf gewrichtsfamilies onderzocht. Dit voorkomt dat alle detailtests uitsluitend onder dubbel aanwezige hoekinformatie plaatsvinden.',
        'De CSV bevat dalingen van macro-precisie en macro-F1. De figuren tonen gemiddelde macro-F1-daling in procentpunten. SD over folds/herhalingen is beschrijvend; het is geen onafhankelijk betrouwbaarheidsinterval. Negatieve dalingen worden niet weggefilterd.',
        '', '## Interpretatie en beperkingen','',
        'Permutatie meet afhankelijkheid van een getraind model onder verstoring; sterk samenhangende features kunnen elkaar vervangen. Het doorbreken van samenhang tussen groepen kan onnatuurlijke combinaties veroorzaken. De acht trainingsvergelijkingen beantwoorden daarom een andere, aanvullende vraag dan de permutatietests.',
        'D/S gebruiken beeldhandpunten; E/J komen uit wereldhandpunten. De bestaande controles met J uit dezelfde beeldpunten blijven apart gerapporteerd. De coördinatenbron, schaling en vaste modelprofielen begrenzen de conclusies. Geen nieuwe sensor of onafhankelijk bewezen anatomische informatie.',
        'De filename-folds zijn tijdens ontwikkeling herhaald gebruikt en zijn geen geverifieerde personen. Scores gelden uitsluitend bij geaccepteerde detectie. Geen onafhankelijke eindtest of aangetoonde webcamverbetering. De eerdere augmentatie- en livevarianten behouden hun eigen protocol.',
        '', '## Bestanden','',
        '`protocol.json`, `input_verification.json`, `metrics.csv`, `per_letter.csv`, `paired_changes.csv`, `predictions.csv`, `confusions.csv`, `factor_effects.csv`, `same_coordinate_controls.csv`, `permutation_selection.csv`, `permutation.csv`, `permutation_summary.csv`, `verification.json`. Foldmodellen en tussenresultaten staan onder `runs/feature_factorial_01/`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    generated=[*PAPER.glob('factorial_*_nl.tex'),*FIG.glob('factorial_*_importance.pdf')]
    atomic_json(OUT/'paper_extracts.json',{'status':'passed','source_sha256':verification['results_sha256'],
        'reporter_sha256':digest(Path(__file__)),'generated_sha256':{str(p):digest(p) for p in generated},
        'control_source_sha256':digest('results/direction_joint_01/oof_predictions.csv')})
    print(main_text+'\n'+perm_text,flush=True)


if __name__=='__main__':main()
