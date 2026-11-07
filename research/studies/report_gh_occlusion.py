"""Dutch paper extracts for the exploratory occlusion proxy."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from research.data import atomic_json
from research.synthetic_hand import render_hand


def main():
    out=Path('results/gh_occlusion_01');paper=Path('paper')
    scores=pd.read_csv(out/'metrics.csv',keep_default_na=False)
    scores=scores[scores.fold=='pooled'].set_index('model')
    pairs=pd.read_csv(out/'paired.csv',keep_default_na=False)
    pairs=pairs[(pairs.fold=='pooled')&(pairs.reference=='original')].set_index('model')
    print(scores.to_string());print(pairs.to_string())
    labels={'original':'Oorspronkelijk','gh_y180':'Alleen rotatie','noise002':'Rotatie + 2\\% puntverstoring','noise005':'Rotatie + 5\\% puntverstoring'}
    lines=[]
    for arm,label in labels.items():
        r=scores.loc[arm];pair=pairs.loc[arm]
        lines.append(f"{label} & {r.macro_precision*100:.5f} & {r.macro_f1:.6f} & {int(r.correct):,} & {int(pair.fixed)} / {int(pair.lost)} \\\\")
    (paper/'occlusion_results_nl.tex').write_text('\n'.join(lines).replace(',','.')+'\n')
    synthetic=pd.read_csv(out/'synthetic_test_metrics.csv',keep_default_na=False)
    synthetic=synthetic[synthetic.fold=='pooled'].set_index('model')
    a,b=synthetic.loc['original'],synthetic.loc['gh_y180']
    (paper/'occlusion_main_nl.tex').write_text(
        f"Als we ook de G/H-toetslandmarks draaien, stijgt macro-F1 door rotatietraining van {a.macro_f1:.4f} naar {b.macro_f1:.4f}. De tweede vorm wordt dus geleerd. Echte handrugbeelden kunnen echter andere punten opleveren door verborgen vingers. Een aanvullende proef met verstoorde punten geeft hier geen extra winst. De eenvoudige nagebootste handbeelden worden door MediaPipe niet geaccepteerd (bijlage E).\n")
    (paper/'occlusion_interpretation_nl.tex').write_text(
        "De 2\\%-variant geeft op deze oorspronkelijke foto's exact dezelfde voorspellingen als alleen rotatie. De 5\\%-variant verliest één extra correcte voorspelling. Dit toont geen extra winst op het oorspronkelijke aanzicht; het beoordeelt de verstoorde punten nog niet als realistische handrugwaarnemingen.\n")
    lines=[]
    for arm,label in labels.items():
        r=synthetic.loc[arm]
        lines.append(f"{label} & {r.macro_precision*100:.4f} & {r.macro_f1:.6f} & {int(r.correct):,} \\\\")
    (paper/'occlusion_synthetic_nl.tex').write_text(
        "\n\\begin{table}[h]\\centering\\small\n"
        "\\caption{Kunstmatige toets: G/H-landmarks gedraaid, overige letters ongewijzigd. Dezelfde 60.251 bronnen; geen nieuwe RGB-detecties.}\n"
        "\\begin{tabular}{lrrr}\\toprule\nTraining & P (\\%) & Macro-F1 & Correct\\\\\\midrule\n"+
        '\n'.join(lines).replace(',','.')+
        "\n\\bottomrule\\end{tabular}\\end{table}\n"
        "Met alleen rotatie worden netto 4.773 extra letters goed herkend ten opzichte van de oorspronkelijke training. De recall op de gedraaide G/H-punten stijgt van 0\\% / 0,04\\% naar 99,47\\% / 99,45\\%. De puntverstoring geeft ook in deze controle geen extra winst. De rotatie leert dus de aangeboden geometrie, terwijl de vrijwel gelijke oorspronkelijke score alleen behoud op het oude aanzicht beschrijft.\n")
    fig,axes=plt.subplots(2,3,figsize=(10.3,6.8));fig.patch.set_facecolor('white')
    examples=json.loads(Path('results/gh_training_rotation_01/provenance.json').read_text())['examples']
    for row,example in enumerate(examples):
        path=Path(example['image'])
        if not path.exists():path=Path('data/asl_kaggle')/path
        with Image.open(path) as image:axes[row,0].imshow(image)
        for col,degrees in ((1,0),(2,180)):
            image,info=render_hand(example['original_image_landmarks'],degrees)
            axes[row,col].imshow(image)
            if col==2:
                points=np.asarray(info['projected_landmarks']);hidden=info['hidden_finger_joints']
                axes[row,col].scatter(points[hidden,0],points[hidden,1],s=30,facecolors='none',edgecolors='#b53635',linewidths=1.3)
        axes[row,0].set_ylabel(example['label'],fontsize=18,fontweight='bold')
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],('Echte bronfoto','Geschat volume: 0°','Geschat volume: 180°')):ax.set_title(title,fontsize=12,pad=9)
    fig.text(.5,.02,'Rode cirkels: gewrichten achter een ander geschat handdeel; geen nieuwe MediaPipe-detectie.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.04,right=.99,bottom=.06,top=.94,wspace=.04,hspace=.08)
    figure=paper/'figures/gh_occlusion_proxy.pdf';fig.savefig(figure);plt.close(fig)
    atomic_json(out/'paper_extracts.json',{'metrics_sha256':sha256((out/'metrics.csv').read_bytes()).hexdigest(),
            'paired_sha256':sha256((out/'paired.csv').read_bytes()).hexdigest(),
            'figure_examples':[e['id'] for e in examples],
            'figure_scope':'Illustrative source-derived approximate volume, not a photographed dorsal hand or new detector landmarks',
            'generated':{str(p):sha256(p.read_bytes()).hexdigest() for p in [figure,*sorted(paper.glob('occlusion_*_nl.tex'))]}})


if __name__=='__main__':main()
