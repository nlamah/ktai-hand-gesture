"""Visualize every letter confusion before/after adding SVM joint angles."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scripts.cross_validate_shin_workbook import ANGLE_COLUMNS

LABELS=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
OUT=Path('experiments/kaggle_angles_scores_5fold_01/confusions')


def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parent=OUT.parent
    with (parent/'oof_predictions_svm.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    matrices={};records={}
    for variant in ('shin','shin_angles'):
        subset=[r for r in rows if r['variant']==variant]
        matrices[variant]=confusion_matrix([r['label'] for r in subset],[r['prediction'] for r in subset],labels=LABELS)
        records[variant]={r['id']:r for r in subset}
        write_csv(f'{variant}_counts.csv',[{'echte_letter':label,**dict(zip(LABELS,map(int,row)))} for label,row in zip(LABELS,matrices[variant])])
    before=matrices['shin'];after=matrices['shin_angles'];delta=after-before;support=before.sum(axis=1)
    assert np.array_equal(support,after.sum(axis=1)) and support.sum()==60251
    fig,axes=plt.subplots(1,2,figsize=(23,11),layout='constrained')
    maximum=max(np.max(np.where(np.eye(26,dtype=bool),0,m/support[:,None]*100)) for m in matrices.values())
    for ax,(name,matrix) in zip(axes,matrices.items()):
        rates=matrix/support[:,None]*100;np.fill_diagonal(rates,np.nan)
        cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#e6e6e6')
        im=ax.imshow(rates,cmap=cmap,vmin=0,vmax=maximum)
        for i in range(26):
            for j in range(26):
                if matrix[i,j]:ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=6.5,
                                      color='white' if i!=j and rates[i,j]>maximum*.6 else '#202020')
        ax.set_xticks(range(26),LABELS);ax.set_yticks(range(26),LABELS)
        ax.set_xlabel('Voorspelde letter');ax.set_ylabel('Echte letter')
        ax.set_title('Shin alleen' if name=='shin' else 'Shin + gewrichtshoeken')
    fig.colorbar(im,ax=axes,label='Verkeerd voorspeld (% van de echte letter)',shrink=.7)
    fig.suptitle('SVM: alle 60.251 out-of-foldvoorspellingen\nGetallen = aantallen; kleur = foutpercentage per rij; grijze diagonaal = correct',fontsize=14)
    fig.savefig(OUT/'confusion_before_after.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,12),layout='constrained')
    changes=delta.astype(float);np.fill_diagonal(changes,np.nan)
    bound=max(1,np.nanmax(np.abs(changes)));cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e6e6e6')
    im=ax.imshow(changes,cmap=cmap,vmin=-bound,vmax=bound)
    for i in range(26):
        for j in range(26):
            if i!=j and delta[i,j]:ax.text(j,i,f'{delta[i,j]:+d}',ha='center',va='center',fontsize=8,
                      color='white' if abs(delta[i,j])>bound*.6 else '#202020')
    ax.set_xticks(range(26),LABELS);ax.set_yticks(range(26),LABELS)
    ax.set_xlabel('Voorspelde letter');ax.set_ylabel('Echte letter')
    ax.set_title('Verandering door gewrichtshoeken: rood = meer fouten, blauw = minder fouten')
    fig.colorbar(im,ax=ax,label='Verschil in aantal verkeerde voorspellingen')
    fig.savefig(OUT/'confusion_changes.png',dpi=150);plt.close(fig)
    write_csv('changes_counts.csv',[{'echte_letter':label,**dict(zip(LABELS,map(int,row)))} for label,row in zip(LABELS,delta)])
    pairs=[]
    for i,label in enumerate(LABELS):
        for j,pred in enumerate(LABELS):
            if i!=j and (before[i,j] or after[i,j]):
                pairs.append({'echte_letter':label,'voorspeld_als':pred,'shin':int(before[i,j]),
                              'met_hoeken':int(after[i,j]),'verschil':int(delta[i,j]),
                              'foutpercentage_met_hoeken':float(after[i,j]/support[i])})
    write_csv('all_error_pairs.csv',pairs)
    lines=['# Verwarring voor alle letters: SVM met en zonder gewrichtshoeken','',
           'Dezelfde vijf folds en 60.251 bruikbare Kaggle-beelden. Rijen zijn de echte letter; '
           'kolommen de voorspelling. Aantallen zijn out-of-fold: geen trainingsvoorspellingen.', '',
           '![Verwarringsmatrices voor en na toevoegen van hoeken](confusion_before_after.png)', '',
           '![Verandering van de verwarring](confusion_changes.png)', '',
           '## Overzicht per echte letter','',
           'Per rij maximaal drie grootste verkeerde bestemmingen na toevoegen van hoeken. '
           'Notatie `R: 149 → 142` betekent dat de echte letter uit deze rij eerst 149 keer als R '
           'werd voorspeld en daarna 142 keer. De CSV en matrices bevatten alle bestemmingen.', '',
           '| Echte letter | Beelden | Correct Shin → hoeken | Meeste verwarring na hoeken (Shin → hoeken) |',
           '|---|---:|---:|---|']
    for i,label in enumerate(LABELS):
        destinations=sorted((j for j in range(26) if i!=j and after[i,j]),key=lambda j:(-after[i,j],LABELS[j]))[:3]
        text='; '.join(f'{LABELS[j]}: {before[i,j]} → {after[i,j]}' for j in destinations) or 'Geen'
        lines.append(f'| {label} | {support[i]} | {before[i,i]} → {after[i,i]} | {text} |')
    lines+=['','## Grootste toename van specifieke fouten','',
            '| Echte letter → voorspelling | Shin | Met hoeken | Verschil |','|---|---:|---:|---:|']
    for row in sorted(pairs,key=lambda r:-r['verschil'])[:10]:
        lines.append(f"| {row['echte_letter']} → {row['voorspeld_als']} | {row['shin']} | {row['met_hoeken']} | {row['verschil']:+d} |")
    lines+=['','## Betekenis voor featureweging','',
            'Extra gewicht voor de drie wijsvingerhoeken is een zinvolle hypothese voor O/D, '
            'maar volgt niet automatisch uit de verwarringsmatrix. Meer gewicht kan bestaande '
            'landmarkfouten versterken en de SVM-grens voor andere letters verplaatsen. '
            'Een featuremultiplier 2 telt bovendien als factor 4 in het gekwadrateerde afstandsdeel van de RBF-kernel.', '',
            'De grootste resterende paren zijn M/N en R/U. Voor M/N zijn duim- en gezamenlijke '
            'vingerstanden plausibele kandidaten; voor R/U zijn zijdelingse volgorde en kruising '
            'kandidaten. Dit zijn onderzoekshypotheses, geen causaliteitsconclusies uit de matrix. '
            'De eerdere crossing-feature had geen duidelijke extra bijdrage; bekijk daarom eerst '
            'de landmarks voordat je haar zwaarder maakt.', '',
            'Volgende gerichte test: alleen de wijsvinger-MCP/PIP/DIP als groep wegen met '
            '0,5 / 1 / 2, gekozen in interne trainingsfolds. Beoordeel buitenfolds op totale '
            'macro-F1, O/D én M/N en R/U; kies niet alleen op verbetering van O. '
            'Inspecteer de acht nieuw foute O-beelden daarnaast visueel. '
            'Deze herhaald gebruikte ontwikkelfolds zijn geen onafhankelijke persoonstest.', '',
            'Bestanden: shin_counts.csv, shin_angles_counts.csv, changes_counts.csv en all_error_pairs.csv.','']
    cache=Path('experiments/kaggle_nested_feature_weights_01')
    metadata=json.loads((cache/'inputs_metadata.json').read_text())
    angles=np.load(cache/'inputs.npz')['angles']
    actual_labels=np.array([r['label'] for r in metadata['samples']])
    index_by_id={r['id']:i for i,r in enumerate(metadata['samples'])}
    lost=[index_by_id[key] for key,row in records['shin'].items()
          if row['label']=='O' and row['prediction']=='O'
          and records['shin_angles'][key]['prediction']=='D']
    diagnostics=[]
    for title,indices in [('O',np.flatnonzero(actual_labels=='O')),
                          ('D',np.flatnonzero(actual_labels=='D')),
                          ('Nieuw fout: O naar D',lost)]:
        entry={'groep':title,'beelden':len(indices)}
        for key in ('index_mcp_angle_deg','index_pip_angle_deg','index_dip_angle_deg'):
            entry[key]=float(np.median(angles[indices,ANGLE_COLUMNS.index(key)]))
        diagnostics.append(entry)
    (OUT/'index_angle_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    lines+=['## Wijsvinger: gerichte aanwijzing voor O/D','',
            'Mediane gemeten hoeken in graden (dit zijn geen getunede grenzen):','',
            '| Groep | Beelden | MCP | PIP | DIP |','|---|---:|---:|---:|---:|']
    for row in diagnostics:
        lines.append(f"| {row['groep']} | {row['beelden']} | {row['index_mcp_angle_deg']:.1f} | "
                     f"{row['index_pip_angle_deg']:.1f} | {row['index_dip_angle_deg']:.1f} |")
    lines+=['','De PIP-mediaan van de nieuw verkeerd herkende O-beelden ligt aan de O-kant. '
            'Dit maakt afzonderlijk meer gewicht voor PIP een toetsbare hypothese; niet noodzakelijk '
            'voor alle drie wijsvingerhoeken tegelijk. Medianen tonen niet de volledige overlap of '
            'de SVM-beslisgrens, dus verbetering staat niet vast.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    print(OUT/'report.md')
    print('Worst pairs:',sorted(pairs,key=lambda r:-r['met_hoeken'])[:6])


if __name__=='__main__':main()
