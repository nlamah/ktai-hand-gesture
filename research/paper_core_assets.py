"""Build core-report tables and plots from saved, matched experimental evidence."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'


def read(path):
    with (ROOT/path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def direction_control():
    """Reconcile paired counts and export the coordinate control for the paper."""
    metrics = {(r['model'], r['variant']): r
               for r in read('results/direction_joint_01/metrics.csv') if r['fold'] == 'pooled'}
    paired = {(r['model'], r['variant']): r
              for r in read('results/direction_joint_01/paired_changes.csv')}
    names = [('shin', 'Reference'),
             ('distance_joint', 'Distances + same-coordinate joints'),
             ('shin_joint', 'Shin + same-coordinate joints'),
             ('shin_world', 'Shin + world joints')]
    table = []
    for model in ('svm', 'lightgbm'):
        base = metrics[model, 'shin']
        for arm, label in names:
            row, changes = metrics[model, arm], paired[model, arm]
            images = int(row['images'])
            assert images == int(base['images']) == 60251
            fixed, lost, net = (int(changes[k]) for k in ('fixed', 'lost', 'net'))
            assert fixed - lost == net
            assert abs((float(row['accuracy']) - float(base['accuracy'])) * images - net) < 1e-8
            table.append(f"{'SVM' if model == 'svm' else 'LightGBM'} & {label} & "
                         f"{row['features']} & {100*float(row['accuracy']):.5f} & "
                         f"{float(row['macro_f1']):.6f} & {fixed} & {lost} & {net:+d} " + r'\\')
    (PAPER/'direction_joint_rows.tex').write_text('\n'.join(table) + '\n')
    return metrics


def main():
    control = direction_control()
    metrics={}
    for model in ['svm','lightgbm']:
        for r in read(f'results/angles/metrics_{model}.csv'):
            if r['fold']=='pooled':metrics[model,r['variant']]=r
        for r in read('results/model_comparison/model_metrics.csv'):
            if r['model']==model and r['variant']=='combined' and r['fold']=='pooled':metrics[model,'full']=r
    rows=[]
    for variant,label,width in [('shin','Reference',820),('full','All additions',931),('shin_angles','World joint measurements',835),('shin_excel','Workbook scores',846),('shin_angles_excel','World joints + workbook',861)]:
        a,b=metrics['svm',variant],metrics['lightgbm',variant]
        rows.append(f"{label} & {width} & {100*float(a['accuracy']):.3f} & {float(a['macro_f1']):.5f} & {100*float(b['accuracy']):.3f} & {float(b['macro_f1']):.5f} \\")
    for model in ('svm', 'lightgbm'):
        for arm, original in [('shin', 'shin'), ('shin_world', 'shin_angles')]:
            for metric in ('accuracy', 'macro_f1'):
                assert abs(float(control[model, arm][metric]) - float(metrics[model, original][metric])) < 1e-12
    a,b = control['svm','shin_joint'],control['lightgbm','shin_joint']
    rows.append(f"Same-coordinate joint measurements & 835 & {100*float(a['accuracy']):.3f} & "
                f"{float(a['macro_f1']):.5f} & {100*float(b['accuracy']):.3f} & {float(b['macro_f1']):.5f} " + '\\')
    (PAPER/'core_results.tex').write_text('\n'.join(r+'\\' for r in rows)+'\n')
    letters=read('results/angles/per_letter_svm.csv')
    base={r['letter']:r for r in letters if r['variant']=='shin'}
    angles={r['letter']:r for r in letters if r['variant']=='shin_angles'}
    deltas=np.array([int(angles[l]['correct'])-int(base[l]['correct']) for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
    assert deltas.sum()==67
    selected=[]
    for l in ['M','N','O','R','S','Z']:
        a,b=base[l],angles[l];delta=int(b['correct'])-int(a['correct'])
        selected.append(f"{l} & {a['images']} & {100*float(a['f1']):.2f} & {100*float(b['f1']):.2f} & {delta:+d} \\\\")
    (PAPER/'core_letter_rows.tex').write_text('\n'.join(selected)+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7,2.3))
    ax.bar(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),deltas,color=['#1b786b' if d>0 else '#bd654a' if d<0 else '#aaa' for d in deltas])
    ax.axhline(0,color='#48545c',lw=.7);ax.set_ylabel('Change in correct images');ax.set_xlabel('True letter')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for l in ['M','O']:
        i=ord(l)-65
        if deltas[i] > 0:
            ax.annotate(f'{deltas[i]:+d}',(i,deltas[i]),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        else:
            ax.text(i,deltas[i]/2,f'{deltas[i]:+d}',ha='center',va='center',color='white',fontsize=8)
    fig.tight_layout();fig.savefig(PAPER/'figures/core_letter_effects.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,1.15));ax.axis('off')
    labels=['Observe webcam\nconfusions','Describe hand shape\nwith categories / angles','Compare folds and\ncoordinate sources','Inspect gains, losses\nand feature reliance']
    for i,label in enumerate(labels):
        x=.005+i*.252
        ax.add_patch(FancyBboxPatch((x,.23),.235,.56,boxstyle='round,pad=0.006',transform=ax.transAxes,facecolor='#e9f1f4',edgecolor='#637e8a'))
        ax.text(x+.1175,.51,label,transform=ax.transAxes,ha='center',va='center',fontsize=8)
        if i<3:ax.annotate('',xy=(x+.25,.51),xytext=(x+.236,.51),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#637e8a'})
    fig.tight_layout(pad=.1);fig.savefig(PAPER/'figures/core_pipeline.pdf');plt.close(fig)
    print('Core and coordinate-control tables reconciled; world-joint net +67, same-coordinate net +4.')


if __name__=='__main__':main()
