from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from sklearn.metrics import confusion_matrix
from PIL import Image,ImageDraw
root=Path.cwd();out=root/'paper';fig=out/'figures';fig.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
a=pd.read_csv(root/'results/angles/metrics.csv');a=a[a.fold=='pooled']
# Main compact pipeline
f,ax=plt.subplots(figsize=(6.8,1.55));ax.set(xlim=(0,10),ylim=(0,2));ax.axis('off')
boxes=[(.05,.7,1.65,1,'78,000 images\nASL Alphabet A-Z'),(2,.7,1.65,1,'One hand detected\n60,251 images'),(4,.7,2.1,1,'820 Shin features\n+ selected additions'),(6.45,.7,1.5,1,'Frozen grouped\nfive-fold CV'),(8.3,.7,1.65,1,'SVM / LightGBM\nOOF evaluation')]
for x,y,w,h,t in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#e8f0f4',edgecolor='#37576a',linewidth=.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7)
for i in range(4):
 x,y,w,h,_=boxes[i];xx=boxes[i+1][0];ax.annotate('',(xx-.07,1.2),(x+w+.07,1.2),arrowprops=dict(arrowstyle='->',color='#37576a'))
ax.text(5,.24,'Preprocessing fitted on training folds only; groups are filename blocks, not people.',ha='center',fontsize=8)
f.savefig(fig/'pipeline.pdf');plt.close(f)
# Model effect
f,ax=plt.subplots(figsize=(3.25,2.05));variants=['shin_angles','shin_excel','shin_angles_excel'];labels=['+ joint angles','+ Excel scores','+ angles + scores'];yy=np.arange(3)
for model,offset,color in [('svm',-.16,'#285f82'),('lightgbm',.16,'#bd6a35')]:
 t=a[a.model==model].set_index('variant');v=[100*(t.loc[x,'accuracy']-t.loc['shin','accuracy']) for x in variants];ax.barh(yy+offset,v,.29,label='SVM' if model=='svm' else 'LightGBM',color=color)
ax.axvline(0,color='gray',lw=.6);ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlabel('Accuracy change (percentage points)');ax.legend(frameon=False,fontsize=7,loc='lower right');f.tight_layout();f.savefig(fig/'effects.pdf');plt.close(f)
# All letter metrics table and descriptive paired statistics
pred=pd.read_csv(root/'results/angles/oof_predictions_svm.csv');b=pred[pred.variant=='shin'].set_index('id');c=pred[pred.variant=='shin_angles'].set_index('id').loc[b.index];y=b.label.to_numpy();p=b.prediction.to_numpy();q=c.prediction.to_numpy();delta=(q==y).astype(int)-(p==y).astype(int)
meta=json.loads((root/'data/asl_kaggle/research/inputs_metadata.json').read_text());groups=np.array([r['group'] for r in meta['samples']]);ids=[r['id'] for r in meta['samples']];groupmap=dict(zip(ids,groups));g=np.array([groupmap[i] for i in b.index]);stats=[]
for group in sorted(set(g)):
 ix=g==group;stats.append({'group':group,'n':int(ix.sum()),'net_correct':int(delta[ix].sum()),'delta_pp':float(100*delta[ix].mean())})
rng=np.random.default_rng(42);ng=np.array([s['n'] for s in stats]);dg=np.array([s['net_correct'] for s in stats]);ix=rng.integers(0,len(stats),size=(10000,len(stats)));boot=100*dg[ix].sum(axis=1)/ng[ix].sum(axis=1)
stat={'fixed':int(((p!=y)&(q==y)).sum()),'lost':int(((p==y)&(q!=y)).sum()),'net':int(delta.sum()),'groups':stats,'group_resampling_95_sensitivity_pp':np.quantile(boot,[.025,.975]).tolist(),'description':'Descriptive resampling of six filename blocks; not an independent inferential confidence interval.'}
(out/'paired_statistics.json').write_text(json.dumps(stat,indent=2));print(stat)
l=pd.read_csv(root/'results/angles/per_letter_svm.csv');x=l[l.variant=='shin'].set_index('letter');z=l[l.variant=='shin_angles'].set_index('letter')
lines=[]
for letter in x.index:
 r=x.loc[letter];s=z.loc[letter];lines.append(f"{letter} & {int(r.images):,} & {100*r.precision:.2f} & {100*s.precision:.2f} & {100*r.recall:.2f} & {100*s.recall:.2f} & {100*r.f1:.2f} & {100*s.f1:.2f} & {int(s.correct-r.correct):+d} \\\\")
(out/'letter_rows.tex').write_text('\n'.join(lines)+'\n')
# Full confusion matrices, count display offdiagonal + diagonal correct totals at readable full width.
letters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
for name,pr in [('shin',p),('angles',q)]:
 cm=confusion_matrix(y,pr,labels=letters);f,ax=plt.subplots(figsize=(8.6,7.1));v=cm.copy();np.fill_diagonal(v,0);im=ax.imshow(v,cmap='Blues',vmin=0,vmax=150)
 for i in range(26):
  for j in range(26):
   if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=5.8,color=('#16724b' if i==j else 'white' if cm[i,j]>85 else '#182e40'))
 ax.set_xticks(range(26),letters);ax.set_yticks(range(26),letters);ax.set_xlabel('Predicted letter');ax.set_ylabel('True letter');ax.set_title('SVM: '+('Shin baseline' if name=='shin' else 'Shin + joint angles'));f.colorbar(im,ax=ax,shrink=.65,label='Off-diagonal error count');f.tight_layout();f.savefig(fig/f'confusion_{name}.pdf');plt.close(f)
# Actual images, not fabricated webcam observations.
manifest=json.loads((root/'data/asl_kaggle/manifest.json').read_text());byid={s['id']:s for s in manifest['samples']};new=np.flatnonzero((y=='O')&(p=='O')&(q=='D'))[:4]
f,axs=plt.subplots(1,4,figsize=(8,2.25))
for ax,i in zip(axs,new):
 s=byid[b.index[i]];image=Image.open(root/'data/asl_kaggle'/s['image_path']);ax.imshow(image);pts=np.array(s['image_landmarks']);ax.plot(pts[5:9,0]*image.width,pts[5:9,1]*image.height,'o-',color='#00ffb0',ms=3,lw=1);ax.set_title(f"O / Shin: O / +angles: D\n{s['filename']}",fontsize=8);ax.axis('off')
f.tight_layout();f.savefig(fig/'o_errors.pdf');plt.close(f)
# Joint angle schematic
f,ax=plt.subplots(figsize=(3.3,2.05));ax.plot([0,1,1.7],[0,0,.7],'o-',lw=3,color='#285f82');ax.text(0,-.15,'a',ha='center');ax.text(1,-.15,'b',ha='center');ax.text(1.7,.82,'c',ha='center');ax.text(.68,.2,r'$\theta$',fontsize=17);ax.text(.85,-.42,'180 degrees = straight joint',ha='center');ax.set(xlim=(-.25,2),ylim=(-.55,1));ax.axis('off');f.tight_layout();f.savefig(fig/'angle.pdf');plt.close(f)
# pipeline result table
rows=[]
for variant,label in [('shin','Shin'),('shin_angles','Shin + angles'),('shin_excel','Shin + Excel'),('shin_angles_excel','Shin + angles + Excel')]:
 s=a[(a.model=='svm')&(a.variant==variant)].iloc[0];t=a[(a.model=='lightgbm')&(a.variant==variant)].iloc[0];rows.append(f"{label} & {int(s.features)} & {100*s.accuracy:.3f} & {s.macro_f1:.5f} & {100*t.accuracy:.3f} & {t.macro_f1:.5f} \\\\")
(out/'result_rows.tex').write_text('\n'.join(rows)+'\n')
