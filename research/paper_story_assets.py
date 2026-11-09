"""Regenerate the report's plain-language overview and recorded C examples."""
from pathlib import Path
import json,csv
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
root=Path.cwd();paper=root/'paper';fig=paper/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
f,ax=plt.subplots(figsize=(6.8,1.25));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,2))
labels=['Notice webcam\nconfusions','Describe fingers\nand hand shape','Compare models\non the same images','Test wrist-tilt\ntraining variants']
for i,label in enumerate(labels):
 x=.1+i*2.5;ax.add_patch(FancyBboxPatch((x,.6),2.2,1.05,boxstyle='round,pad=.05',facecolor='#e8f0f4',edgecolor='#37576a'));ax.text(x+1.1,1.13,label,ha='center',va='center',fontsize=9)
 if i<3:ax.annotate('',(x+2.47,1.13),(x+2.27,1.13),arrowprops={'arrowstyle':'->','color':'#37576a'})
f.savefig(fig/'research_story.pdf',bbox_inches='tight');plt.close(f)
# Three real C poses, one repetition, cropped to the measured hand; no faces.
paths=list((root/'captures/guided/development_20261031T160814Z_3dd178e6').glob('*/capture.json'))
bytask={json.loads(p.read_text())['protocol']['task']['id']:p for p in paths}
from matplotlib.font_manager import findfont
label_font=ImageFont.truetype(findfont('DejaVu Sans'),17)
sheet=Image.new('RGB',(960,350),'white');draw=ImageDraw.Draw(sheet)
for i,(tilt,title) in enumerate([('neutraal','Neutral wrist'),('pols_omhoog','Wrist raised'),('pols_omlaag','Wrist lowered')]):
 p=bytask['C_'+tilt+'_1'];r=json.loads(p.read_text());im=Image.open(p.parent/'model_input.png').convert('RGB');w,h=im.size;pts=np.array(r['detection']['image_landmarks'])[:,:2]*[w,h];lo=np.maximum(pts.min(0)-12,0);hi=np.minimum(pts.max(0)+12,[w,h]);im=ImageOps.contain(im.crop((*lo,*hi)),(295,275));x=i*320;sheet.paste(im,(x+(320-im.width)//2,40+(275-im.height)//2));draw.text((x+12,8),title+' / predicted '+r['predicted_letter'],fill='black',font=label_font)
sheet.save(fig/'C_wrist_observation.png')
# Friendly table derived from frozen metrics, not copied rounded values.
import csv
metrics=list(csv.DictReader((root/'results/angles/metrics.csv').open(encoding='utf-8-sig')))
base=next(r for r in metrics if r['model']=='svm' and r['variant']=='shin' and r['fold']=='pooled')
angles=next(r for r in metrics if r['model']=='svm' and r['variant']=='shin_angles' and r['fold']=='pooled')
all_cv=json.loads((root/'results/tilt_augmentation_01/cv_summary.json').read_text())['tilt15'];only_cv=json.loads((root/'results/tilt_C_only_01/cv_summary.json').read_text())['tilt15']
ref_live=json.loads((root/'results/live_reference_comparison.json').read_text())
angle_cap=json.loads((root/'results/tilt_augmentation_01/capture_summary.json').read_text())
c_cap=json.loads((root/'results/tilt_C_only_01/capture_summary.json').read_text())
rows=[('Reference',float(base['accuracy']),float(base['macro_f1']),ref_live['correct']),('With joint angles',float(angles['accuracy']),float(angles['macro_f1']),angle_cap['development']['baseline']['correct']),('With tilt training',all_cv['accuracy'],all_cv['macro_f1'],angle_cap['development']['tilt15']['correct']),('C-only tilt trial',only_cv['accuracy'],only_cv['macro_f1'],c_cap['development']['tilt15']['correct'])]
(paper/'story_results.tex').write_text('\n'.join(f'{name} & {100*a:.2f} & {ff:.4f} & {n}/27 '+r'\\' for name,a,ff,n in rows)+'\n')
print('Narrative figures and tables created.')
