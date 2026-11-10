from pathlib import Path
import re,json,hashlib,csv,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from research.feature_review import CONNECTIONS,hand_crop

ATTACH=Path('/Users/tonheijligers/.codex/attachments/30356e58-b872-42c4-9afe-90514bd7a8f2/Pasted text.txt')
OUT=Path('paper/group_integration');AUDIT=Path('results/group_integration_20261107');AUDIT.mkdir(parents=True,exist_ok=True)
original=ATTACH.read_text();text=original;ops=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(old,new,reason):
 global text
 assert text.count(old)==1,(reason,text.count(old))
 text=text.replace(old,new,1);ops.append({'old':old,'new':new,'reason':reason})
def snippet(name):return (OUT/f'ton_{name}_en.tex').read_text().strip()
def tagged(name,s):return f'\n% BEGIN TON: {name}\n{s}\n% END TON: {name}\n'
def hidden(s):return '\\iffalse\n'+s+'\n\\fi'
def dots():return '\n\\par\\noindent ...\\par\\medskip\n'
# Resolve an existing package conflict without changing any prose.
replace('\\usepackage{subfigure}',hidden('\\usepackage{subfigure}'),'Technical: suppress legacy subfigure; source already loads subcaption')
for package in ['multirow','caption','subcaption','listings']:
 old='\\usepackage{'+package+'}'
 replace(old,'\\IfFileExists{'+package+'.sty}{'+old+'}{}','Technical: optional local-preview package '+package)
replace('\\usepackage{hyperref}','\\usepackage{hyperref}'+tagged('portable type encoding','\\hypersetup{hypertexnames=false}'),'Technical additions only')
replace('\\usepackage[accepted]{icml2021}','\\usepackage[accepted]{icml2021}'+tagged('font encoding','\\usepackage[T1]{fontenc}\n\\makeatletter\\let\\tonTableInput\\@@input\\makeatother'),'Technical: font encoding after ICML selects its font family')
layout = r"""\let\tonOriginalTexttt\texttt
\renewrobustcmd{\texttt}[1]{%
\ifstrequal{#1}{experiments/hand\_detection\_recovery/plot\_results.py}%
{\tonOriginalTexttt{experiments/\allowbreak hand\_detection\_recovery/\allowbreak plot\_results.py}}%
{\tonOriginalTexttt{#1}}}
"""
replace('\\usepackage[T1]{fontenc}', '\\usepackage[T1]{fontenc}'+tagged('line breaks for inherited long code path',layout),'Technical line wrapping of one inherited code path; its text remains identical')
# Only the template introduction is replaced; the student's own introductory paragraphs stay intact.
start=text.index('Write a brief introduction of your chosen assignment')
end=text.index('\\\\\\\\',start)
old=text[start:end]
intro=("This project investigates how a computer can locate a hand and recognize the ASL letter shown in a photograph. We study three complementary changes: targeted training augmentation, a second detection attempt after image preprocessing, and descriptions of hand shape. MediaPipe estimates 21 hand points; a classifier then learns from their geometry \\citep{ton-zhang2020,ton-shin2021}. We distinguish detection coverage from correct letter recognition: more accepted hand detections do not automatically mean better classification. Static A--Z images are studied, rather than continuous signing or the movements of J and Z.\n")
replace(old,tagged('shared introduction',intro)+'\\leavevmode ','Authorized shared introduction edit: replace template instructions')
replace('    \\item Research Question 3','    \\item Do explicit hand-shape categories and joint angles improve on, or replace, Shin\'s landmark directions?','Fill own RQ3 placeholder; retain RQ1/RQ2 wording and order')
# Suppress only literal template scaffolding. Its original bytes remain in the source.
a=text.index('\\begin{abstract}');b=text.index('\\end{abstract}',a)+len('\\end{abstract}')
old=text[a:b]
replace(old,hidden(old),'Template-only abstract instruction retained but not printed')
replace(hidden(old),hidden(old)+tagged('local preview notice','\\InputIfFileExists{ton_preview_notice.tex}{}{}'),'Conditional preview notice; absent in final group project')
for beginning,ending in [
 ('Briefly provide an overview of previous research','\\section{Methods}'),
 ('Describe the methods used. How it works.','\\subsection{Image Preprocessing}'),
]:
 a=text.index(beginning);b=text.index(ending,a);old=text[a:b]
 own=snippet('related') if beginning.startswith('Briefly') else ''
 replace(old,hidden(old)+(tagged('related work',own) if own else '')+'\n','Suppress template-only instructions')
# New methods follow the intact coauthor methods.
replace('\\section{Experimental Results}',tagged('methods',dots()+snippet('methods')+dots())+'\n\\section{Experimental Results}','Insert own method block after coauthor method')
# Dataset contribution replaces only a generic template prompt before the existing dots.
old='Include a description of the data used, what it contains, train test splits, etc.'
replace(old,hidden(old)+tagged('datasets',snippet('dataset'))+'\\leavevmode ','Replace generic dataset instruction, retain separator/coauthor data')
# Own implementation fills the first standalone placeholder; the second remains the original separator.
old='\\subsection{Implementation Details}\n        ...'
replace(old,'\\subsection{Implementation Details}'+tagged('implementation',snippet('implementation'))+'\\leavevmode ','Fill empty own implementation slot')
# Hide result/discussion prompts and fill only the empty own slot before the separator.
a=text.index('        Analyze and discuss your results.');b=text.index('        In the baseline run,',a)
old=text[a:b];prompt=old[:old.index('        \\\\')]
new=hidden(prompt)+tagged('results',snippet('results'))+dots()
replace(old,new,'Fill own result placeholder; coauthor results remain byte-identical')
a=text.index('        Additionally, you can include a separate discussion');b=text.index('        Dilation produced the highest',a)
old=text[a:b];prompt=old[:old.index('        \\\\')]
replace(old,hidden(prompt)+tagged('discussion',snippet('discussion'))+dots(),'Fill own discussion placeholder; coauthor discussion unchanged')
# Retain the template citation tutorial and dummy table literally, without printing them.
a=text.index('    \\subsection{Citing references}');b=text.index('\\bibliography{references}',a)
old=text[a:b];replace(old,hidden(old)+'\n','Template citation tutorial and dummy table suppressed, source retained')
replace('\\bibliography{references}','\\IfFileExists{references.bib}{\\bibliography{references,ton_references}}{\\bibliography{ton_references}}','Add prefixed own bibliography without replacing a group references file')
replace('\\end{document}',tagged('technical appendix',dots()+'\\input{ton_appendix_en.tex}')+'\n\\end{document}','Append own technical supplement after coauthor code inputs')
# Source-level literals remain visible in the main file for easy group editing.
(OUT/'group_report.tex').write_text(text)
(OUT/'source_group_original.tex').write_text(original)
# Strong preservation check: every substantive coauthor block is still verbatim and contiguous.
protected=[
 ('intro','In the landmark-based recognition pipeline','\n\n%   Image N47'),
 ('methods','\\subsection{Image Preprocessing}','\\section{Experimental Results}'),
 ('datasets','For RQ2, we use the A-Z training images','    \\subsection{Implementation Details}'),
 ('implementation','MediaPipe Hand Landmarker processes each image','    \\subsection{Results}'),
 ('results','In the baseline run, 60,251','    \\subsection{Discussion}'),
 ('discussion','Dilation produced the highest technical recovery','    \\subsection{Citing references}'),
]
checks=[]
for name,a,b in protected:
 chunk=original[original.index(a):original.index(b,original.index(a))]
 # method block ends immediately before the new addition; do not count whitespace after it.
 if name=='methods':chunk=chunk.rstrip()
 assert chunk in text,('changed coauthor block',name)
 checks.append({'block':name,'bytes':len(chunk.encode()),'sha256':hashlib.sha256(chunk.encode()).hexdigest(),'verbatim':True})
for question in ["Does targeted data augmentation reduce confusion between certain letters more effectively than general data augmentation?","To what extent does applying a \\(3 \\times 3\\) dilation filter recover visually plausible hand landmarks in ASL Alphabet training images for which MediaPipe initially detects \\textit{no\\_hand}?"]:
 assert question in text
restored=text
for op in reversed(ops):
 assert restored.count(op['new'])==1,op['reason'];restored=restored.replace(op['new'],op['old'],1)
assert restored==original
# Prefix all own BibTeX keys so a group bibliography cannot collide.
bib=Path('paper/references.bib').read_text();bib=re.sub(r'^(@\w+\{)([^,]+),',lambda m:m[1]+'ton-'+m[2]+',',bib,flags=re.M)
(OUT/'ton_references.bib').write_text(bib)
for f in ['icml2021.sty','icml2021.bst','algorithm.sty','algorithmic.sty']:shutil.copyfile(Path('paper')/f,OUT/f)
(OUT/'figures').mkdir(exist_ok=True)
shutil.copyfile('paper/figures/feature_tool_screenshot.png',OUT/'figures/ton_feature_tool_screenshot.png')
# Verified numerical extracts; no rerun of experiments or classifiers.
def csvrows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
metrics=csvrows('results/feature_factorial_01/metrics.csv');variants=['DS','D','DSE','DE','DSJ','DJ','DSEJ','DEJ']
names={'DS':'Distances + directions (DS)','D':'Distances only (D)','DSE':'DS + own categories (DSE)','DE':'D + own categories (DE)','DSJ':'DS + world joints (DSJ)','DJ':'D + world joints (DJ)','DSEJ':'DS + categories + joints (DSEJ)','DEJ':'D + categories + joints (DEJ)'}
lines=[]
for v in variants:
 r=[next(r for r in metrics if r['model']==model and r['variant']==v and r['fold']=='pooled') for model in ['svm','lightgbm']]
 vals=' & '.join(f"{float(x['macro_precision'])*100:.2f} & {float(x['macro_f1']):.4f}" for x in r)
 lines.append(names[v]+' & '+vals+' \\\\')
(OUT/'ton_feature_rows_en.tex').write_text('\n'.join(lines)+'\n')
# Paired rows are already published with language-neutral variant IDs and integer counts.
paired_rows=csvrows('results/feature_factorial_01/paired_changes.csv');lines=[]
for v in variants:
    values=[]
    for model in ['svm','lightgbm']:
        r=next(r for r in paired_rows if r['model']==model and r['variant']==v and r['fold']=='pooled')
        values += [f"{int(r['fixed']):,}",f"{int(r['lost']):,}",f"{int(r['net']):+,}"]
    lines.append(v+' & '+' & '.join(values)+' \\\\')
(OUT/'ton_paired_rows_en.tex').write_text('\n'.join(lines)+'\n')
# Translate display labels only; preserve published accepted-only values exactly.
s=Path('paper/handedness_rows_nl.tex').read_text();lines=[]
for l in s.splitlines():
 if not l.startswith('SVM'):continue
 l=l.removeprefix('SVM & ').replace(' & 60.251 & ',' & 60,251 & ').replace(' & 60.201 & ',' & 60,201 & ').replace(' & 60.225 & ',' & 60,225 & ')
 for a,b in [('Origineel','Original'),('Gespiegeld','Reflected RGB'),('Vast mengsel','Fixed mixture'),('Zonder aanpassing','Original training'),('Spiegeltraining','Mirror training'),('Rechtsnormalisatie','Right normalized')]:l=l.replace(a,b)
 cells=[c.strip() for c in l.removesuffix('\\\\').strip().split('&')]
 for i in [3,4]:cells[i]=cells[i].replace(',','.')
 cells[5]=cells[5].replace('.',',')
 lines.append(' & '.join(cells)+' \\\\')
(OUT/'ton_orientation_rows_en.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for condition,p in [('Original','results/gh_occlusion_01/metrics.csv'),('Artificial G/H rotation','results/gh_occlusion_01/synthetic_test_metrics.csv')]:
 rows=csvrows(p)
 for arm,name in [('original','Original training'),('gh_y180','G/H rotation training')]:
  r=next(r for r in rows if r['model']==arm and r['fold']=='pooled');lines.append(f"{condition} & {name} & {float(r['macro_precision'])*100:.4f} & {float(r['macro_f1']):.6f} & {int(r['correct']):,} \\\\")
(OUT/'ton_gh_rows_en.tex').write_text('\n'.join(lines)+'\n')
# English labels for the same four diagnostic photos; no changes to original files.
rows=csvrows('results/gh_live_diagnostics_20261107/predictions.csv');fig,axes=plt.subplots(2,2,figsize=(9.2,6.2),layout='constrained')
for r,ax in zip(rows,axes.flat):
 folder=Path('captures/confusions')/r['id'];record=json.loads((folder/'capture.json').read_text());d=record['detection']
 with Image.open(folder/'model_input.png') as im:
  crop,box=hand_crop(im,d['image_landmarks'],overlay=False);points=np.asarray(d['image_landmarks'])[:,:2]*im.size-np.array(box[:2])
 ax.imshow(crop)
 for a,b in CONNECTIONS:ax.plot(points[[a,b],0],points[[a,b],1],color='#00ba89',lw=1.5)
 ax.scatter(points[:,0],points[:,1],s=13,c='#ffcc33',edgecolors='none',zorder=3)
 view='back visible' if r['view_visual']=='handrug zichtbaar' else 'palm visible'
 ax.set_title(f"{r['panel']}. Intended {r['expected']} | {view}\nReference: {r['reference']} | Rotation: {r['rotation']}",fontsize=11);ax.axis('off')
fig.savefig(OUT/'figures/ton_live_examples_en.pdf');plt.close(fig)
(AUDIT/'edits.json').write_text(json.dumps(ops,indent=2)+'\n')
(AUDIT/'preservation.json').write_text(json.dumps({'source_attachment':str(ATTACH),'source_sha256':sha(ATTACH),'original_copy_sha256':sha(OUT/'source_group_original.tex'),'merged_tex_sha256':sha(OUT/'group_report.tex'),'reversible_edit_check':True,'coauthor_blocks':checks,'RQ1_and_RQ2_verbatim':True,'all_new_prose_language':'English','own_RQ':'RQ3','original_group_research_question_numbering_preserved':True,'own_blocks_marked':'BEGIN TON / END TON; dots retained or added','template_only_suppression':'Original generic prompts, abstract instruction, citation tutorial and example table retained under iffalse; not printed','technical_changes':'Disable legacy subfigure conflict; optional availability guards for unused packages; T1 encoding; add noncolliding prefixed bibliography; own appendix input','missing_group_assets':['figures/recovery_by_variant.pdf','code-snippets/settings-all-run.tex','code-snippets/repeatability.tex'],'numeric_sources':{p:sha(p) for p in ['results/feature_factorial_01/metrics.csv','results/feature_factorial_01/paired_changes.csv','paper/handedness_rows_nl.tex','results/gh_occlusion_01/metrics.csv','results/gh_occlusion_01/synthetic_test_metrics.csv','results/gh_live_diagnostics_20261107/predictions.csv']}},indent=2)+'\n')
print('Verbatim coauthor blocks:',[(r['block'],r['bytes']) for r in checks]);print('Original restored exactly after reversing',len(ops),'explicit integration edits.')
