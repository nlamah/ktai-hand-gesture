"""Vector teaching diagrams for the hand-feature appendix; no measured examples."""
from hashlib import sha256
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from research.data import atomic_json

FIG = Path('paper/figures')
OUT = Path('results/feature_illustrations_01')
TEAL = '#087f8c'
THUMB = '#ac542f'
GREY = '#b9c6d1'
INK = '#213547'


def stroke(ax, points, color, width=7, outline=False):
    xs, ys = zip(*points)
    if outline:
        ax.plot(xs, ys, color='white', linewidth=width+3, solid_capstyle='round', solid_joinstyle='round')
    ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle='round', solid_joinstyle='round')


def hand(ax, offset, mode='closed', highlight=('index','middle'), thumb='out'):
    """Stylized palm and fingers; coordinates are drawings, not MediaPipe inputs."""
    def shifted(points):return [(x+offset,y) for x,y in points]
    palm = FancyBboxPatch((offset+19,24),43,34,boxstyle='round,pad=0,rounding_size=8',
                         facecolor='#edf2f5',edgecolor=GREY,linewidth=1.2)
    ax.add_patch(palm)
    stroke(ax, shifted([(29,15),(29,26)]), GREY, width=6)
    stroke(ax, shifted([(52,15),(52,26)]), GREY, width=6)
    fingers={
        'index':[(25,54),(25,66),(25,78),(25,86)],
        'middle':[(36,54),(36,70),(36,84),(36,94)],
        'ring':[(47,54),(47,68),(47,79),(47,88)],
        'pinky':[(58,53),(58,64),(58,73),(58,79)],
    }
    if mode=='spread_index':fingers['index']=[(25,54),(20,66),(15,77),(10,85)]
    if mode=='spread_ring':
        fingers['middle']=[(36,54),(33,70),(30,84),(28,94)]
        fingers['ring']=[(47,54),(51,68),(55,80),(58,88)]
        fingers['pinky']=[(58,53),(64,64),(68,72),(71,78)]
    if mode=='crossed':
        fingers['index']=[(25,54),(30,66),(37,78),(43,86)]
        fingers['middle']=[(36,54),(32,70),(27,84),(24,94)]
    if mode in ('contact','no_contact'):
        fingers['index'] = ([(25,54),(31,66),(24,74),(15,70)] if mode=='contact'
                            else [(25,54),(29,66),(29,77),(24,84)])
    order=['pinky','ring','middle','index']
    for name in order:
        points=shifted(fingers[name]);active=name in highlight
        stroke(ax,points,TEAL if active else GREY,width=7,outline=(mode=='crossed' and name=='index'))
        if active:
            xs,ys=zip(*points[1:-1]);ax.scatter(xs,ys,s=9,color='white',zorder=5)
    points = ([(22,31),(30,39),(40,46),(50,48)] if thumb=='across'
              else [(22,31),(12,40),(5,49),(2,57)])
    if mode=='contact':points=[(22,31),(9,45),(5,61),(15,70)]
    if mode=='no_contact':points=[(22,31),(9,42),(4,54),(9,65)]
    stroke(ax,shifted(points),THUMB if 'thumb' in highlight else GREY,width=7,outline=True)
    if mode.startswith('spread'):
        x1,x2,y = (11,36,77) if mode=='spread_index' else (30,56,80)
        ax.annotate('',xy=(offset+x1,y),xytext=(offset+x2,y),
                    arrowprops={'arrowstyle':'<->','color':TEAL,'lw':1.4})
    if mode=='contact':ax.scatter([offset+15],[70],s=140,facecolors='none',edgecolors=THUMB,lw=1.5,zorder=6)


def panel(ax, title):
    ax.set(xlim=(-2,190),ylim=(0,126));ax.axis('off')
    ax.add_patch(FancyBboxPatch((-1,1),190,124,boxstyle='round,pad=0,rounding_size=4',
                              facecolor='white',edgecolor='#d0dce5',linewidth=.9))
    ax.text(7,119,'\n'.join(textwrap.wrap(title,37)),ha='left',va='top',fontsize=11.8,
            color=INK,weight='bold',linespacing=1.15)


def pair(ax, title, left, right, left_label, right_label, highlight=('index','middle'), thumb_modes=('out','out')):
    panel(ax,title)
    hand(ax,11,left,highlight,thumb_modes[0]);hand(ax,104,right,highlight,thumb_modes[1])
    ax.text(52,6,left_label,ha='center',va='bottom',fontsize=10.3,color=INK)
    ax.text(145,6,right_label,ha='center',va='bottom',fontsize=10.3,color=INK)


def main():
    FIG.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(9,9),layout='constrained')
    ax=axes[0,0];panel(ax,'A. Buiging per gewricht')
    for x,state in [(33,'Gestrekt'),(94,'Deels gebogen'),(155,'Gebogen')]:
        points = ([(x,33),(x,51),(x,70),(x,88)] if state=='Gestrekt' else
                  [(x,33),(x,51),(x+9,66),(x+18,73)] if state=='Deels gebogen' else
                  [(x,33),(x,51),(x+17,55),(x+18,40)])
        stroke(ax,points,TEAL,8)
        xs,ys=zip(*points[:-1]);ax.scatter(xs,ys,s=28,facecolors='white',edgecolors=TEAL,zorder=5)
        ax.text(x,10,state.replace('Deels gebogen','Deels\ngebogen'),ha='center',va='bottom',fontsize=10.3,color=INK)
    for y,name in [(33,'MCP'),(51,'PIP'),(70,'DIP')]:ax.text(7,y,name,ha='left',va='center',fontsize=8.4,color=INK)
    pair(axes[0,1],'B. Vingers naast of over elkaar','closed','crossed',
         'Naast elkaar','Vermoede kruising')
    pair(axes[1,0],'C. Spreiding: wijs- en middelvinger','closed','spread_index',
         'Niet gespreid','Gespreid')
    pair(axes[1,1],'D. Spreiding: middel- en ringvinger','closed','spread_ring',
         'Niet gespreid','Gespreid',highlight=('middle','ring'))
    pair(axes[2,0],'E. Idee achter de duimstand','closed','closed',
         'Over de palm','Naast de palm',highlight=('thumb',),thumb_modes=('across','out'))
    pair(axes[2,1],'F. Contact van de duimtop','contact','no_contact',
         'Contact','Geen contact',highlight=('thumb','index'))
    path=FIG/'feature_concepts_nl.pdf'
    fig.savefig(path,bbox_inches='tight',metadata={'Title':'Schematische uitleg van de eigen handkenmerken',
                                                'Author':'Projectillustraties uit code; geen meetdata'})
    plt.close(fig)
    screenshot=FIG/'feature_tool_screenshot.png'
    atomic_json(OUT/'provenance.json',{
        'kind':'Vector explanatory diagrams; stylized coordinates, not measured landmarks or classification results.',
        'figure':str(path),'figure_sha256':sha256(path.read_bytes()).hexdigest(),
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'screenshot':str(screenshot),'screenshot_sha256':sha256(screenshot.read_bytes()).hexdigest(),
        'screenshot_source':'User-supplied Screenshot 2026-11-07 at 14.10.54.png; copied unchanged',
        'screenshot_zoom_pixel_boxes':{'index_pip':[617,655,807,715],
                                       'spread_index_middle':[225,725,414,788]},
        'technical_sources_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in
            map(Path,('research/geometry.py','research/workbook.py','config/thresholds.json','assets/asl_feature_references.xlsx'))},
        'interpretation':'Not-spread does not imply contact. Crossing is suspected. Thumb-over-palm concept is approximated by signed MCP deviation, not a palm-area overlap detector.'})
    print(f'Created vector feature overview: {path}',flush=True)


if __name__=='__main__':main()
