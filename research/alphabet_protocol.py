"""Named, example-guided A-Z capture sessions with frozen paired models."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from research.data import atomic_json
from research.protocol import ProtocolSession
from research.inputs import LABELS

ROOT=Path(__file__).resolve().parents[1]
EXAMPLES=ROOT/'config/alphabet_capture_examples.json'
SCHEMA='asl-alphabet-capture-v1'


def tasks(repeats=5):
    if not isinstance(repeats,int) or isinstance(repeats,bool) or not 1<=repeats<=10:
        raise ValueError('Choose 1 to 10 repetitions.')
    return [{'id':f'{letter}_{repeat}','letter':letter,'repeat':repeat,'participant':'participant'}
            for repeat in range(1,repeats+1) for letter in LABELS]


def profile(name,kind,experience,hand):
    name=name.strip()
    if not name or len(name)>80:raise ValueError('Enter a name or participant code of at most 80 characters.')
    if kind not in ('development','new'):raise ValueError('Choose a development participant or a new test participant.')
    if experience not in ('beginner','familiar','experienced','unknown'):raise ValueError('Unknown experience level.')
    if hand not in ('right','left'):raise ValueError('Choose left or right.')
    return {'name':name,'kind':kind,'experience':experience,'hand':hand}


def participant_form():
    """Native text entry supports names with spaces/accents; no camera yet."""
    import tkinter as tk
    from tkinter import ttk,messagebox
    window=tk.Tk();window.title('ASL captures - participant');window.resizable(False,False)
    container=ttk.Frame(window,padding=22);container.grid()
    ttk.Label(container,text='Capture all letters',font=('Helvetica',18,'bold')).grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,12))
    ttk.Label(container,text='Name or participant code').grid(row=1,column=0,sticky='w')
    name=ttk.Entry(container,width=32);name.grid(row=1,column=1,pady=7);name.focus_set()
    kind=tk.StringVar(value='');experience=tk.StringVar(value='unknown');hand=tk.StringVar(value='right')
    ttk.Label(container,text='Previous development involvement').grid(row=2,column=0,sticky='nw',pady=7)
    choices=ttk.Frame(container);choices.grid(row=2,column=1,sticky='w')
    ttk.Radiobutton(choices,text='My hand has already been used during development',variable=kind,value='development').pack(anchor='w')
    ttk.Radiobutton(choices,text='New participant, not previously used',variable=kind,value='new').pack(anchor='w')
    ttk.Label(container,text='ASL fingerspelling experience').grid(row=3,column=0,sticky='w')
    levels={'Unknown':'unknown','Beginner':'beginner','Some familiarity':'familiar','Experienced':'experienced'}
    display=tk.StringVar(value='Unknown');ttk.Combobox(container,textvariable=display,values=list(levels),state='readonly',width=29).grid(row=3,column=1,pady=7)
    hands=ttk.Frame(container);hands.grid(row=4,column=1,sticky='w',pady=7)
    ttk.Label(container,text='Hand for all captures').grid(row=4,column=0,sticky='w')
    ttk.Radiobutton(hands,text='Right',variable=hand,value='right').pack(side='left')
    ttk.Radiobutton(hands,text='Left',variable=hand,value='left').pack(side='left')
    ttk.Label(container,text='Default: 5 A-Z rounds = 130 photos per person.\nPredictions stay hidden. Space saves immediately.').grid(row=5,column=0,columnspan=2,sticky='w',pady=14)
    result=[]
    def start():
        try:result.append(profile(name.get(),kind.get(),levels[display.get()],hand.get()))
        except ValueError as e:messagebox.showerror('Check participant details',str(e));return
        window.destroy()
    ttk.Button(container,text='Start capture',command=start).grid(row=6,column=0,columnspan=2,sticky='ew')
    window.bind('<Return>',lambda event:start());window.mainloop()
    return result[0] if result else None


class AlphabetSession(ProtocolSession):
    participant_label='Participant'
    capture_immediately=True
    hide_landmarks=True
    capture_cooldown=1.0

    def __init__(self,directory,model_sha):
        self.directory=Path(directory);self.plan=json.loads((self.directory/'session.json').read_text(encoding='utf-8'))
        if self.plan.get('schema')!=SCHEMA or self.plan.get('tasks')!=tasks(self.plan['repeats']):raise ValueError('Modified or unknown alphabet protocol.')
        if self.plan['model_sha256']!=model_sha:raise ValueError('The capture model has changed.')
        for path,expected in self.plan['frozen_models'].items():
            if sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('A comparison model has changed.')
        if profile(**self.plan['participant'])!=self.plan['participant']:
            raise ValueError('Invalid participant details.')
        expected_role='development' if self.plan['participant']['kind']=='development' else 'evaluation'
        if self.plan['role']!=expected_role:raise ValueError('Participant role has changed.')
        if set(self.plan['examples'])!=set(LABELS):raise ValueError('Examples are missing.')
        self.mirror_preview=self.plan.get('display',{}).get('preview_mirrored',True)
        self.example_images={}
        for letter,info in self.plan['examples'].items():
            path=ROOT/info['path']
            if sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('An example has changed.')
            image=cv2.imread(str(path))
            if image is None:raise ValueError('An example cannot be opened.')
            # Old sessions retain their recorded display. New sessions normalize each example to right hand.
            self.example_images[letter]=cv2.flip(image,1) if info.get('display_mirror',True) else image
        self.participant_confirmed=None;self.assistance=False
        self.refresh()

    @classmethod
    def create(cls,root,participant,model_sha,repeats=5):
        from research.live_model import DEFAULT_MODEL,DEFAULT_BASELINE_MODEL
        participant=profile(**participant)
        plan=tasks(repeats)
        examples=json.loads(EXAMPLES.read_text(encoding='utf-8'))['examples']
        if set(examples)!=set(LABELS):raise ValueError('An example is required for all 26 letters.')
        frozen={str(p.resolve()):sha256(p.read_bytes()).hexdigest() for p in [DEFAULT_BASELINE_MODEL,DEFAULT_MODEL]}
        session_id='alphabet_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
        directory=Path(root)/session_id;directory.mkdir(parents=True,exist_ok=False)
        atomic_json(directory/'session.json',{'schema':SCHEMA,'id':session_id,
            'role':'development' if participant['kind']=='development' else 'evaluation',
            'participant':participant,'display':{'preview_mirrored':False,'example_hand':'right'},'repeats':repeats,'tasks':plan,'examples':examples,
            'model_sha256':model_sha,'frozen_models':frozen,
            'instructions':'Example-guided static A-Z production; same hand/camera/lighting/distance. One attempt per planned task including failed detections. Predictions hidden. H marks operator guidance. Never retry based on model outcomes. J/Z are static dataset shapes, not evaluated movement.',
            'guidance_policy':'Standard instructions and practice before collection. During collection only standardized framing reminders; any extra gesture correction must be marked as assistance. No model-driven coaching.',
            'example_scope':'Original dataset-label examples, not independently ASL-validated. Review examples before collecting a final test; changing them requires a new protocol.',
            'purpose':'Paired frozen Shin/angle comparison; never automatically used for training.'})
        return cls(directory,model_sha)

    def refresh(self):
        super().refresh();self.assistance=False
        for directory in self.completed.values():
            record=json.loads((Path(directory)/'capture.json').read_text(encoding='utf-8'))
            metadata=record['protocol']
            if metadata.get('participant')!=self.plan['participant'] or metadata.get('example')!=self.plan['examples'][record['expected_letter']]:
                raise ValueError('Participant or example differs from the fixed capture plan.')

    def metadata(self):
        return {**super().metadata(),'schema':SCHEMA,'participant':self.plan['participant'],
                'operator_guidance':self.assistance,'example':self.plan['examples'][self.current['letter']]}

    def toggle_guidance(self):self.assistance=not self.assistance

    def render_panel(self,preview,prompt=None,notice=None):
        t=self.current;p=self.plan['participant'];panel=np.zeros((752,960,3),np.uint8);panel[:480,:640]=preview
        cv2.putText(panel,'EXAMPLE',(672,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        if t:
            image=self.example_images[t['letter']];h,w=image.shape[:2];scale=min(280/w,310/h)
            im=cv2.resize(image,(round(w*scale),round(h*scale)));x=660+(280-im.shape[1])//2;y=55+(310-im.shape[0])//2
            panel[y:y+im.shape[0],x:x+im.shape[1]]=im
            cv2.putText(panel,f"Letter {t['letter']}",(675,398),cv2.FONT_HERSHEY_SIMPLEX,1,(120,240,240),2,cv2.LINE_AA)
            cv2.putText(panel,'Right hand - dataset' if t and self.plan['examples'][t['letter']].get('display_hand')=='right' else 'Dataset example (old session)',(660,428),cv2.FONT_HERSHEY_SIMPLEX,.49,(220,220,220),1,cv2.LINE_AA)
            cv2.putText(panel,'Unmirrored camera image' if not self.mirror_preview else 'No prediction',(660,453),cv2.FONT_HERSHEY_SIMPLEX,.49,(220,220,220),1,cv2.LINE_AA)
        cv2.rectangle(panel,(12,488),(628,532),(235,235,235),-1)
        cv2.putText(panel,prompt or 'Take photo [Space] - save immediately',(24,516),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,0),1,cv2.LINE_AA)
        if t:
            cv2.putText(panel,f"{t['letter']} | Round {t['repeat']}/{self.plan['repeats']} | Photo {len(self.completed)+1}/{len(self.plan['tasks'])}",(12,572),cv2.FONT_HERSHEY_SIMPLEX,.8,(120,240,240),1,cv2.LINE_AA)
        cv2.putText(panel,notice or 'Relax your hand after each photo and form the next sign.',(12,614),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(panel,f"Assistance provided: {'YES' if self.assistance else 'no'} [H] | Prediction hidden",(12,651),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        hint='J/Z: static dataset shape only, no movement.' if t and t['letter'] in 'JZ' else 'Space: capture | H: mark assistance | Q/Esc: pause'
        cv2.putText(panel,hint,(12,730),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1,cv2.LINE_AA)
        # Pillow supports names with accents; cv2's built-in font is ASCII-only.
        pil=Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
        try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
        except OSError:font=ImageFont.load_default()
        label='Development participant' if p['kind']=='development' else 'New test participant'
        draw.text((12,681),f"{p['name'][:55]} | {label} | {'right' if p['hand']=='right' else 'left'}",font=font,fill='white')
        return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
