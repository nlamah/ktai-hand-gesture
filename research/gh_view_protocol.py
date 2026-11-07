"""Prospective paired G/H viewpoint collection: 40 attempts per participant."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from research.data import atomic_json
from research.protocol import ProtocolSession
from research.gh_capture_guidance import render as render_guidance, VERSION as GUIDANCE_VERSION

VIEWS=('handpalm','schuin_palm','schuin_handrug','handrug')
VIEW_LABELS=dict(zip(VIEWS,('Palm','Oblique palm','Oblique back','Back of hand')))
SCHEMA='asl-gh-viewpoint-protocol-v1'


def tasks(participants=3,hands=None):
    if participants not in (1,2,3):raise ValueError('Choose one, two or three participants.')
    hands=hands or ['right']*participants
    if len(hands)!=participants or set(hands)-{'left','right'}:raise ValueError('Specify left or right for each participant.')
    rows=[]
    for person in range(participants):
        for repeat in range(1,6):
            offset=(person+repeat-1)%len(VIEWS)
            views=VIEWS[offset:]+VIEWS[:offset]
            for view in views:
                for letter in ('GH' if (person+repeat)%2 else 'HG'):
                    rows.append({'id':f'P{person+1:02d}_{letter}_{view}_{repeat}',
                        'participant':f'P{person+1:02d}','letter':letter,'view':view,'repeat':repeat,
                        'requested_hand':hands[person],'split':'development' if person==0 else 'evaluation'})
    return rows


class GHViewSession(ProtocolSession):
    def __init__(self,directory,model_sha):
        self.directory=Path(directory)
        if (self.directory/'withdrawn.json').exists():
            raise ValueError('Deze G/H-sessie is ingetrokken. Start een nieuwe opnameronde.')
        self.plan=json.loads((self.directory/'session.json').read_text())
        if self.plan.get('schema')!=SCHEMA or self.plan.get('tasks')!=tasks(self.plan['participants'],self.plan['hands']):
            raise ValueError('Unknown or modified G/H viewpoint protocol.')
        if self.plan['model_sha256']!=model_sha:raise ValueError('The capture model has changed.')
        for path,expected in self.plan['frozen_models'].items():
            if not Path(path).is_file() or sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'A frozen comparison model has changed or is missing: {path}')
        self.participant_confirmed=None
        self.refresh()

    @classmethod
    def create(cls,root,model_sha,participants=3,hands=None):
        from research.model_choices import MODEL_CHOICES
        hands=hands or ['right']*participants;plan=tasks(participants,hands)
        frozen={str(path.resolve()):sha256(path.read_bytes()).hexdigest() for _,_,path in MODEL_CHOICES if path.is_file()}
        for path in sorted(Path('models/gh_diagnostic').glob('*.joblib')):
            frozen[str(path.resolve())]=sha256(path.read_bytes()).hexdigest()
        session_id='GH_views_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
        directory=Path(root)/session_id;directory.mkdir(parents=True,exist_ok=False)
        atomic_json(directory/'session.json',{'schema':SCHEMA,'id':session_id,'role':'viewpoint_study',
            'participants':participants,'hands':hands,'tasks':plan,'model_sha256':model_sha,'frozen_models':frozen,
            'purpose':'Paired viewpoint comparison; never automatically added to training.',
            'viewpoint_instruction_version':'explicit-gh-pose-v3',
            'guidance_display_version':GUIDANCE_VERSION,
            'instructions':'G: extend index and thumb approximately parallel, bend the other three fingers. H: extend index and middle together, bend ring and little fingers and fold thumb. Then select the view: palm faces camera; oblique palm about 45 degrees from palm view; oblique back about 45 degrees from back view; back faces camera. Keep the same hand, intrinsic finger shape, distance and lighting. Finger direction on screen may change naturally with the turn: do not force a fixed left/right pointing direction. The diagram arrow denotes the palm side, not finger direction. Angles are qualitative, not measured. Reform the sign on each attempt. Save every planned attempt including detection failures.',
            'evaluation_policy':'P01 is development; other participants are reserved evaluation. Keep models and analysis choices fixed before inspecting reserved results. Do not retake based on prediction.'})
        return cls(directory,model_sha)

    def label(self):
        t=self.current
        if t is None:return 'G/H capture session complete.'
        return f"{len(self.completed)+1}/{len(self.plan['tasks'])}: {t['participant']} | {t['letter']} | {VIEW_LABELS[t['view']]} | {t['repeat']}/5"

    def metadata(self):
        return {**super().metadata(),'schema':SCHEMA,'viewpoints_qualitative':True,
                'viewpoint_instruction_version':self.plan.get('viewpoint_instruction_version','preserve-direction-v1'),
                'guidance_display_version':GUIDANCE_VERSION}

    def render_panel(self,preview,prompt=None,notice=None):
        return render_guidance(preview,self.current,len(self.completed),len(self.plan['tasks']),prompt,notice)
