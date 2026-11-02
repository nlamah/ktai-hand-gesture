"""Fixed C/D/O tilt collection; roles stay separate from training data."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from research.data import atomic_json

TILTS = ('neutraal', 'pols_omhoog', 'pols_omlaag')


def tasks():
    return [{'id': f'{letter}_{tilt}_{repeat}', 'letter': letter, 'tilt': tilt, 'repeat': repeat}
            for repeat in range(1, 4) for tilt in TILTS for letter in 'CDO']


class ProtocolSession:
    def __init__(self, directory, model_sha):
        self.directory = Path(directory)
        self.plan = json.loads((self.directory / 'session.json').read_text())
        if self.plan.get('schema') != 'asl-tilt-protocol-v1' or self.plan.get('tasks') != tasks():
            raise ValueError('Onbekend of gewijzigd meetprotocol.')
        if self.plan['model_sha256'] != model_sha:
            raise ValueError('Deze sessie gebruikt een ander model; start een nieuwe ronde.')
        self.refresh()

    @classmethod
    def create(cls, root, role, model_sha):
        if role not in ('development', 'evaluation'):
            raise ValueError('Ongeldige sessierol.')
        session_id = role + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        directory = Path(root) / session_id
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'session.json', {'schema': 'asl-tilt-protocol-v1', 'id': session_id,
                    'role': role, 'model_sha256': model_sha, 'tasks': tasks(),
                    'purpose': 'Prospective live comparison; never automatically used for training.',
                    'instructions': 'Keep finger shape and camera position fixed; tilt at wrist only. Capture every planned attempt, regardless of prediction. Angles are qualitative, not calibrated degrees.'})
        return cls(directory, model_sha)

    def refresh(self):
        self.completed = {}
        taskmap = {t['id']: t for t in self.plan['tasks']}
        for path in sorted(self.directory.glob('*/capture.json')):
            if path.parent.name.startswith('.'):
                continue
            record = json.loads(path.read_text())
            p = record.get('protocol') or {}
            if not isinstance(p, dict):
                raise ValueError('Ongeldige protocolmetadata in opname.')
            if p.get('session_id') != self.plan['id'] or p.get('role') != self.plan['role']:
                raise ValueError('Een opname hoort niet bij deze sessie.')
            task = taskmap.get(p.get('task', {}).get('id'))
            if task is None or p['task'] != task or record['expected_letter'] != task['letter']:
                raise ValueError('Een opname heeft een onjuiste taak of annotatie.')
            if record['model']['sha256'] != self.plan['model_sha256']:
                raise ValueError('Een opname gebruikt een ander model.')
            if task['id'] in self.completed:
                raise ValueError('Dubbele protocolopname gevonden; controleer de sessiemap.')
            self.completed[task['id']] = str(path.parent)

    @property
    def current(self):
        return next((t for t in self.plan['tasks'] if t['id'] not in self.completed), None)

    def metadata(self):
        return {'session_id': self.plan['id'], 'role': self.plan['role'], 'task': self.current,
                'prediction_hidden': True, 'not_training_data': True}

    def label(self):
        t = self.current
        if t is None:
            return 'Meetronde voltooid.'
        role = "Ontwikkeling" if self.plan["role"] == "development" else "Evaluatie"
        return f"{role} {len(self.completed)+1}/27: {t['letter']} | {t['tilt'].replace('_',' ')} | herhaling {t['repeat']}/3"
