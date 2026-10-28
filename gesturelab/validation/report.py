"""Explicit evidence, unknowns and limitations for each audit assertion."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def new_output(path, protected=()):
    path = Path(path).resolve()
    if any(path.is_relative_to(Path(root).resolve()) for root in protected):
        raise ValueError('Audit output must be outside the source dataset and experiment directories.')
    path.mkdir(parents=True, exist_ok=False)
    return path


class AuditReport:
    def __init__(self, title):
        self.title, self.checks = title, []

    def add(self, name, method, expected, observed, condition=None, files=(), limitation='', next_step=''):
        status = 'not_checked' if condition is None else ('passed' if condition else 'failed')
        self.checks.append(dict(name=name, status=status, method=method, files=list(map(str, files)),
            expected=expected, observed=observed, limitation=limitation, next_step=next_step))

    def save(self, out, **metadata):
        result = dict(title=self.title, created_at=datetime.now(timezone.utc).isoformat(),
            statuses=dict(Counter(c['status'] for c in self.checks)), checks=self.checks,
            qualification='Technical evidence only; unknowns are not passes. Reproducibility does not establish scientific validity.',
            ai_use='Codex implemented this workflow and ran the reported available checks. not_checked entries are not claimed as performed. No independent human review or user-performed checks have been confirmed.',
            **metadata)
        write_json(Path(out) / 'audit.json', result)
        Path(out, 'AUDIT.md').write_text(self.markdown(result), encoding='utf-8')
        return result

    def markdown(self, result):
        lines = [f'# {self.title}', '', result['qualification'], '', result['ai_use'], '',
                 'Status counts: ' + json.dumps(result['statuses']), '']
        for check in self.checks:
            lines.extend([f'## {check["name"]} — {check["status"]}', ''])
            for key in ('method', 'files', 'expected', 'observed', 'limitation', 'next_step'):
                value = check[key]
                value = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
                lines.append(f'**{key.replace("_", " ").title()}:** {value or "None recorded."}\n')
        lines.extend(['## Human review still required', '',
            '- [ ] Recalculate a distance, a 90°/180° joint angle and a signed axis cosine by hand.',
            '- [ ] Read the Shin mapping and approve or challenge its normalization/coordinate assumptions.',
            '- [ ] Trace ID selection → features → fold-only fit → validation transform → prediction.',
            '- [ ] Compare independently recomputed CSV scores and investigate every failed/unknown check.',
            '- [ ] Assess group meaning, near-duplicates and whether the research conclusions exceed the evidence.',
            '- [ ] Record your own actions and dates; these unchecked items do not assert your review.'])
        return '\n'.join(lines) + '\n'


class ReportScope:
    def __init__(self, parent, prefix):
        self.parent, self.prefix = parent, prefix

    def add(self, name, *args, **kwargs):
        self.parent.add(self.prefix + ' / ' + name, *args, **kwargs)
