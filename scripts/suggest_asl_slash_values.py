"""Suggest bounded slash relaxations using fitting groups, with frozen thresholds."""
import copy
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from gesturelab.features.asl_excel_review import load_reference_workbook, load_landmark_cache, categorize_workbook_features, measure_hand_features
from gesturelab.features.asl_threshold_tuning import _grouped_split, _source_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/asl_slash_proposal_20261031'
LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
FLEXION = ['bent', 'partly_bent', 'extended']
NAMES = {'bent': 'gebogen', 'partly_bent': 'licht gebogen', 'extended': 'gestrekt'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'assets/asl_feature_references.xlsx'
    refs = load_reference_workbook(source.read_bytes())
    params = json.loads((ROOT / 'config/asl_landmark_thresholds_tuned.json').read_text())
    cache = load_landmark_cache(ROOT / 'data/reports/asl_letters_v1_train_landmarks.json',
                                ROOT / 'data/external/asl_letters_v1/train')
    if cache is None:
        raise ValueError('Stale cache')
    samples = cache['samples']
    fitting, holdout, *_ = _grouped_split(samples)
    fit_groups = {_source_group(sample) for sample in fitting}
    fit = np.array([_source_group(sample) in fit_groups for sample in samples])
    actual = [categorize_workbook_features(measure_hand_features(sample['world_landmarks'], sample.get('handedness')), params)
              if sample.get('status') == 'ok' and sample.get('world_landmarks') is not None else {}
              for sample in samples]
    for row in actual:
        for key in list(row):
            if key.startswith('thumb_tip_') and row[key] in ('near', 'far'):
                row[key] = 'no_contact'
    features = list(refs['A'])
    def masks(references):
        return np.array([[[cell['allowed'] is None or actual[i].get(key, 'unknown') in cell['allowed']
                           for key, cell in references[label].items()]
                          for label in LABELS] for i in range(len(samples))])
    original = masks(refs)
    proposed = copy.deepcopy(refs)
    changes = []
    for label in LABELS:
        own = np.array([sample['label'] == label for sample in samples])
        ownfit = own & fit
        otherfit = ~own & fit
        used = set()
        for _ in range(3):
            current = masks(proposed)[:, LABELS.index(label), :]
            choices = []
            for j, feature in enumerate(features):
                # Protect MCP positions, thumb B, contacts, spread and crossing.
                if not feature.startswith('flexion_') or feature.endswith('_mcp') or feature in used:
                    continue
                allowed = proposed[label][feature]['allowed']
                if allowed is None or len(allowed) != 1:
                    continue
                category = allowed[0]
                for extra in FLEXION:
                    if abs(FLEXION.index(extra) - FLEXION.index(category)) != 1:
                        continue
                    accepts = np.array([row.get(feature) == extra for row in actual])
                    improvement = accepts & ~current[:, j]
                    gain = int((improvement & ownfit).sum())
                    if gain < max(3, int(ownfit.sum() * .15)):
                        continue
                    candidate = current.copy()
                    candidate[:, j] |= accepts
                    old_full, new_full = current.all(axis=1), candidate.all(axis=1)
                    own_gain = int((~old_full & new_full & ownfit).sum())
                    wrong_gain = int((~old_full & new_full & otherfit).sum())
                    if wrong_gain > own_gain:
                        continue
                    # Prefer recovering own full images, then large own cell gains
                    # with a low increase in this feature's acceptance on other letters.
                    foreign_rate = float((improvement & otherfit).sum() / otherfit.sum())
                    own_rate = gain / ownfit.sum()
                    score = (own_gain - wrong_gain, own_rate - .5 * foreign_rate, own_rate)
                    choices.append((score, feature, extra, own_gain, wrong_gain, gain, own_rate, foreign_rate))
            if not choices:
                break
            _, feature, extra, own_gain, wrong_gain, gain, own_rate, foreign_rate = max(choices)
            cell = proposed[label][feature]
            old = cell['raw']
            cell['allowed'] = sorted(cell['allowed'] + [extra])
            cell['raw'] = '/'.join(NAMES[value] for value in FLEXION if value in cell['allowed'])
            used.add(feature)
            changes.append({'letter': label, 'feature': feature, 'before': old, 'after': cell['raw'],
                            'fit_added_feature_matches': gain, 'fit_own_images_added': own_gain,
                            'fit_wrong_letter_acceptances_added': wrong_gain,
                            'fit_own_feature_gain_fraction': own_rate,
                            'fit_other_feature_gain_fraction': foreign_rate})
    relaxed = masks(proposed)
    def metrics(matrix, subset):
        indices = np.flatnonzero(subset)
        ownidx = np.array([LABELS.index(sample['label']) for sample in samples])
        full = matrix.all(axis=2)
        correct = full[np.arange(len(samples)), ownidx]
        wrong = full.copy()
        wrong[np.arange(len(samples)), ownidx] = False
        unique = correct & ~wrong.any(axis=1)
        known = np.array([sample.get('status') == 'ok' for sample in samples])
        matching_features = total_features = 0
        for i in indices:
            for j, feature in enumerate(features):
                if refs[samples[i]['label']][feature]['allowed'] is not None:
                    total_features += 1
                    matching_features += int(matrix[i, ownidx[i], j])
        return {'images':len(indices), 'detected':int((known & subset).sum()),
                'correct_letter_full_matches':int(correct[subset].sum()),
                'correct_letter_full_fraction':float(correct[subset].mean()),
                'unique_correct_full_matches':int(unique[subset].sum()),
                'wrong_letter_full_acceptances':int(wrong[subset].sum()),
                'images_with_any_wrong_full_match':int(wrong[subset].any(axis=1).sum()),
                'feature_match_fraction':matching_features/total_features}
    comparison = {part:{'before':metrics(original, subset),'proposal':metrics(relaxed,subset)}
                  for part,subset in [('all',np.ones(len(samples),dtype=bool)),('fit',fit),('holdout',~fit)]}
    perletter = []
    for label in LABELS:
        subset=np.array([sample['label']==label for sample in samples])
        perletter.append({'letter':label,'before':metrics(original,subset),'proposal':metrics(relaxed,subset)})
    workbook = load_workbook(source,read_only=True,data_only=True)
    rows = [list(row[:20]) for row in workbook.active.values if row[0] is not None or any(row[1:20])][:27]
    # First header cell can be blank. Preserve importer-compatible first sheet.
    if rows[0][0] is None: rows[0][0]='Letter'
    from gesturelab.features.asl_excel_review import REFERENCE_COLUMNS, _normalize
    for row in rows[1:]:
        for column,header in enumerate(rows[0][1:],start=1):
            feature=REFERENCE_COLUMNS[_normalize(header)]
            if any(change['letter']==row[0] and change['feature']==feature for change in changes):
                row[column]=proposed[row[0]][feature]['raw']
    payload={'rows':rows,'changes':changes,'comparison':comparison,'per_letter':perletter,
             'parameters':params, 'source':str(source),
             'method':'Maximaal 3 verruimde buigfeatures per letter; één aangrenzende categorie extra. '
             'MCP, duim B/niet-B, spreiding, kruising en contact blijven beschermd. '
             'Selectie uitsluitend op afstelgroepen; grenzen vast. Geen kan en geen drie-categoriecellen toegevoegd. '
             'Per stap niet meer extra volledige verkeerde matches dan extra eigen matches op afsteldata. '
             'Holdout is eerder gebruikte ontwikkeldata. Geen testbeelden gebruikt.'}
    (OUT/'proposal_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'changes':len(changes),'comparison':comparison},indent=2))


if __name__=='__main__':main()
