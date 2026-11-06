"""Independently recompute metrics and replay guarded G/H development models."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from research.data import atomic_json
from research.gh_view_adapter import horizontal_mask
from research.inputs import LABELS
from research.live_model import load_bundle, predict_letter, detection_inputs
from research.studies.mirror_augmentation import load_extracted, assignments_for, balanced_views, VIEWS


def run(study):
    out = Path('results') / study
    work = Path('runs') / study
    protocol = json.loads((out/'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == expected, path
    assert sha256(Path(protocol['base_model']).read_bytes()).hexdigest() == protocol['base_sha256']
    assert sha256(Path('results/mirror_01/protocol.json').read_bytes()).hexdigest() == protocol['mirror_protocol_sha256']
    vertical = study == 'gh_vertical_01'
    model_path = Path('models') / ('asl_gh_vertical_adapter.joblib' if vertical else 'asl_gh_back_adapter.joblib')
    expected_model = json.loads((out/'model.json').read_text())['model_sha256']
    assert sha256(model_path.read_bytes()).hexdigest() == expected_model
    bundle = load_bundle(model_path)
    assert bundle['gh_view_adaptation'] == protocol
    for source in protocol['calibration_sources']:
        folder = Path('captures/confusions') / source['id']
        assert sha256((folder/'capture.json').read_bytes()).hexdigest() == source['capture_json_sha256']
        assert sha256((folder/'model_input.png').read_bytes()).hexdigest() == source['image_sha256']
        assert source['role'] == 'calibration_training'
    with np.load(work/'predictions.npz') as archive:
        d = {key: archive[key] for key in archive.files}
    X, y, ids, groups, status, _ = load_extracted(Path('runs/mirror_01'), Path('results/mirror_01'))
    frozen_folds = json.loads(Path('data/asl_kaggle/research/summary.json').read_text())['folds']
    assignment = assignments_for(groups, frozen_folds)
    for actual, expected in ((d['ids'], ids), (d['labels'], y), (d['fold'], assignment),
                             (d['status'], status), (d['selected_view'], balanced_views(y, assignment))):
        np.testing.assert_array_equal(actual, expected)
    assert not set(ids) & {s['id'] for s in protocol['calibration_sources']}
    reference = pd.read_csv('results/mirror_01/predictions.csv', keep_default_na=False)
    for view, name in enumerate(VIEWS):
        old = reference[(reference.model == 'svm') & (reference.view == name)].set_index('id').loc[ids]
        np.testing.assert_array_equal(old.mirror_training, d['baseline'][:, view])
    changed = d['baseline'] != d['adapted']
    assert np.isin(d['adapted'][changed], ['G', 'H']).all()
    for view in range(2):
        eligible = np.zeros(len(y), bool)
        accepted = status[:,view] == 'ok'
        eligible[accepted] = horizontal_mask(X[accepted,view])
        assert not changed[~eligible,view].any()
        assert (d['adapted'][~accepted,view] == '?').all()
    metric_rows = pd.read_csv(out/'metrics.csv', keep_default_na=False)
    letter_rows = pd.read_csv(out/'per_letter.csv', keep_default_na=False)
    paired_rows = pd.read_csv(out/'paired_changes.csv', keep_default_na=False)
    metric_checks = letter_checks = paired_checks = 0
    presentations = {}
    for presentation in (*VIEWS, 'balanced_mix'):
        view = np.full(len(y), VIEWS.index(presentation)) if presentation in VIEWS else d['selected_view']
        ix = np.arange(len(y))
        accepted = status[ix,view] == 'ok'
        a, b = d['baseline'][ix,view], d['adapted'][ix,view]
        presentations[presentation] = (a,b,accepted)
        for scope in ('all_attempts','accepted_only'):
            mask = np.ones(len(y),bool) if scope == 'all_attempts' else accepted
            for fold in ('pooled', *map(str, sorted(set(assignment)))):
                keep = mask if fold == 'pooled' else mask & (assignment == int(fold))
                for name, pred in (('baseline',a),('adapted',b)):
                    p,r,f,s = precision_recall_fscore_support(y[keep],pred[keep],labels=LABELS,zero_division=0)
                    row = metric_rows[(metric_rows.presentation==presentation)&(metric_rows.scope==scope)&
                                      (metric_rows.fold==fold)&(metric_rows.model==name)].iloc[0]
                    values = [keep.sum(), (pred[keep]==y[keep]).sum(), (pred[keep]==y[keep]).mean(),p.mean(),r.mean(),f.mean()]
                    np.testing.assert_allclose(row[['images','correct','accuracy','macro_precision','macro_recall','macro_f1']].to_numpy(float),values,rtol=0,atol=1e-12)
                    metric_checks += 1
                    if fold == 'pooled':
                        for j,letter in enumerate(LABELS):
                            detail = letter_rows[(letter_rows.presentation==presentation)&(letter_rows.scope==scope)&
                                                 (letter_rows.model==name)&(letter_rows.letter==letter)].iloc[0]
                            np.testing.assert_allclose(detail[['images','precision','recall','f1']].to_numpy(float),[s[j],p[j],r[j],f[j]],rtol=0,atol=1e-12)
                            letter_checks += 1
                row = paired_rows[(paired_rows.presentation==presentation)&(paired_rows.scope==scope)&(paired_rows.fold==fold)].iloc[0]
                fixed = ((a[keep]!=y[keep])&(b[keep]==y[keep])).sum()
                lost = ((a[keep]==y[keep])&(b[keep]!=y[keep])).sum()
                np.testing.assert_array_equal(row[['fixed','lost','net']].to_numpy(int),[fixed,lost,fixed-lost])
                paired_checks += 1
    replayed = 0
    for fold in sorted(set(assignment)):
        assert not set(groups[assignment==fold]) & set(groups[assignment!=fold])
        candidate = joblib.load(work/f'fold{fold}.joblib')
        baseline = joblib.load(Path('runs/mirror_01')/f'svm_mirrored_fold{fold}.joblib')
        assert candidate[-1].adapted.gamma == baseline[-1].gamma
        assert candidate[-1].adapted.class_weight == baseline[-1].class_weight
        for view in range(2):
            sample = []
            for letter in LABELS:
                indices = np.flatnonzero((assignment==fold)&(status[:,view]=='ok')&(y==letter))
                sample.append(indices[len(indices)//2])
            rows = X[sample,view]
            np.testing.assert_allclose(candidate[0].transform(rows),baseline[0].transform(rows),rtol=0,atol=0)
            np.testing.assert_array_equal(candidate.predict(rows),d['adapted'][sample,view])
            np.testing.assert_array_equal(baseline.predict(rows),d['baseline'][sample,view])
            replayed += len(sample)
    captures = pd.read_csv(out/'capture_replay.csv', keep_default_na=False)
    base_bundle = load_bundle(Path(protocol['base_model']))
    for row in captures.itertuples():
        record = json.loads((Path('captures/confusions')/row.id/'capture.json').read_text())
        assert predict_letter(bundle,record['detection'])['letter'] == row.adapted
        assert predict_letter(base_bundle,record['detection'])['letter'] == row.baseline
    # Also verify the current cached RGB detections, including reflected calibration views.
    with np.load(work/'calibration.npz') as archive:
        calibration = {key:archive[key] for key in archive.files}
    records = json.loads((work/'calibration_records.json').read_text())
    calibration_reencoding = 0
    for i,record in enumerate(records):
        for view,entry in enumerate(record['views']):
            if calibration['status'][i,view] == 'ok':
                np.testing.assert_allclose(detection_inputs(entry['detection'])[0],calibration['features'][i,view],rtol=0,atol=0)
                calibration_reencoding += 1
    verification = {'status':'passed','metric_rows':metric_checks,'per_letter_rows':letter_checks,
                    'paired_rows':paired_checks,'fold_replay_predictions_per_arm':replayed,
                    'capture_replays':len(captures),'calibration_reencodings':calibration_reencoding,
                    'model_sha256':expected_model,'predictions_sha256':sha256((work/'predictions.npz').read_bytes()).hexdigest(),
                    'note':'Calibration replay is training fit. Source-group CV does not measure unseen back-of-hand accuracy.'}
    atomic_json(out/'verification.json',verification)
    title = 'Camera-y180 G/H adaptation' if vertical else 'Index-axis G/H roll adaptation'
    lines = [f'# {title}', '', 'Development experiment using existing Kaggle data and three already saved back-of-hand images. '
             'Those three images are calibration **training**, not held-out webcam evidence.', '',
             f"Transformation: {protocol['roll']}.", '',
             'The matched reference is the mirror-trained 26-letter SVM. Original source folds and the fixed 50/50 '
             'presentation mix are unchanged. G/H-only rotation copies preserve total source weight; '
             'each of the three calibration sources has total weight 20. Both arms use the same preprocessing, gamma and class weights. '
             'New G/H predictions require the fixed horizontal-index gate. All other proposals retain the reference prediction.', '',
             '## All 26 letters', '', '| Presentation | Scope | Reference precision | Candidate precision | Reference F1 | Candidate F1 | Fixed / lost |',
             '|---|---|---:|---:|---:|---:|---:|']
    for presentation in presentations:
        for scope in ('all_attempts','accepted_only'):
            rows=metric_rows[(metric_rows.presentation==presentation)&(metric_rows.scope==scope)&(metric_rows.fold=='pooled')].set_index('model')
            pair=paired_rows[(paired_rows.presentation==presentation)&(paired_rows.scope==scope)&(paired_rows.fold=='pooled')].iloc[0]
            a,b=rows.loc['baseline'],rows.loc['adapted']
            lines.append(f'| {presentation} | {scope} | {a.macro_precision:.6f} | {b.macro_precision:.6f} | {a.macro_f1:.6f} | {b.macro_f1:.6f} | {pair.fixed} / {pair.lost} |')
    lines += ['', 'Precision and F1 are macro averages over A–Z. All-attempt scores count failed detections as missed attempts; '
              'accepted-only scores condition on a valid detection. These are the dataset’s usual viewpoints, not a new back-of-hand test.', '',
              '## G and H on the balanced presentation mix', '', '| Letter | Model | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for row in letter_rows[(letter_rows.presentation=='balanced_mix')&(letter_rows.scope=='all_attempts')&letter_rows.letter.isin(['G','H'])].itertuples():
        lines.append(f'| {row.letter} | {row.model} | {row.precision:.6f} | {row.recall:.6f} | {row.f1:.6f} |')
    a,b,_=presentations['balanced_mix'];other=~np.isin(y,['G','H'])
    lines += ['', f'Non-G/H attempts predicted as G/H: {np.isin(a[other],["G","H"]).sum()} reference, '
              f'{np.isin(b[other],["G","H"]).sum()} candidate. '
              'A horizontal-pose gate does not eliminate false G/H proposals.', '', '## Previously saved camera images', '',
              '| Intended | Reference | Candidate | Role |', '|---|---|---|---|']
    for row in captures.itertuples():
        lines.append(f'| {row.intended} | {row.baseline} | {row.adapted} | {row.role} |')
    if vertical and (out/'rotation_only_check.json').exists():
        ablation=json.loads((out/'rotation_only_check.json').read_text())
        assert ablation['calibration_sources']==0
        assert sha256(Path(ablation['model_path']).read_bytes()).hexdigest()==ablation['model_sha256']
        for path,expected in ablation['source_sha256'].items():
            assert sha256(Path(path).read_bytes()).hexdigest()==expected
        rotation_only=load_bundle(Path(ablation['model_path']))
        rotation_rows=pd.read_csv(out/'rotation_only_capture_replay.csv',keep_default_na=False)
        for row in rotation_rows.itertuples():
            record=json.loads((Path('captures/confusions')/row.id/'capture.json').read_text())
            assert predict_letter(rotation_only,record['detection'])['letter']==row.rotation_only
        verification['rotation_only_capture_replays']=len(rotation_rows)
        atomic_json(out/'verification.json',verification)
        lines += ['', '## Rotation without camera calibration', '',
                  'A separate full-data fit excludes all camera captures and adds only the G/H y180 copies to '
                  'the mirror-trained source data. This isolates calibration from rotation on the known selected cases, '
                  'not on an independent test set. This arm has no source-data cross-validation.', '',
                  '| Intended | Reference | Rotation only | Rotation + calibration |', '|---|---|---|---|']
        combined=captures.set_index('id')
        for row in rotation_rows[rotation_rows.intended.isin(['G','H'])].itertuples():
            lines.append(f'| {row.intended} | {row.baseline} | {row.rotation_only} | {combined.loc[row.id,"adapted"]} |')
    lines += ['', 'The first three back-view G/H cases used for calibration cannot validate this model. '
              'All remaining cases were previously inspected and selected for errors; they are development controls, not a representative test. '
              'Rotation and calibration are combined here; this comparison does not isolate their contributions. '
              'The earlier G/H rotation candidate also used ±90° rotations; this candidate uses only 180°, a mirror-trained reference and explicit calibration.', '',
              '## Checks and limits', '',
              f'Verified {metric_checks} aggregate rows, {letter_checks} per-letter rows, {paired_checks} paired rows, '
              f'{replayed} held-out predictions per arm, {len(captures)} capture replays and {calibration_reencoding} calibration re-encodings. '
              'Source/model/image hashes, fixed folds, unchanged preprocessing and restricted routing pass. See `verification.json`.', '',
              'The rotation acts on normalized image-landmark geometry. It cannot reconstruct hidden fingers or improve MediaPipe detections. '
              'The original source blocks are not verified signer identities. No independent back-view accuracy or general robustness is claimed. '
              'This remains an optional live development model.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(verification,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('gh_view_adaptation_01','gh_vertical_01'),default='gh_view_adaptation_01')
    run(parser.parse_args().study)


if __name__ == '__main__': main()
