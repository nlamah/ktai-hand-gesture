"""Audit the isolated y180 comparison and generate its report and paper tables."""
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from research.data import atomic_json
from research.inputs import LABELS, write_csv
from research.live_model import DEFAULT_MODEL, load_bundle, predict_letter
from research.studies.gh_rotation_standalone import OUT, WORK, MODEL, CACHE, ARMS, inputs, augmented_rows, frozen_protocol


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol==frozen_protocol()
    assert not protocol['mirror_augmentation'] and protocol['camera_calibration_sources']==0 and not protocol['routing_gate']
    X,y,ids,splits=inputs();predictions=pd.read_csv(OUT/'predictions.csv',keep_default_na=False)
    np.testing.assert_array_equal(predictions.id,ids);np.testing.assert_array_equal(predictions.label,y)
    assert len(set(ids))==60251
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    groups=np.array([r['group'] for r in meta['samples']]);np.testing.assert_array_equal(predictions.group,groups)
    old=pd.read_csv('results/angles/oof_predictions_svm.csv',keep_default_na=False)
    old=old[old.variant=='shin_angles'].set_index('id').loc[ids]
    np.testing.assert_array_equal(predictions.original,old.prediction)
    np.testing.assert_array_equal(predictions.fold,old.fold)
    tables=pd.read_csv(OUT/'metrics.csv',keep_default_na=False)
    letters=pd.read_csv(OUT/'per_letter.csv',keep_default_na=False)
    paired=pd.read_csv(OUT/'paired_changes.csv',keep_default_na=False)
    metric_checks=letter_checks=paired_checks=replays=0
    for fold in ('pooled',*map(str,predictions.fold.unique())):
        keep=np.ones(len(y),bool) if fold=='pooled' else predictions.fold.to_numpy()==int(fold)
        for arm in ARMS:
            pred=predictions[arm].to_numpy()[keep]
            p,r,f,s=precision_recall_fscore_support(y[keep],pred,labels=LABELS,zero_division=0)
            row=tables[(tables.model==arm)&(tables.fold==fold)].iloc[0]
            np.testing.assert_allclose(row[['images','correct','accuracy','macro_precision','macro_recall','macro_f1']].to_numpy(float),
                                       [keep.sum(),(pred==y[keep]).sum(),(pred==y[keep]).mean(),p.mean(),r.mean(),f.mean()],rtol=0,atol=1e-12)
            metric_checks+=1
            for j,letter in enumerate(LABELS):
                detail=letters[(letters.model==arm)&(letters.fold==fold)&(letters.letter==letter)].iloc[0]
                np.testing.assert_allclose(detail[['images','precision','recall','f1']].to_numpy(float),[s[j],p[j],r[j],f[j]],rtol=0,atol=1e-12)
                letter_checks+=1
        a=predictions.original.to_numpy()[keep];b=predictions.gh_y180.to_numpy()[keep];truth=y[keep]
        fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
        row=paired[paired.fold==fold].iloc[0]
        np.testing.assert_array_equal(row[['fixed','lost','net','wrong_to_wrong']].to_numpy(int),
                                     [fixed,lost,fixed-lost,((a!=truth)&(b!=truth)&(a!=b)).sum()])
        paired_checks+=1
    for fold,train,test in splits:
        assert not set(groups[train])&set(groups[test])
        np.testing.assert_array_equal(predictions.fold.to_numpy()[test],np.full(len(test),fold))
        baseline=joblib.load(WORK/f'original_fold{fold}.joblib')
        candidate=joblib.load(WORK/f'gh_y180_fold{fold}.joblib')
        assert type(candidate[-1]).__name__=='SVC' and len(candidate)==2
        assert baseline[-1]._gamma==candidate[-1]._gamma
        np.testing.assert_array_equal(baseline[-1].class_weight_,candidate[-1].class_weight_)
        # Reconstruct the original-only preprocessor independently, including its imputation.
        from research.studies.compare_angles import classifier
        expected=classifier('svm','shin_angles',len(train))[0].fit(X[train],y[train])
        selected=np.array([test[np.flatnonzero(y[test]==letter)[len(np.flatnonzero(y[test]==letter))//2]] for letter in LABELS])
        np.testing.assert_array_equal(expected.transform(X[selected]),baseline[0].transform(X[selected]))
        np.testing.assert_array_equal(expected.transform(X[selected]),candidate[0].transform(X[selected]))
        for arm,model in zip(ARMS,(baseline,candidate)):
            np.testing.assert_array_equal(model.predict(X[selected]),predictions[arm].to_numpy()[selected]);replays+=len(selected)
        # Fitting-only duplication and per-source mass: the transformed tail is still invariant.
        rows,targets,weights=augmented_rows(X[train],y[train]);gh=np.isin(y[train],['G','H'])
        assert len(rows)==len(train)+gh.sum() and len(targets)==len(weights)==len(rows)
        np.testing.assert_array_equal(rows[:len(train)],X[train])
        np.testing.assert_array_equal(targets[len(train):],y[train][gh])
        np.testing.assert_array_equal(weights[:len(train)][gh]+weights[len(train):],np.ones(gh.sum()))
        np.testing.assert_array_equal(weights[:len(train)][~gh],np.ones((~gh).sum()))
        np.testing.assert_array_equal(rows[len(train):,:190],X[train][gh,:190])
        np.testing.assert_array_equal(rows[len(train):,820:],X[train][gh,820:])
    model_info=json.loads((OUT/'model.json').read_text())
    assert sha256(MODEL.read_bytes()).hexdigest()==model_info['candidate_sha256']
    assert sha256((WORK/'original_full.joblib').read_bytes()).hexdigest()==model_info['fresh_original_sha256']
    bundle=load_bundle(MODEL);base=load_bundle(DEFAULT_MODEL)
    assert bundle['augmentation']==protocol and type(bundle['estimator'][-1]).__name__=='SVC'
    assert bundle['training_rows']==65052 and model_info['GH_sources']==4801
    np.testing.assert_array_equal(bundle['estimator'][-1].class_weight_,base['estimator'][-1].class_weight_)
    assert bundle['estimator'][-1]._gamma==base['estimator'][-1]._gamma
    captures=pd.read_csv(OUT/'capture_replay.csv',keep_default_na=False)
    for row in captures.itertuples():
        folder=Path('captures/confusions')/row.id;r=json.loads((folder/'capture.json').read_text())
        for name,info in r['files'].items():assert sha256((folder/name).read_bytes()).hexdigest()==info['sha256']
        assert predict_letter(base,r['detection'])['letter']==row.original
        assert predict_letter(bundle,r['detection'])['letter']==row.gh_y180
        assert row.role=='previously_inspected_development_control_not_used_for_fitting'
    for arm in ARMS:
        cm=confusion_matrix(y,predictions[arm],labels=LABELS)
        write_csv(OUT/f'confusion_{arm}.csv',[{'true_letter':letter,**dict(zip(LABELS,map(int,cm[j])))} for j,letter in enumerate(LABELS)])
    summary=tables[tables.fold=='pooled'].set_index('model');a,b=summary.loc['original'],summary.loc['gh_y180']
    change=paired[paired.fold=='pooled'].iloc[0]
    verification={'status':'passed','source_images':len(y),'metric_rows':metric_checks,'per_letter_rows':letter_checks,
                  'paired_rows':paired_checks,'fold_model_replay_predictions':replays,'capture_replays':len(captures),
                  'historical_OOF_baseline_exact_match':True,'training_only_preprocessing_verified':True,
                  'no_mirrors_calibration_or_router':True,'candidate_sha256':model_info['candidate_sha256'],
                  'predictions_sha256':sha256((OUT/'predictions.csv').read_bytes()).hexdigest()}
    atomic_json(OUT/'verification.json',verification)
    lines=['# Standalone G/H y180 augmentation', '',
           'Question: does adding 180-degree camera-y turns to G/H fitting examples improve the 26-letter SVM, '
           'without mirror augmentation, camera calibration or routing rules?', '',
           'Both arms use the same 60,251 historical accepted original images and frozen five filename-block folds. '
           'The source population contains 2,430 G and 2,371 H examples. Only G/H training directions change: '
           '`(dx,dy,dz) -> (-dx,dy,-dz)`. Originals are retained. G/H originals and turns have weight 0.5 each; '
           'other sources have weight one. The full fit has 65,052 rows. Scaling, resolved gamma and balanced class priors '
           'are learned from original fitting rows and shared. Both models are ordinary 26-class SVM pipelines.', '',
           'Validation uses original images only, never transformed fitting siblings. A freshly fitted baseline '
           'exactly reproduces all 60,251 historical world-joint OOF predictions. No source images or camera captures are added.', '',
           '## Pooled five-fold results', '', '| Training | Accuracy (%) | Macro-precision (%) | Macro-F1 | Correct |',
           '|---|---:|---:|---:|---:|']
    for arm,label in [('original','Original only'),('gh_y180','Original + G/H y180')]:
        m=summary.loc[arm];lines.append(f'| {label} | {100*m.accuracy:.5f} | {100*m.macro_precision:.5f} | {m.macro_f1:.8f} | {int(m.correct)} / {len(y)} |')
    lines += ['',f'Paired changes: **{int(change.fixed)} fixed, {int(change.lost)} lost, net {int(change.net):+d}**; '
              f'{int(change.wrong_to_wrong)} other wrong-to-wrong changes. Scores condition on successful historical landmark extraction '
              '(60,251 of 78,000 source images). Macro scores average A–Z; this is not a binary G/H task.', '',
              '| Letter | Reference P | Candidate P | Reference R | Candidate R | Reference F1 | Candidate F1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for letter in LABELS:
        details=letters[(letters.fold=='pooled')&(letters.letter==letter)].set_index('model');u,v=details.loc['original'],details.loc['gh_y180']
        lines.append(f'| {letter} | {u.precision:.6f} | {v.precision:.6f} | {u.recall:.6f} | {v.recall:.6f} | {u.f1:.6f} | {v.f1:.6f} |')
    lines += ['', '## Fold changes', '', '| Fold | Reference F1 | Candidate F1 | Fixed | Lost | Net |', '|---|---:|---:|---:|---:|---:|']
    for fold in sorted(predictions.fold.unique()):
        rows=tables[tables.fold==str(fold)].set_index('model');pair=paired[paired.fold==str(fold)].iloc[0]
        lines.append(f'| {fold} | {rows.loc["original","macro_f1"]:.8f} | {rows.loc["gh_y180","macro_f1"]:.8f} | {pair.fixed} | {pair.lost} | {pair.net:+d} |')
    lines += ['', '## Previously selected G/H camera cases', '',
              'None enter fitting. These cases informed earlier development, so their replay is diagnostic and cannot '
              'estimate unseen back-of-hand accuracy. All 22 saved cases are in `capture_replay.csv`.', '',
              '| Intended | Reference | G/H y180 |', '|---|---|---|']
    for row in captures[captures.intended.isin(['G','H'])].itertuples():lines.append(f'| {row.intended} | {row.original} | {row.gh_y180} |')
    lines += ['', '## Interpretation and scope', '',
              'This isolates adding G/H y180 fitting copies on the original-data SVM. It does not measure an interaction '
              'with mirroring or the value of camera calibration. The earlier multi-angle G/H recipe and the '
              'mirror-plus-calibration candidates are different experiments with separate artifacts. '
              'This was requested after earlier development and is exploratory, not a newly untouched final test.', '',
              'The rotation is exact for the descriptor geometry but operates in normalized landmark coordinates with '
              'estimated depth. It cannot reveal hidden fingers or simulate MediaPipe errors in real back views. '
              'Good original-view scores do not validate back-view recognition. Filename blocks are not verified signers. '
              'Historical detector confidence thresholds are not retained. No significance, new-person accuracy or general '
              'webcam robustness is claimed.', '',
              f'Checks passed: {metric_checks} aggregate rows, {letter_checks} letter rows, {paired_checks} paired rows, '
              f'{replays} fold-model replays and {len(captures)} camera replays, plus source/model/image hashes, '
              'disjoint source groups and original-only preprocessing/weights.', '',
              '## Reproduce and use live', '', '```bash',
              'python -m research.studies.gh_rotation_standalone --stage freeze',
              'python -m research.studies.gh_rotation_standalone --stage cv',
              'python -m research.studies.gh_rotation_standalone --stage train',
              'python -m research.studies.report_gh_rotation_standalone',
              'python -m research.live_camera --model models/asl_shin_angles_GH_y180.joblib', '```', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    numbers={'GHRotPrecisionBase':f'{100*a.macro_precision:.5f}','GHRotPrecisionAug':f'{100*b.macro_precision:.5f}',
             'GHRotFoneBase':f'{a.macro_f1:.8f}','GHRotFoneAug':f'{b.macro_f1:.8f}',
             'GHRotAccBase':f'{100*a.accuracy:.5f}','GHRotAccAug':f'{100*b.accuracy:.5f}',
             'GHRotFixed':str(int(change.fixed)),'GHRotLost':str(int(change.lost)),'GHRotNet':f'{int(change.net):+d}'}
    Path('paper/gh_y180_numbers.tex').write_text('% Generated from the standalone experiment; do not edit numbers manually.\n'+
        ''.join('\\newcommand{\\'+key+'}{'+value+'}\n' for key,value in numbers.items()))
    pooled_rows=[];letter_tex=[];fold_tex=[]
    for arm,label in [('original','Original only'),('gh_y180','Original + G/H $y180$')]:
        m=summary.loc[arm];pooled_rows.append(f'{label} & {100*m.accuracy:.5f} & {100*m.macro_precision:.5f} & {m.macro_f1:.8f} & {int(m.correct):,}\\\\')
    for letter in LABELS:
        rows=letters[(letters.fold=='pooled')&(letters.letter==letter)].set_index('model');u,v=rows.loc['original'],rows.loc['gh_y180']
        letter_tex.append(f'{letter} & {int(u.images)} & {u.precision:.5f} & {v.precision:.5f} & {u.recall:.5f} & {v.recall:.5f} & {u.f1:.5f} & {v.f1:.5f}\\\\')
    for fold in sorted(predictions.fold.unique()):
        rows=tables[tables.fold==str(fold)].set_index('model');pair=paired[paired.fold==str(fold)].iloc[0]
        fold_tex.append(f'{fold} & {rows.loc["original","macro_f1"]:.8f} & {rows.loc["gh_y180","macro_f1"]:.8f} & {int(pair.fixed)} & {int(pair.lost)} & {int(pair.net):+d}\\\\')
    for name,rows in [('main',pooled_rows),('letters',letter_tex),('folds',fold_tex)]:
        Path(f'paper/gh_y180_{name}_rows.tex').write_text('\n'.join(rows)+'\n')
    camera_tex=[]
    for row in captures[captures.intended.isin(['G','H'])].itertuples():
        view='Palm control' if row.id=='20261031T212827197875Z_759e34da' else 'Back-view development'
        camera_tex.append(f'{view} & {row.intended} & {row.original} & {row.gh_y180}\\\\')
    Path('paper/gh_y180_captures_rows.tex').write_text('\n'.join(camera_tex)+'\n')
    print(json.dumps({'verification':verification,'pooled':summary.reset_index().to_dict('records'),'paired':change.to_dict()},indent=2),flush=True)


if __name__=='__main__':main()
