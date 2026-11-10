"""Separate synthetic test-view control; never treated as real dorsal RGB data."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np

from research.data import atomic_json
from research.gh_vertical import rotate_vertical
from research.inputs import LABELS, write_csv
from research.studies.gh_occlusion import OUT, WORK, PREVIOUS, digest
from research.studies.gh_rotation_standalone import inputs
from research.studies.mirror_augmentation import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',default=['original','gh_y180','noise002','noise005'])
    args=parser.parse_args()
    protocol={'version':'synthetic-gh-test-view-v1',
        'motivation':'User clarified that retaining palm-facing GH classification is not the target; separate transformed-geometry test requested after primary study freeze',
        'test':'In held-out source blocks, rotate only G/H feature rows by diag(-1,1,-1); other letters retain original OOF predictions',
        'fitting':'Reuse frozen original/rotation/proxy fold models; no refitting, new selection or threshold tuning',
        'denominator':'60251 accepted sources, 4801 synthetic G/H descriptions replacing their originals; 26-letter precision and F1',
        'claim_limit':'Verifies learned synthetic geometry only; does not simulate actual RGB occlusion or test ASL correctness',
        'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path('research/gh_vertical.py')]},
        'parent_protocol_sha256':digest(OUT/'protocol.json')}
    path=OUT/'synthetic_test_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:atomic_json(path,protocol)
    X,y,ids,splits=inputs();pred={a:np.full(len(y),'',dtype='<U1') for a in args.arms};assignment=np.zeros(len(y),int)
    for fold,_,te in splits:
        assignment[te]=fold;gh=te[np.isin(y[te],['G','H'])]
        for arm in args.arms:
            folder=PREVIOUS if arm in ('original','gh_y180') else WORK
            with np.load(folder/f'{arm}_fold{fold}.npz') as saved:
                np.testing.assert_array_equal(ids[te],saved['ids']);pred[arm][te]=saved['prediction']
            checkpoint=WORK/f'synthetic_GH_test_{arm}_fold{fold}.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as saved:
                    np.testing.assert_array_equal(saved['ids'],ids[gh]);turned=saved['prediction']
            else:
                model=joblib.load(folder/f'{arm}_fold{fold}.joblib');turned=model.predict(rotate_vertical(X[gh]))
                np.savez_compressed(checkpoint,ids=ids[gh],prediction=turned)
            pred[arm][gh]=turned
        print(f'synthetic GH test: fold {fold}',flush=True)
    rows=[];letters=[];pairs=[]
    for fold in ('pooled',1,2,3,4,5):
        keep=np.ones(len(y),bool) if fold=='pooled' else assignment==fold
        for arm in args.arms:
            score,detail=metrics(y[keep],pred[arm][keep]);rows.append({'model':arm,'fold':fold,**score})
            for j,label in enumerate(LABELS):letters.append({'model':arm,'fold':fold,'letter':label,'n':int(detail[3][j]),'precision':float(detail[0][j]),'recall':float(detail[1][j]),'f1':float(detail[2][j])})
            truth=y[keep];a,b=pred['original'][keep],pred[arm][keep]
            fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
            pairs.append({'model':arm,'fold':fold,'fixed':fixed,'lost':lost,'net':fixed-lost})
    suffix='' if len(args.arms)==4 else '_controls'
    write_csv(OUT/f'synthetic_test_metrics{suffix}.csv',rows);write_csv(OUT/f'synthetic_test_letters{suffix}.csv',letters);write_csv(OUT/f'synthetic_test_paired{suffix}.csv',pairs)
    write_csv(OUT/f'synthetic_test_predictions{suffix}.csv',[{'id':ids[i],'label':y[i],'fold':int(assignment[i]),**{a:pred[a][i] for a in args.arms}} for i in range(len(y))])
    print(json.dumps([r for r in rows if r['fold']=='pooled'],indent=2),flush=True)


if __name__=='__main__':main()
