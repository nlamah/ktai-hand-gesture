"""Precision/recall of manual finger assessments on development photos."""
import argparse
from gesturelab.features.audit import load_thresholds,write_audit
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt');p.add_argument('--out',default='results/feature_check')
if __name__=='__main__':
    a=p.parse_args()
    try:
        print(write_audit(a.project,a.out,load_thresholds(a.project)))
    except ValueError as e:
        p.exit(1,str(e)+'\n')
