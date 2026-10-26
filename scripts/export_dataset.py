"""Save original photos, labels, landmarks, references and assessments as a ZIP."""
import argparse
from pathlib import Path
from gesturelab.datasets.data import export_project
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt')
p.add_argument('--out',default='ngt_dataset.zip')
if __name__=='__main__':
    a=p.parse_args();Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    print(export_project(a.project,a.out))
