"""Freeze the data split, feature code, thresholds and model choice for both variants."""
import argparse
from gesturelab.training.experiment_plan import prepare_experiment
from gesturelab.training.experiments import MODEL_NAMES
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt');p.add_argument('--out',default='experiments/ngt_01')
p.add_argument('--folds',type=int,default=3);p.add_argument('--grouping',choices=['session','participant'],default='session')
p.add_argument('--model',choices=list(MODEL_NAMES),default='svm');p.add_argument('--labels',nargs='+')
p.add_argument('--seed',type=int,default=42);p.add_argument('--allow-no-test',action='store_true')
if __name__=='__main__':
    a=p.parse_args()
    try:
        print(prepare_experiment(a.project,a.out,a.folds,a.grouping,a.labels,a.model,a.seed,a.allow_no_test))
    except (ValueError,OSError) as e:p.exit(1,str(e)+'\n')
