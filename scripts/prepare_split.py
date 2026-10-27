"""Freeze the data split, feature code, thresholds and model choice for both variants."""
import argparse
from gesturelab.training.experiment_plan import prepare_experiment
from gesturelab.training.experiments import MODEL_NAMES
from gesturelab.training.classifiers import PROFILES
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt');p.add_argument('--out',default='experiments/ngt_01')
p.add_argument('--folds',type=int,default=3);p.add_argument('--grouping',choices=['session','participant','source'],default='session')
p.add_argument('--model',choices=list(MODEL_NAMES),default='svm');p.add_argument('--labels',nargs='+')
p.add_argument('--classifier-profile', choices=PROFILES, default='project')
p.add_argument('--split-from', help='Existing split.json or experiment directory; reuse assignments in a new experiment.')
p.add_argument('--seed',type=int,default=42);p.add_argument('--allow-no-test',action='store_true')
p.add_argument('--allow-incomplete-test',action='store_true',help='Explicitly allow letters without reserved test photos.')
if __name__=='__main__':
    a=p.parse_args()
    try:
        print(prepare_experiment(a.project,a.out,a.folds,a.grouping,a.labels,a.model,a.seed,a.allow_no_test,a.allow_incomplete_test,a.classifier_profile,a.split_from))
    except (ValueError,OSError) as e:p.exit(1,str(e)+'\n')
