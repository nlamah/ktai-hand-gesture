import argparse
from gesturelab.training.fitting import train_variant
from gesturelab.features.training_inputs import TRAINING_INPUTS

def main(variant=None):
    p=argparse.ArgumentParser(description='Train '+(variant or 'an input variant')+' on the frozen shared data split.')
    p.add_argument('--experiment',default='experiments/ngt_01')
    p.add_argument('--project',help='New location of the same unchanged dataset after moving it.')
    if variant is None:
        p.add_argument('--variant', required=True, choices=list(TRAINING_INPUTS))
    a=p.parse_args()
    variant = variant or a.variant
    try:print(train_variant(a.experiment,variant,a.project))
    except (ValueError,OSError) as e:p.exit(1,str(e)+'\n')
