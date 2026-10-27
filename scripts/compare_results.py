"""Compare input variants on the same photos; optionally run the reserved final test."""
import argparse
from gesturelab.training.comparison import compare_variants
from gesturelab.features.training_inputs import TRAINING_INPUTS, DEFAULT_VARIANTS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',default='experiments/ngt_01');p.add_argument('--project')
p.add_argument('--final-test',action='store_true')
p.add_argument('--variants', nargs='+', choices=list(TRAINING_INPUTS), default=DEFAULT_VARIANTS)
if __name__=='__main__':
    a=p.parse_args()
    try:print(compare_variants(a.experiment,a.final_test,a.project,a.variants)/'report.html')
    except (ValueError,OSError) as e:p.exit(1,str(e)+'\n')
