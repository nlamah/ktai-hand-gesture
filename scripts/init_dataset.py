"""Create a dataset directory with alphabet A–Z and reference images."""
import argparse
from gesturelab.datasets.alphabet import initialize_alphabet
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt')
if __name__=='__main__':
    print(initialize_alphabet(p.parse_args().project).resolve())
    print('26 letter examples ready.')
