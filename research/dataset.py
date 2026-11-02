"""Import or extract landmarks for the sole ASL Alphabet dataset."""
import argparse
from pathlib import Path
from research.kaggle import import_kaggle_dataset
from research.data import analyze_project

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('import');a.add_argument('--source',type=Path,required=True);a.add_argument('--project',type=Path,required=True);a.add_argument('--resume',action='store_true')
 b=sub.add_parser('analyze');b.add_argument('--project',type=Path,required=True)
 args=p.parse_args()
 if args.command=='import':print(import_kaggle_dataset(args.source,args.project,resume=args.resume,exclude_overlapping_test=True))
 else:print(analyze_project(args.project,checkpoint_every=1000,cache_features=False))
if __name__=='__main__':main()
