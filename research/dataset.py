"""Import or extract landmarks for the sole ASL Alphabet dataset."""
import argparse
from pathlib import Path
from research.kaggle import import_kaggle_dataset
from research.data import analyze_project

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('import');a.add_argument('--source',type=Path,required=True);a.add_argument('--project',type=Path,required=True);a.add_argument('--resume',action='store_true')
 b=sub.add_parser('analyze');b.add_argument('--project',type=Path,required=True)
 b.add_argument('--detection-confidence',type=float,default=.5,help='Minimum hand detection confidence (0–1; default .5).')
 b.add_argument('--presence-confidence',type=float,default=.5,help='Minimum landmark presence confidence (0–1; default .5).')
 b.add_argument('--reprocess',action='store_true',help='Analyze every image again, replacing existing detection results. Otherwise resume pending images only.')
 args=p.parse_args()
 if args.command=='import':print(import_kaggle_dataset(args.source,args.project,resume=args.resume,exclude_overlapping_test=True))
 else:print(analyze_project(args.project,checkpoint_every=1000,cache_features=False,
     min_hand_detection_confidence=args.detection_confidence,
     min_hand_presence_confidence=args.presence_confidence,reprocess=args.reprocess))
if __name__=='__main__':main()
