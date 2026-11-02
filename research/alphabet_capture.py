"""Named, example-guided alphabet test: one Space per photo; both models offline."""
import argparse
from hashlib import sha256
from pathlib import Path
import cv2
from research.alphabet_protocol import AlphabetSession,participant_form,profile
from research.live_camera import run_camera
from research.live_model import DEFAULT_MODEL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',help='Name/code; omit to open the participant form.')
    parser.add_argument('--kind',choices=('development','new'))
    parser.add_argument('--experience',choices=('beginner','familiar','experienced','unknown'),default='unknown')
    parser.add_argument('--hand',choices=('right','left'),default='right')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('captures/alphabet_test'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status and not args.resume:parser.error('--status requires --resume')
    if args.name and not args.kind and not args.resume:parser.error('--name requires --kind development or new')
    try:
        sha=sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
        if args.resume:session=AlphabetSession(args.resume,sha)
        else:
            p=profile(args.name,args.kind,args.experience,args.hand) if args.name else participant_form()
            if p is None:return
            session=AlphabetSession.create(args.output,p,sha,args.repeats)
        print(f'Sessie: {session.directory.resolve()}',flush=True)
        print(f"Participant: {session.plan['participant']['name']} | {len(session.completed)}/{len(session.plan['tasks'])}",flush=True)
        if not args.status:run_camera(DEFAULT_MODEL,args.camera,protocol_session=session)
    except KeyboardInterrupt:pass
    except (OSError,ValueError,RuntimeError,ImportError,cv2.error) as error:parser.exit(1,f'Alphabet capture stopped: {error}\n')


if __name__=='__main__':main()
