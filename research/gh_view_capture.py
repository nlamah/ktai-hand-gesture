"""Large Dutch G/H instructions: 40 photos per person, one to three people."""
import argparse
from hashlib import sha256
from pathlib import Path

import cv2

from research.gh_view_protocol import GHViewSession
from research.live_camera import run_camera
from research.live_model import DEFAULT_MODEL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participants',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--hands',nargs='+',choices=('right','left'),help='One hand per participant; default all right.')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=Path('captures/gh_viewpoint'))
    parser.add_argument('--resume',type=Path,help='Existing session folder; retains its original participants and hands.')
    parser.add_argument('--status',action='store_true',help='Show status without opening camera (requires --resume).')
    args=parser.parse_args()
    if args.status and not args.resume:parser.error('--status requires --resume')
    if args.hands and len(args.hands)!=args.participants:parser.error('Specify exactly one hand per participant.')
    try:
        model_sha=sha256(args.model.read_bytes()).hexdigest()
        session=GHViewSession(args.resume,model_sha) if args.resume else GHViewSession.create(args.output,model_sha,args.participants,args.hands)
        print(f'Sessie: {session.directory.resolve()}',flush=True)
        print(f'Voortgang: {len(session.completed)}/{len(session.plan["tasks"])}',flush=True)
        print('P01: development. Other participants: reserved evaluation; do not adapt models to their results.',flush=True)
        if args.status:return
        run_camera(args.model,args.camera,session.directory,protocol_session=session)
    except KeyboardInterrupt:pass
    except (OSError,ValueError,RuntimeError,ImportError,cv2.error) as error:parser.exit(1,f'G/H capture stopped: {error}\n')


if __name__=='__main__':main()
