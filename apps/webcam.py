"""Start the local camera window directly, without Streamlit."""
import argparse
from gesturelab.camera.runtime import run_camera,new_session_name
from gesturelab.research.cli import add_native_commands, execute


def _legacy_commands(sub):
    record=sub.add_parser("record",help="Guided photo capture with reference and countdown.")
    record.add_argument("--labels",nargs="+",required=True)
    record.add_argument("--photos",type=int,default=3)
    record.add_argument("--interval",type=float,default=3)
    record.add_argument("--change-delay",type=float,default=5)
    record.add_argument("--participant",default="participant1")
    record.add_argument("--condition",default="baseline")
    record.add_argument("--role",choices=["development","test"],default="development")
    live=sub.add_parser("predict",help="Continuous recognition using a trained model.")
    live.add_argument("--model",required=True)
    live.add_argument("--inference-hz",type=float,default=10)
    live.add_argument("--smoothing",type=int,default=5)
    for p in (record,live):
        p.add_argument("--project",default="projects/my_gestures")
        p.add_argument("--camera",type=int,default=0)
        p.add_argument("--no-mirror",action="store_true")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    add_native_commands(sub)
    _legacy_commands(sub)
    args=vars(parser.parse_args())
    if 'function' in args:
        return execute(parser, args)
    args["mode"]="capture" if args["mode"]=="record" else "predict"
    args["camera_index"]=args.pop("camera")
    args["mirror"]=not args.pop("no_mirror")
    if args["mode"]=="capture":
        args["photos_per_label"]=args.pop("photos")
        args["session"]=new_session_name()
    try:
        run_camera(args)
    except Exception as error:
        parser.exit(1,f"Camera not started: {error}\n")


if __name__=="__main__":
    main()
