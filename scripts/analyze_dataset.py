"""Compute hand landmarks for uploaded photos. Webcam captures already have them."""
import argparse
from pathlib import Path
from gesturelab.datasets.data import analyze_project, read_manifest
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt')
p.add_argument('--checkpoint-every', type=int, default=1, help='Save analysis after this many images.')
p.add_argument('--no-cache-features', action='store_true', help='Store landmarks; recompute features during training.')
p.add_argument('--min-hand-detection-confidence', type=float, default=.5,
               help='MediaPipe detection confidence threshold (0 to 1).')
p.add_argument('--min-hand-presence-confidence', type=float, default=.5,
               help='MediaPipe hand-presence confidence threshold (0 to 1).')
p.add_argument('--reprocess', action='store_true',
               help='Reanalyze every image, replacing stored detection results.')
if __name__=='__main__':
    args=p.parse_args()
    project=Path(args.project).expanduser().resolve()
    count=analyze_project(
        project, checkpoint_every=args.checkpoint_every, cache_features=not args.no_cache_features,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence, reprocess=args.reprocess)
    samples=read_manifest(project)['samples']
    successful=sum(sample['status']=='ok' for sample in samples)
    pending=sum(sample['status']=='pending' for sample in samples)
    failed=len(samples)-successful-pending
    print(f"Project: {project}")
    print(f"{count} {'images reanalyzed' if args.reprocess else 'new images analyzed'}")
    if args.reprocess:
        print("All saved detection results were replaced.")
    print(f"Dataset: {len(samples)} images; {successful} successful, {pending} pending, {failed} failed detections.")
    if samples and count==0 and pending==0:
        print("No images are waiting for analysis. Guided webcam photos are analyzed when saved.")
    elif not samples:
        print("No images found. Check the project directory used during recording.")
