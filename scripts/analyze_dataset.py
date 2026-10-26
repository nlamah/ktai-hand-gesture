"""Compute hand landmarks for uploaded photos. Webcam captures already have them."""
import argparse
from pathlib import Path
from gesturelab.datasets.data import analyze_project, read_manifest
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',default='data/ngt')
if __name__=='__main__':
    project=Path(p.parse_args().project).expanduser().resolve()
    count=analyze_project(project)
    samples=read_manifest(project)['samples']
    successful=sum(sample['status']=='ok' for sample in samples)
    pending=sum(sample['status']=='pending' for sample in samples)
    failed=len(samples)-successful-pending
    print(f"Project: {project}")
    print(f"{count} new images analyzed")
    print(f"Dataset: {len(samples)} images; {successful} successful, {pending} pending, {failed} failed detections.")
    if samples and count==0 and pending==0:
        print("No images are waiting for analysis. Guided webcam photos are analyzed when saved.")
    elif not samples:
        print("No images found. Check the project directory used during recording.")
