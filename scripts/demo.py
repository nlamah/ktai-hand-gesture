"""Generate explicitly synthetic hand skeletons for software tests, not accuracy claims."""
import argparse
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from gesturelab.datasets.data import add_image,read_manifest,atomic_json,EDGES
from gesturelab.features.inspection import extract_features,FEATURE_VERSION

LABELS=["fist","open_hand","peace","pointing","thumbs_up"]


def skeleton(label):
    p=np.zeros((21,3),float)
    p[0]=[0,-.9,0]
    bases={5:(-.4,0,0),9:(-.12,.1,0),13:(.18,.05,0),17:(.45,-.05,0)}
    extended={"fist":[],"open_hand":[5,9,13,17],"peace":[5,9],"pointing":[5],"thumbs_up":[]}[label]
    for base,point in bases.items():
        p[base]=point
        if base in extended:
            direction=np.array([-.20 if base==5 and label=="peace" else .03,.98,.02])
            p[base+1]=p[base]+direction*.35
            p[base+2]=p[base+1]+direction*.25
            p[base+3]=p[base+2]+direction*.2
        else:
            p[base+1]=p[base]+[0,.17,0]
            p[base+2]=p[base+1]+[0,-.10,.16]
            p[base+3]=p[base+2]+[0,-.17,-.02]
    if label=="thumbs_up":
        p[1:5]=[[-.45,-.3,.03],[-.7,.12,.03],[-.9,.47,.03],[-1.1,.8,.03]]
    elif label=="open_hand":
        p[1:5]=[[-.45,-.3,.03],[-.65,-.15,.03],[-.85,0,.03],[-1.05,.15,.03]]
    else:
        p[1:5]=[[-.45,-.3,.03],[-.5,-.1,.05],[-.35,0,.12],[-.15,0,.10]]
    return p


def _demo_coordinates(rng, label, person, session):
    points=skeleton(label)+rng.normal(0,.008,(21,3))
    theta=rng.normal(.3*session,.12)
    rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    points=points@rot.T
    world=points*(.065+.007*person)
    scale=rng.uniform(.22,.32)
    raw=points*scale
    raw[:,0]+=rng.uniform(.50,.63)
    raw[:,1]=.60-raw[:,1]
    raw[:,2]-=raw[0,2]
    return raw, world


def _demo_image(raw, label, person, session, i):
    image=Image.new("RGB",(320,320),(240,244,248))
    draw=ImageDraw.Draw(image)
    xy=[(float(p[0]*320),float(p[1]*320)) for p in raw]
    for a,b in EDGES:
        draw.line([xy[a],xy[b]],fill=(40,140,130),width=4)
    for x,y in xy:
        draw.ellipse((x-3,y-3,x+3,y+3),fill=(230,150,40))
    draw.text((10,10),"SYNTHETIC SOFTWARE TEST",fill=(100,40,40))
    draw.text((10,295),f"{label} / {person} / {session} / {i}",fill=(60,60,60))
    data=BytesIO()
    image.save(data,format="PNG")
    return image, data.getvalue()


def _store_demo_sample(project, image, data, raw, world, label, person, session, i):
    sample,_=add_image(project,data,f"synthetic_{person}_{session}_{label}_{i}.png",label,
        f"session{session+1}",f"participant{person+1}","rotated" if session else "baseline",
        "test" if person==3 else "development",{"type":"synthetic","note":"Known generated landmarks; MediaPipe is not used."})
    manifest=read_manifest(project)
    record=next(s for s in manifest["samples"] if s["id"]==sample["id"])
    record.update({"status":"ok","image_landmarks":raw.tolist(),"world_landmarks":world.tolist(),
        "features":extract_features(raw,world,image.width,image.height),"feature_version":FEATURE_VERSION,
        "mediapipe_version":"not-used-synthetic","handedness":"Right","handedness_score":1.0})
    atomic_json(project/"manifest.json",manifest)


def make_demo(project):
    project = Path(project)
    if read_manifest(project)['samples']:
        raise ValueError("Use an empty demo project; do not mix synthetic data with real photos.")
    rng = np.random.default_rng(2026)
    for person in range(4):
        for session in range(2):
            for label in LABELS:
                for i in range(4):
                    raw, world = _demo_coordinates(rng, label, person, session)
                    image, data = _demo_image(raw, label, person, session, i)
                    _store_demo_sample(project, image, data, raw, world, label, person, session, i)
    return project


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="projects/demo_synthetic")
    args=parser.parse_args()
    print(make_demo(args.out))
