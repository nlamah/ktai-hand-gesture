'Inspectable finger features. Geometry only; no ML or letter labels.\n\nStraight joints have an interior angle of 180 degrees.\nStraightness is the base-to-tip distance divided by the bone-path length.\nThresholds are hypotheses to check using development photos.\n'
from itertools import combinations
import numpy as np

FINGERS={"thumb":(1,2,3,4),"index":(5,6,7,8),"middle":(9,10,11,12),
         "ring":(13,14,15,16),"pinky":(17,18,19,20)}
FINGER_NAMES={"thumb":"thumb","index":"index finger","middle":"middle finger","ring":"ring finger","pinky":"little finger"}
DEFAULT_THRESHOLDS={"extended_angle":155.0,"thumb_extended_angle":145.0,
                    "bent_angle":125.0,"extended_straightness":0.8,"bent_straightness":0.65}


def validate_thresholds(config=None):
    result=dict(DEFAULT_THRESHOLDS)
    if config is not None:
        if not isinstance(config,dict) or set(config)-set(result):
            raise ValueError("Unknown threshold setting.")
        result.update(config)
    for key,value in result.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
    if not 0<=result['bent_angle']<min(result['extended_angle'],result['thumb_extended_angle'])<=180:
        raise ValueError("The bending angle must be below both extension angles (0–180 degrees).")
    if max(result['extended_angle'],result['thumb_extended_angle'])>180:
        raise ValueError("A joint angle cannot exceed 180 degrees.")
    if not 0<=result['bent_straightness']<result['extended_straightness']<=1:
        raise ValueError("Straightness thresholds must satisfy 0 ≤ bent < extended ≤ 1.")
    return result


def unit(vector):
    length=np.linalg.norm(vector)
    if length<1e-8:
        raise ValueError("Coincident hand landmarks: reliable geometry is not possible.")
    return vector/length


def angle(a,b,c):
    'Interior angle at B, in degrees. Straight: 180; right angle: 90.'
    a,b,c=map(lambda v:np.asarray(v,dtype=float),(a,b,c))
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a-b),unit(c-b)),-1,1))))


def classify_finger(angle1,angle2,straightness,finger="index",thresholds=None):
    'Two booleans; both False means intermediate. Independent of letter labels.'
    t=validate_thresholds(thresholds)
    if finger not in FINGERS:
        raise ValueError("Unknown finger.")
    values=np.array([angle1,angle2,straightness],dtype=float)
    if not np.isfinite(values).all() or not (0<=angle1<=180 and 0<=angle2<=180 and 0<=straightness<=1+1e-8):
        raise ValueError("Invalid angle or straightness.")
    limit=t['thumb_extended_angle'] if finger=='thumb' else t['extended_angle']
    extended=angle1>=limit and angle2>=limit and straightness>=t['extended_straightness']
    bent=angle1<t['bent_angle'] or angle2<t['bent_angle'] or straightness<t['bent_straightness']
    return bool(extended),bool(bent)


def _hand_geometry(world_landmarks):
    world=np.asarray(world_landmarks,dtype=float)
    if world.shape!=(21,3) or not np.isfinite(world).all():
        raise ValueError("Expected 21 x 3 finite world coordinates.")
    scale=np.linalg.norm(world[9]-world[0])
    if scale<1e-8:
        raise ValueError("Measured hand size is zero.")
    points=(world-world[0])/scale
    ey=unit(points[9])
    ex0=points[5]-points[17]
    ex=unit(ex0-np.dot(ex0,ey)*ey)
    ez=unit(np.cross(ex,ey))
    local=points@np.stack([ex,ey,ez],axis=1)
    palm_center=np.mean(points[[0,5,9,13,17]],axis=0)
    return points, local, palm_center


def _per_finger_features(points, local, palm_center, thresholds):
    t = thresholds
    features={}
    # 10 features per finger: 50 in total.
    for finger,(a,b,c,d) in FINGERS.items():
        angle1=angle(points[a],points[b],points[c])
        angle2=angle(points[b],points[c],points[d])
        path=sum(np.linalg.norm(points[v]-points[u]) for u,v in [(a,b),(b,c),(c,d)])
        straightness=float(np.linalg.norm(points[d]-points[a])/path)
        extended,bent=classify_finger(angle1,angle2,straightness,finger,t)
        values={"joint1_deg":angle1,"joint2_deg":angle2,"base_proxy_deg":angle(points[0],points[a],points[b]),
                "straightness":straightness,"extended":float(extended),"bent":float(bent),
                "tip_palm_distance":float(np.linalg.norm(points[d]-palm_center))}
        values.update({f"tip_local_{axis}":float(local[d,j]) for j,axis in enumerate('xyz')})
        features.update({f"geom_{finger}_{name}":value for name,value in values.items()})
    return features


def _fingertip_distances(points):
    features = {}
    # 10 fingertip distances: for example thumb/index contact and V versus U.
    for left,right in combinations(FINGERS,2):
        i,j=FINGERS[left][-1],FINGERS[right][-1]
        features[f"geom_tip_distance_{left}_{right}"]=float(np.linalg.norm(points[i]-points[j]))
    return features


def _thumb_distances(points):
    features = {}
    # 12 thumb-to-knuckle distances: thumb position differences in closed hands.
    for finger in ('index','middle','ring','pinky'):
        a,b,c,_=FINGERS[finger]
        for joint,point in [('mcp',a),('pip',b),('dip',c)]:
            features[f"geom_thumb_{finger}_{joint}_distance"]=float(np.linalg.norm(points[4]-points[point]))
    return features


def _adjacent_finger_features(points, local):
    features = {}
    # 9 adjacent-finger features: spread, crossing and relative depth difference.
    for left,right in [('index','middle'),('middle','ring'),('ring','pinky')]:
        a,d=FINGERS[left][0],FINGERS[left][-1]
        b,e=FINGERS[right][0],FINGERS[right][-1]
        features[f"geom_spread_{left}_{right}_deg"]=float(np.degrees(np.arccos(np.clip(
            np.dot(unit(points[d]-points[a]),unit(points[e]-points[b])),-1,1))))
        features[f"geom_crossed_{left}_{right}"]=float((local[d,0]-local[e,0])*(local[a,0]-local[b,0])<0)
        features[f"geom_depth_{left}_{right}"]=float(local[d,2]-local[e,2])
    return features


def compute_finger_features(world_landmarks, thresholds=None):
    """Calculate 81 label-independent measurements in normalized palm coordinates."""
    thresholds = validate_thresholds(thresholds)
    points, local, palm_center = _hand_geometry(world_landmarks)
    features = _per_finger_features(points, local, palm_center, thresholds)
    features.update(_fingertip_distances(points))
    features.update(_thumb_distances(points))
    features.update(_adjacent_finger_features(points, local))
    if not np.isfinite(list(features.values())).all():
        raise ValueError('Invalid finger features.')
    return features


def finger_features_for_sample(sample, thresholds=None):
    """Adapt one recorded sample to the independent geometric calculations."""
    return compute_finger_features(sample['world_landmarks'], thresholds)
