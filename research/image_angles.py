"""Projected joint angles in aspect-correct pixels, complementary to 3D angles."""
import numpy as np
from research.geometry import FINGER_JOINTS

VERSION='pixel-internal-joint-angles-v1'
PIXEL_ANGLE_COLUMNS=tuple(f'image2d_{joint}_angle_deg' for joint in FINGER_JOINTS)


def projected_angles_batch(landmarks, image_sizes):
    """Identical batch/train and single-image/live computation, returning float32."""
    points=np.asarray(landmarks,dtype=float)
    sizes=np.asarray(image_sizes,dtype=float)
    if points.ndim!=3 or points.shape[1:]!=(21,3) or not np.isfinite(points).all():
        raise ValueError('Expected finite N x 21 x 3 image landmarks.')
    if sizes.shape!=(len(points),2) or not np.isfinite(sizes).all() or np.any(sizes<=0):
        raise ValueError('Image sizes must contain positive width and height.')
    xy=points[:,:,:2]*sizes[:,None,:]
    indices=np.array(list(FINGER_JOINTS.values()))
    u=xy[:,indices[:,0]]-xy[:,indices[:,1]]
    v=xy[:,indices[:,2]]-xy[:,indices[:,1]]
    length=np.linalg.norm(u,axis=2)*np.linalg.norm(v,axis=2)
    cosine=np.full(length.shape,np.nan)
    np.divide(np.sum(u*v,axis=2),length,out=cosine,where=length>1e-12)
    return np.rad2deg(np.arccos(np.clip(cosine,-1,1))).astype(np.float32)


def projected_angles(landmarks, image_size):
    """14 internal pixel angles; NaN for coincident projected segments.

    These are view-dependent projected angles, not anatomical 3D joint angles.
    Normalized x/y are scaled with the detector-image width/height first.
    """
    return projected_angles_batch(np.asarray(landmarks)[None,...],np.asarray(image_size)[None,...])[0]
