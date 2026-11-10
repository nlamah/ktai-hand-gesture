"""Label-blind coarse hand volumes with an orthographic depth buffer.

This renderer estimates finger/palm thickness; it is not an anatomical hand fit.
Coordinates come from the stored image-space landmarks, never from class labels.
"""
import numpy as np
from PIL import Image

VERSION = 'landmark-volumes-depthbuffer-v1'
FINGERS = ((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20))
PARAMETERS = {'resolution':384,'finger_base_radius':.13,'finger_tip_radius':.085,
              'thumb_radius':.16,'palm_half_thickness':.16,'wrist_length':.65,
              'wrist_radius':.25,
              'skin_rgb':[213,161,123],'background_rgb':[226,230,233]}


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-10:raise ValueError('Degenerate synthetic hand geometry.')
    return vector/length


class Surface:
    def __init__(self, size, scale, center):
        self.size,self.scale,self.center=size,scale,center
        self.depth=np.full((size,size),np.inf,np.float32)
        self.color=np.zeros((size,size,3),np.float32)
        self.light=unit(np.array([-.35,-.55,-1.]))

    def bounds(self, lo, hi):
        left,top=np.maximum(np.floor(lo*self.scale+self.center).astype(int),0)
        right,bottom=np.minimum(np.ceil(hi*self.scale+self.center).astype(int)+1,self.size)
        if right<=left or bottom<=top:return None
        xx,yy=np.meshgrid((np.arange(left,right)-self.center[0])/self.scale,
                          (np.arange(top,bottom)-self.center[1])/self.scale)
        return left,top,right,bottom,xx,yy

    def paint(self,bounds,z,normal,color,inside):
        left,top,right,bottom,*_=bounds
        depth=self.depth[top:bottom,left:right]
        mask=inside&(z<depth)
        shade=.44+.56*np.maximum(0,np.sum(normal*self.light,axis=-1))
        colors=np.clip(np.asarray(color)*shade[...,None]+5,0,255)
        colors=np.broadcast_to(colors,z.shape+(3,))
        depth[mask]=z[mask]
        self.color[top:bottom,left:right][mask]=colors[mask]

    def sphere(self,point,radius,color):
        bounds=self.bounds(point[:2]-radius,point[:2]+radius)
        if bounds is None:return
        *_,xx,yy=bounds
        dx,dy=xx-point[0],yy-point[1]
        remaining=radius*radius-dx*dx-dy*dy
        dz=np.sqrt(np.maximum(0,remaining))
        normal=np.stack([dx/radius,dy/radius,-dz/radius],axis=-1)
        self.paint(bounds,point[2]-dz,normal,color,remaining>=0)

    def capsule(self,a,b,ra,rb,color):
        self.sphere(a,ra,color);self.sphere(b,rb,color)
        length=np.linalg.norm(b-a)
        if length<1e-10:return
        v=(b-a)/length;k=(rb-ra)/length
        bounds=self.bounds(np.minimum(a[:2],b[:2])-max(ra,rb),np.maximum(a[:2],b[:2])+max(ra,rb))
        if bounds is None:return
        *_,xx,yy=bounds
        q=np.stack([xx-a[0],yy-a[1],np.full_like(xx,-a[2])],axis=-1)
        h=np.sum(q*v,axis=-1)
        perpendicular=q-h[...,None]*v
        d=np.array([0.,0.,1.])-v[2]*v
        A=1-v[2]**2-(k*v[2])**2
        if abs(A)<1e-10:return
        B=2*np.sum(perpendicular*d,axis=-1)-2*(ra+k*h)*k*v[2]
        C=np.sum(perpendicular**2,axis=-1)-(ra+k*h)**2
        discriminant=B*B-4*A*C
        z=(-B-np.sqrt(np.maximum(0,discriminant)))/(2*A)
        along=h+z*v[2]
        offset=q+z[...,None]*np.array([0,0,1])-along[...,None]*v
        r=ra+k*along
        normal=(offset/np.maximum(r[...,None],1e-8)-k*v)/np.sqrt(1+k*k)
        self.paint(bounds,z,normal,color,(discriminant>=0)&(along>=0)&(along<=length)&(r>0))

    def ellipsoid(self,point,radii,basis,color):
        Q=(basis/(np.asarray(radii)**2))@basis.T
        bounds_xy=np.sqrt(np.sum((basis[:2]*np.asarray(radii))**2,axis=1))
        bounds=self.bounds(point[:2]-bounds_xy,point[:2]+bounds_xy)
        if bounds is None:return
        *_,xx,yy=bounds
        q=np.stack([xx-point[0],yy-point[1],np.full_like(xx,-point[2])],axis=-1)
        A=Q[2,2];B=2*np.sum(q*Q[:,2],axis=-1)
        C=np.sum((q@Q)*q,axis=-1)-1
        disc=B*B-4*A*C
        z=(-B-np.sqrt(np.maximum(0,disc)))/(2*A)
        normal=(q+z[...,None]*np.array([0,0,1]))@Q
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
        self.paint(bounds,z,normal,color,disc>=0)

    def triangle(self,vertices,color):
        bounds=self.bounds(vertices[:,:2].min(axis=0),vertices[:,:2].max(axis=0))
        if bounds is None:return
        *_,xx,yy=bounds
        a,b,c=vertices
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-10:return
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w=1-u-v
        normal=unit(np.cross(b-a,c-a))
        if normal[2]>0:normal=-normal
        self.paint(bounds,u*a[2]+v*b[2]+w*c[2],normal,color,(u>=-1e-8)&(v>=-1e-8)&(w>=-1e-8))


def render_hand(image_landmarks, degrees=180, image_size=(200,200)):
    points=np.asarray(image_landmarks,dtype=float)
    if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('Expected finite 21 x 3 hand points.')
    points=points-points[0]
    points[:,1]*=image_size[1]/image_size[0]
    points/=np.linalg.norm(points[9])
    forward=unit(points[9])
    normal=unit(np.cross(points[5],points[17]))
    if normal[2]>0:normal=-normal
    wrist=points[0]-forward*PARAMETERS['wrist_length']
    lateral=unit(points[5]-points[17]-forward*np.dot(points[5]-points[17],forward))
    palm_center=points[[5,9,13,17]].mean(axis=0)*.47
    palm_radii=[max(abs((points[[1,5,9,13,17]]-palm_center)@lateral))+.08,
                np.linalg.norm(points[[5,9,13,17]].mean(axis=0))*.57,
                PARAMETERS['palm_half_thickness']]
    angle=np.deg2rad(degrees)
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    original_normal=unit(np.cross(lateral,forward))
    if original_normal[2]>0:original_normal=-original_normal
    points=points@rotation.T;palm_center=palm_center@rotation.T;wrist=wrist@rotation.T;normal=original_normal@rotation.T
    basis=rotation@np.column_stack([lateral,forward,original_normal])
    all_points=np.r_[points,wrist[None,:]]
    lo,hi=all_points[:,:2].min(axis=0)-.32,all_points[:,:2].max(axis=0)+.32
    size=PARAMETERS['resolution'];scale=(size-1)/max(hi-lo)
    center=np.array([size/2,size/2])-(lo+hi)*.5*scale
    surfaces=[Surface(size,scale,center) for _ in range(6)]
    skin=PARAMETERS['skin_rgb']
    surfaces[0].ellipsoid(palm_center,palm_radii,basis,skin)
    surfaces[0].capsule(wrist,points[0],.24,PARAMETERS['wrist_radius'],skin)
    for f,indices in enumerate(FINGERS):
        radii=np.linspace(PARAMETERS['finger_base_radius'],PARAMETERS['finger_tip_radius'],4)
        if f==0:radii*=PARAMETERS['thumb_radius']/PARAMETERS['finger_base_radius']
        if f==4:radii*=.8
        for j in range(3):surfaces[f+1].capsule(points[indices[j]],points[indices[j+1]],radii[j],radii[j+1],skin)
        # Small light volumes on the inferred dorsal side approximate fingernails.
        a,b=points[indices[-2]],points[indices[-1]]
        direction=unit(b-a)
        back=-normal+direction*np.dot(normal,direction)
        if np.linalg.norm(back)>1e-8:
            back=unit(back)
            p=a*.2+b*.8+back*radii[-1]*.92
            surfaces[f+1].ellipsoid(p,[radii[-1]*.65,min(np.linalg.norm(b-a)*.25,radii[-1]),.018],
                                  np.column_stack([unit(np.cross(direction,back)),direction,back]),[232,198,176])
    depths=np.stack([s.depth for s in surfaces])
    owner=np.argmin(depths,axis=0)
    depth=depths.min(axis=0)
    colors=np.stack([s.color for s in surfaces])
    color=np.take_along_axis(colors,owner[None,:,:,None],axis=0)[0]
    background=np.broadcast_to(np.array(PARAMETERS['background_rgb'],np.float32),(size,size,3)).copy()
    background+=np.linspace(7,-7,size)[:,None,None]
    color=np.where(np.isfinite(depth)[...,None],color,background)
    stats={}
    for i,name in enumerate(('palm','thumb','index','middle','ring','little')):
        area=np.isfinite(depths[i]);count=int(area.sum())
        stats[name]={'projected_pixels':count,'hidden_pixels':int((area&(owner!=i)).sum())}
    projected=points[:,:2]*scale+center
    hidden=[]
    for j in (2,3,4,6,7,8,10,11,12,14,15,16,18,19,20):
        x,y=np.clip(np.rint(projected[j]).astype(int),0,size-1)
        if owner[y,x]!=1+(j-1)//4:hidden.append(j)
    return Image.fromarray(np.clip(color,0,255).astype(np.uint8)),{
        'degrees':degrees,'component_occlusion':stats,'foreground_pixels':int(np.isfinite(depth).sum()),
        'projected_landmarks':projected.tolist(),'hidden_finger_joints':hidden,
        'rotation_determinant':float(np.linalg.det(rotation))}
