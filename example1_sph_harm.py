import numpy as np
from scipy.special import sph_harm
from curvpack import icosphere,curvature1,curvature2,curvature3,curvature4,curvature5,LB1

def cart2sph(xyz):
    ptsnew = np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

isph=icosphere(30)
npt=isph.p.shape[0]
nelem=isph.tri.shape[0]
vertices=isph.p
faces=isph.tri

spvert=cart2sph(vertices)
# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
dr1 = sph_harm(2, 5, spvert[:,2], spvert[:,1]).real

vertices+=0.2*dr1[:,None]*vertices

GC,MC,Normals=curvature1(vertices,faces)

with open(f'ex1_out.dat','wb') as f:
  f.write(bytes('Variables = "X" "Y" "Z" "U" "V" "W" "MC" "GC" \n',"utf-8"))
  f.write(bytes('ZONE F=FEPOINT,ET=TRIANGLE,N='+str(npt)+',E='+str(nelem)+'SOLUTIONTIME=0 \n','utf-8'))
  np.savetxt(f,np.c_[vertices,Normals,MC,GC],fmt='%16.9E '*8)
  np.savetxt(f,1+faces,fmt='%i %i %i')
