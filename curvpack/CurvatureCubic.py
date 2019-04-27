
"""
Abhilash Reddy Malipeddi. January 2017
Calculate the mean and gaussian curvature of a tri mesh using a modified version of cubic order algorithm
given in Goldfeather and Interrante 2004. I have have included the linear terms in the polynomial as well.


The first two functions are modified from MNE surface project

"""

import numpy as np
from numpy.linalg import lstsq
from .utils import  triangle_neighbors,GetVertexNormals,get_surf_neighbors,fastcross,normr

def CurvatureCubic(vertices,faces):

  npt=vertices.shape[0]
  neighbor_tri=triangle_neighbors(faces,npt)

  neighbor_verts= np.array([get_surf_neighbors(faces,neighbor_tri, k)
                                   for k in range(npt)])

  e0=vertices[faces[:,2]]-vertices[faces[:,1]]
  e1=vertices[faces[:,0]]-vertices[faces[:,2]]
  e2=vertices[faces[:,1]]-vertices[faces[:,0]]

  e0_norm=normr(e0)
  e1_norm=normr(e1)
  e2_norm=normr(e2)

  FaceNormals=0.5*fastcross(e0,e1)
  VN=GetVertexNormals(vertices,faces,FaceNormals,e0,e1,e2)
  up        = np.zeros(vertices.shape)
  #Calculate initial coordinate system
  up[faces[:,0]]=e2_norm
  up[faces[:,1]]=e0_norm
  up[faces[:,2]]=e1_norm

  #Calculate initial vertex coordinate system
  up=fastcross(VN,up)
  up=normr(up)
  vp=fastcross(up,VN)
  vp=normr(vp)

  qj=np.zeros([12,5])
  A =np.zeros([36,7])
  B =np.zeros([36,1])

  H=np.zeros(npt)
  K=np.zeros(npt)

  for i in range(npt):
# local coordinate basis
    n1=up[i]
    n2=vp[i]
    n3=VN[i]

    for j,(pj,nj) in enumerate(zip(vertices[neighbor_verts[i]],VN[neighbor_verts[i]])):
      qj[j]=np.array([np.dot(pj-vertices[i],n1),
                      np.dot(pj-vertices[i],n2),
                      np.dot(pj-vertices[i],n3),
                     -np.dot(nj,n1)/np.dot(nj,n3),
                     -np.dot(nj,n2)/np.dot(nj,n3)])

    j=0
    k=0

    for (x,y,z,nx,ny) in qj:
      k+=1
      if k==len(neighbor_verts[i]):
         break
      scale=2.0/(x**2+y**2)
      A[j]   =scale*np.array([0.5*x**2, x*y, 0.5*y**2,   x**3, x**2*y, x*y**2,   y**3])
      A[j+1] =scale*np.array([       x,   y,        0, 3*x**2,  2*x*y,   y**2,      0])
      A[j+2] =scale*np.array([       0,   x,        y,      0,   x**2,  2*x*y, 3*y**2])
      B[j]   =scale*z
      B[j+1] =scale*nx
      B[j+2] =scale*ny
      j+=3

    X=lstsq(A[:3*len(neighbor_verts[i])-3,:],B[:3*len(neighbor_verts[i])-3],rcond=None)
    a=0.5*X[0][0]
    b=    X[0][1]
    c=0.5*X[0][2]
    H[i] = -(a+c)
    K[i] = 4*a*c-b**2

  return K,H,VN