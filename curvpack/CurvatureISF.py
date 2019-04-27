
"""
Abhilash Reddy Malipeddi. January 2017
Calculate the mean and gaussian curvature at a vertex in a tri mesh using
using an iterative fitting method similar to what is given in [Garimella and Swartz],
[Yazdani and Bagchi], etc.

"""

import numpy as np
from numpy.linalg import lstsq
from .utils import  triangle_neighbors,GetVertexNormals,get_surf_neighbors,fastcross,normr


def CurvatureISF1(vertices,faces):
  '''
  This uses a two-ring neighborhood around a point.  
  '''
  tol=1e-10
  
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

  qj=np.zeros([30,3])
  A =np.zeros([36,5])
  B =np.zeros([36,1])

  H=np.zeros(npt)
  K=np.zeros(npt)


  for i in range(npt):
    n1=up[i]
    n2=vp[i]
    n3=VN[i]

    nbrs=np.unique(np.hstack(neighbor_verts[neighbor_verts[i]].flat))
    nbrs=np.setdiff1d(nbrs,i)

    for _ in range(30):
      for j,pj in enumerate(vertices[nbrs]):
        qj[j]=np.array([np.dot(pj-vertices[i],n1),
                        np.dot(pj-vertices[i],n2),
                        np.dot(pj-vertices[i],n3)])
      j=0
      k=0
      for (x,y,z) in qj:
        k+=1
        if k==len(nbrs):
           break
        scale  = 2/(x**2+y**2)
        A[j]   = scale*np.array([ x**2, x*y, y**2, x, y])
        B[j]   = scale*z
        j+=1

      X=lstsq(A[:len(nbrs),:],B[:len(nbrs)],rcond=None)

      a,b,c,d,e=X[0]

      factor=1.0/np.sqrt(1.0+d[0]**2+e[0]**2)
      oldn3=n3.copy()
      n3=factor*np.array([-d[0],-e[0],1.0])

      n3=np.c_[n1,n2,oldn3].dot(n3)#new normal in local coordinates
      VN[i]=n3                     #new normal in global coordinates. up,vp,VN system is not orthogonal anymore, but that is okay as it is not used again
      n2=np.cross(n1,n3)
      n2=n2/np.linalg.norm(n2)
      n1=np.cross(n3,n2)
      n1=n1/np.linalg.norm(n1)

      H[i]=factor**3*(a+c+a*e**2+c*d**2-b*d*e)
      K[i]=factor**4*(4*a*c-b**2)
      if np.linalg.norm(n3-oldn3) <tol:
          break
  return K,-H,VN

def CurvatureISF2(vertices,faces):
  '''
  This is a slight modification of the previous. Here we only use the one ring
  but we include the vertex normals in the fitting procedure. This indirectly has
  two ring support because the vertex normals themselves are calculated
  as a weighted average of the face normals. Sidenote: I wonder what happens if we include both
  vertex and face normals in the fitting procedure....
  '''
  tol=1e-10
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
  A =np.zeros([36,5])
  B =np.zeros([36,1])

  H=np.zeros(npt)
  K=np.zeros(npt)
  VNnew=np.zeros_like(VN)

  for i in range(npt):
    n1=up[i]
    n2=vp[i]
    n3=VN[i]
    for iter in range(30):
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
        scale=2/(x**2+y**2)
        A[j]   = scale*np.array([ x**2, x*y, y**2, x, y])
        A[j+1] = scale*np.array([  2*x,   y,    0, 1, 0])
        A[j+2] = scale*np.array([    0,   x,  2*y, 0, 1])
        B[j]   = scale*z
        B[j+1] = scale*nx
        B[j+2] = scale*ny
        j+=3

      X=lstsq(A[:3*len(neighbor_verts[i]),:],B[:3*len(neighbor_verts[i])],rcond=None)
      a,b,c,d,e=X[0]
      factor=1.0/np.sqrt(1.0+d[0]**2+e[0]**2)
      H[i]=factor**3*(a+c+a*e**2+c*d**2-b*d*e)
      K[i]=factor**4*(4*a*c-b**2)

      oldn3=n3.copy()
      n3=factor*np.array([-d[0],-e[0],1.0])#new normal in local coordinates
      n3=np.c_[n1,n2,oldn3].dot(n3)        #new normal in global coordinates
      n2=np.cross(n1,n3)
      n2=n2/np.linalg.norm(n2)
      n1=np.cross(n3,n2)
      n1=n1/np.linalg.norm(n1)


      if np.linalg.norm(n3-oldn3) <tol:
          up[i]=n1
          vp[i]=n2
          VN[i]=n3
          break
  return K,-H,VN

def CurvatureISF3(vertices,faces):
  '''
  Uses two ring vertices and normals.
  '''
  tol=1e-10
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

  qj=np.zeros([100,5])
  A =np.zeros([200,5])
  B =np.zeros([200,1])

  H=np.zeros(npt)
  K=np.zeros(npt)
  VNnew=np.zeros_like(VN)

  for i in range(npt):
    n1=up[i]
    n2=vp[i]
    n3=VN[i]
    nbrs=np.unique(np.hstack(neighbor_verts[neighbor_verts[i]].flat))
    nbrs=np.setdiff1d(nbrs,i)

    for iter in range(30):
      for j,(pj,nj) in enumerate(zip(vertices[nbrs],VN[nbrs])):
        qj[j]=np.array([np.dot(pj-vertices[i],n1),
                        np.dot(pj-vertices[i],n2),
                        np.dot(pj-vertices[i],n3),
                       -np.dot(nj,n1)/np.dot(nj,n3),
                       -np.dot(nj,n2)/np.dot(nj,n3)])
      j=0
      k=0
      for (x,y,z,nx,ny) in qj:
        k+=1
        if k==len(nbrs):
           break
        scale=2/(x**2+y**2)
        A[j]   = scale*np.array([ x**2, x*y, y**2, x, y])
        A[j+1] = scale*np.array([  2*x,   y,    0, 1, 0])
        A[j+2] = scale*np.array([    0,   x,  2*y, 0, 1])
        B[j]   = scale*z
        B[j+1] = scale*nx
        B[j+2] = scale*ny
        j+=3

      X=lstsq(A[:3*len(nbrs),:],B[:3*len(nbrs)],rcond=None)
      a,b,c,d,e=X[0]
      factor=1.0/np.sqrt(1.0+d[0]**2+e[0]**2)
      H[i]=factor**3*(a+c+a*e**2+c*d**2-b*d*e)
      K[i]=factor**4*(4*a*c-b**2)

      oldn3=n3.copy()
      n3=factor*np.array([-d[0],-e[0],1.0])#new normal in local coordinates
      n3=np.c_[n1,n2,oldn3].dot(n3)        #new normal in global coordinates
      n2=np.cross(n1,n3)
      n2=n2/np.linalg.norm(n2)
      n1=np.cross(n3,n2)
      n1=n1/np.linalg.norm(n1)

      if np.linalg.norm(n3-oldn3) <tol:
          up[i]=n1
          vp[i]=n2
          VN[i]=n3
          break
  return K,-H,VN
