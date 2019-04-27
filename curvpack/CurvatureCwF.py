
"""
Abhilash Reddy Malipeddi. January 2017
Calculate the mean and gaussian curvature and LaplaceBeltrami at a
vertex in a tri mesh using a fitting method as given in
[Guoliang Xu ANM 2013],[Farutin et. al. JCP 2014], etc.

The first two functions are modified from MNE surface project

"""

import numpy as np
from .utils import  triangle_neighbors,GetVertexNormals,get_surf_neighbors,fastcross,normr

def CurvatureCwF(vertices,faces,opt=1,want_LB=False):

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

  A  =np.zeros([30,5])
  B  =np.zeros([30,3])

  H =np.zeros(npt)
  K =np.zeros(npt)
  LB=np.zeros(npt)

  metric  = np.zeros([npt,2,2])
  hessx   = np.zeros([npt,2,2])
  hessy   = np.zeros([npt,2,2])
  hessz   = np.zeros([npt,2,2])
  drdu    = np.zeros([npt,3])
  drdv    = np.zeros([npt,3])

  for i in range(npt):
    n1=up[i]
    n2=vp[i]

    if(opt==2 or len(neighbor_verts[i])<5 ):
    # 2-ring
      nbrs=np.unique(np.hstack(neighbor_verts[neighbor_verts[i]].flat))
      nbrs=np.setdiff1d(nbrs,i)
    else:
      nbrs=neighbor_verts[i]

    for j,pj in enumerate(vertices[nbrs]):
      u,v = np.dot(pj-vertices[i],n1),np.dot(pj-vertices[i],n2)
      scale=1.0/np.sqrt(u**2+v**2)
      A[j,:]   = scale*np.array([u, v, 0.5*u**2, u*v, 0.5*v**2])
      B[j,:]   = scale*(pj-vertices[i])

    X=np.linalg.pinv(A[:j+1,:])
    X2=X.dot(B[:j+1,:])
    cx=X2[:,0]
    cy=X2[:,1]
    cz=X2[:,2]
    dfdu=np.array([cx[0],cy[0],cz[0]])
    dfdv=np.array([cx[1],cy[1],cz[1]])
    drdu[i,:]=dfdu
    drdv[i,:]=dfdv

    VN[i]=np.cross(dfdv,dfdu)
    VN[i]/=np.sqrt(VN[i,0]**2+VN[i,1]**2+VN[i,2]**2)

    n2=np.cross(n1,VN[i])
    n2=n2/np.linalg.norm(n2)

    n1=np.cross(VN[i],n2)
    n1=n1/np.linalg.norm(n1)

    up[i]=n1
    vp[i]=n2

    E,F,G=dfdu.dot(dfdu),dfdu.dot(dfdv),dfdv.dot(dfdv)
    metric[i,:,:]=np.array([[E,F],[F,G]]).squeeze()
    hessx[i,:,:]=np.array([[cx[2],cx[3]],[cx[3],cx[4]]]).squeeze()
    hessy[i,:,:]=np.array([[cy[2],cy[3]],[cy[3],cy[4]]]).squeeze()
    hessz[i,:,:]=np.array([[cz[2],cz[3]],[cz[3],cz[4]]]).squeeze()

    L,M,N=[-VN[i].dot(np.array([cx[2],cy[2],cz[2]])),
           -VN[i].dot(np.array([cx[3],cy[3],cz[3]])),
           -VN[i].dot(np.array([cx[4],cy[4],cz[4]]))]
    H[i]  = (E*N+G*L-2*F*M)/(2*(E*G-F**2))
    K[i]  = (L*N-M**2)/(E*G-F**2)

  if(want_LB):
  #Laplace Beltrami from fitting
    A  = np.zeros([30,5])
    B  = np.zeros([30,1])
  
    #repeat the fitting because the local coordinate system is is different now
    for i in range(npt):
      n1=up[i]
      n2=vp[i]
  
      if(opt==2 or len(neighbor_verts[i])<5 ):
      # 2-ring
        nbrs=np.unique(np.hstack(neighbor_verts[neighbor_verts[i]].flat))
        nbrs=np.setdiff1d(nbrs,i)
      else:
        nbrs=neighbor_verts[i]
  
      for j,(pj,hj) in enumerate(zip(vertices[nbrs],H[nbrs])):
        u,v      = np.dot(pj-vertices[i],n1),np.dot(pj-vertices[i],n2)
        scale    = 1.0/np.sqrt(u**2+v**2)
        A[j,:]   = scale*np.array([u, v, 0.5*u**2, u*v, 0.5*v**2])
        B[j]     = scale*(hj-H[i])
  
      X3=np.linalg.pinv(A[:j+1,:])
      X4=X3.dot(B[:j+1])
  
      hessh=np.array([[X4[2],X4[3]],[X4[3],X4[4]]]).squeeze()
      hu=X4[0]
      hv=X4[1]
      ginv=np.linalg.inv(metric[i])
  
      LB[i]=np.sum(hessh*ginv)
  
      temp=-np.sum(ginv*hessx[i])*np.sum(ginv*np.array([[hu*drdu[i,0],hv*drdu[i,0]],
                                                        [hu*drdv[i,0],hv*drdv[i,0]]]).squeeze())
      LB[i]+=temp
      temp=-np.sum(ginv*hessy[i])*np.sum(ginv*np.array([[hu*drdu[i,1],hv*drdu[i,1]],
                                                        [hu*drdv[i,1],hv*drdv[i,1]]]).squeeze())
      LB[i]+=temp
      temp=-np.sum(ginv*hessz[i])*np.sum(ginv*np.array([[hu*drdu[i,2],hv*drdu[i,2]],
                                                        [hu*drdv[i,2],hv*drdv[i,2]]]).squeeze())
      LB[i]+=temp
    return  K,H,VN,LB
  return K,H,VN
