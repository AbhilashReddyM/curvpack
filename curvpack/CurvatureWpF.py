"""
Abhilash Reddy Malipeddi
Calculation of curvature using the method outlined in Szymon Rusinkiewicz et. al 2004
Per face curvature is calculated and per vertex curvature is calculated by weighting the
per-face curvatures. I have vectorized the code where possible.
"""

import numpy as np

from numpy.core.umath_tests import inner1d
from .utils import  fastcross,normr

def RotateCoordinateSystem(up,vp,nf):
    """
    RotateCoordinateSystem performs the rotation of the vectors up and vp
    to the plane defined by nf
    INPUT:
      up,vp  - vectors to be rotated (vertex coordinate system)
      nf     - face normal
    OUTPUT:
      r_new_u,r_new_v - rotated coordinate system
    """
    nrp=np.cross(up,vp)
    nrp=nrp/np.sqrt(nrp[0]**2+nrp[1]**2+nrp[2]**2)
    ndot=nf[0]*nrp[0]+nf[1]*nrp[1]+nf[2]*nrp[2]
    if ndot<=-1:
        return -up,-vp
    perp=nf-ndot*nrp
    dperp=(nrp+nf)/(1.0+ndot)
    r_new_u=up-dperp*(perp[0]*up[0]+perp[1]*up[1]+perp[2]*up[2])
    r_new_v=vp-dperp*(perp[0]*vp[0]+perp[1]*vp[1]+perp[2]*vp[2])
    return r_new_u,r_new_v

def ProjectCurvatureTensor(uf,vf,nf,old_ku,old_kuv,old_kv,up,vp):
    """
    ProjectCurvatureTensor performs a projection
    of the tensor variables to the vertexcoordinate system
    INPUT:
        uf,vf                 - face coordinate system
        old_ku,old_kuv,old_kv - face curvature tensor variables
        up,vp                 - vertex cordinate system
    OUTPUT:
        new_ku,new_kuv,new_kv - vertex curvature tensor variables
    """
    r_new_u,r_new_v = RotateCoordinateSystem(up,vp,nf)
    u1=r_new_u[0]*uf[0]+r_new_u[1]*uf[1]+r_new_u[2]*uf[2]
    v1=r_new_u[0]*vf[0]+r_new_u[1]*vf[1]+r_new_u[2]*vf[2]
    u2=r_new_v[0]*uf[0]+r_new_v[1]*uf[1]+r_new_v[2]*uf[2]
    v2=r_new_v[0]*vf[0]+r_new_v[1]*vf[1]+r_new_v[2]*vf[2]
    new_ku  = u1*(u1*old_ku+v1*old_kuv) + v1*(u1*old_kuv+v1*old_kv )
    new_kuv = u2*(u1*old_ku+v1*old_kuv) + v2*(u1*old_kuv+v1*old_kv )
    new_kv  = u2*(u2*old_ku+v2*old_kuv) + v2*(u2*old_kuv+v2*old_kv )
    return new_ku,new_kuv,new_kv

def GetVertexNormalsExtra(vertices,faces,FaceNormals,e0,e1,e2):
    """
    In addition to vertex normals this also returns the mixed area weights per vertex
    which is used in calculating the curvature at the vertex from per face curvature values
    We could have calculated them separetely, but doing both at once is efficient. 
    The calculations involve loops over the faces and vertices in serial and are not easily vectorized 
    INPUT:
    Vertices       : vertices
    Faces          : vertex connectivity
    FaceNormals    : Outer Normal per face, having magnitude equal to area of face
    e0,e1,e2       : edge vectors

    OUTPUT:
    VertNormals    :       Unit normal at the vertex
    wfp            :  Mixed area weights per vertex, as per Meyer 2002

    OTHER:
    Avertex        :     Mixed area associated with a vertex. Meyer 2002
    Acorner        :     part of Avertex associated to
    """

    #edge lengths
    de0=np.sqrt(e0[:,0]**2+e0[:,1]**2+e0[:,2]**2)
    de1=np.sqrt(e1[:,0]**2+e1[:,1]**2+e1[:,2]**2)
    de2=np.sqrt(e2[:,0]**2+e2[:,1]**2+e2[:,2]**2)

    L2=np.c_[de0**2,de1**2,de2**2]

    ew=np.c_[L2[:,0]*(L2[:,1]+L2[:,2]-L2[:,0]),L2[:,1]*(L2[:,2]+L2[:,0]-L2[:,1]),L2[:,2]*(L2[:,0]+L2[:,1]-L2[:,2])]

    #calculate face area
    Af=np.sqrt(FaceNormals[:,0]**2+FaceNormals[:,1]**2+FaceNormals[:,2]**2)

    Avertex       =np.zeros(vertices.shape[0])
    VertNormals   =np.zeros(vertices.shape)

    #Calculate weights according to N.Max [1999] for normals
    wfv1=FaceNormals/(L2[:,1]*L2[:,2])[:,np.newaxis]
    wfv2=FaceNormals/(L2[:,2]*L2[:,0])[:,np.newaxis]
    wfv3=FaceNormals/(L2[:,0]*L2[:,1])[:,np.newaxis]

    verts=faces.T[0]
    for j in [0,1,2]:
      VertNormals[:,j]+=np.bincount(verts,minlength=vertices.shape[0],weights=wfv1[:,j])
    verts=faces.T[1]
    for j in [0,1,2]:
      VertNormals[:,j]+=np.bincount(verts,minlength=vertices.shape[0],weights=wfv2[:,j])
    verts=faces.T[2]
    for j in [0,1,2]:
      VertNormals[:,j]+=np.bincount(verts,minlength=vertices.shape[0],weights=wfv3[:,j])

    Acorner=(0.5*Af/(ew[:,0]+ew[:,1]+ew[:,2]))[:,np.newaxis]*np.c_[ew[:,1]+ew[:,2], ew[:,2]+ew[:,0], ew[:,0]+ew[:,1]]

    #Change the area to barycentric area for obtuse triangles
    for i,f in enumerate(faces):
        if ew[i,0]<=0:
            Acorner[i,2]=-0.25*L2[i,1]*Af[i]/(sum(e0[i]*e1[i]))
            Acorner[i,1]=-0.25*L2[i,2]*Af[i]/(sum(e0[i]*e2[i]))
            Acorner[i,0]=Af[i]-Acorner[i,1]-Acorner[i,2]
        elif ew[i,1]<=0:
            Acorner[i,2]=-0.25*L2[i,0]*Af[i]/(sum(e1[i]*e0[i]))
            Acorner[i,0]=-0.25*L2[i,2]*Af[i]/(sum(e1[i]*e2[i]))
            Acorner[i,1]=Af[i]-Acorner[i,0]-Acorner[i,2]
        elif ew[i,2]<=0:
            Acorner[i,0]=-0.25*L2[i,1]*Af[i]/(sum(e2[i]*e1[i]))
            Acorner[i,1]=-0.25*L2[i,0]*Af[i]/(sum(e2[i]*e0[i]))
            Acorner[i,2]=Af[i]-Acorner[i,0]-Acorner[i,1]

#Accumulate Avertex from Acorner.
    for j,verts in enumerate(faces.T):
       Avertex+=np.bincount(verts,minlength=vertices.shape[0],weights=Acorner[:,j])
    VertNormals=normr(VertNormals)

    #calculate voronoi weights
    wfp=Acorner/Avertex[faces]

    return VertNormals,wfp

def CalcCurvature(vertices,faces):
    """
    CalcCurvature recives a list of vertices and faces
    and the normal at each vertex and calculates the second fundamental
    matrix and the curvature by least squares, by inverting the 3x3 Normal matrix
    INPUT:
    vertices  -nX3 array of vertices
    faces     -mX3 array of faces
    VertexNormals - nX3 matrix (n=number of vertices) containing the normal at each vertex
    FaceNormals - mX3 matrix (m = number of faces) containing the normal of each face
    OUTPUT:
    FaceSFM   - a list of 2x2 np arrays of (m = number of faces) second fundamental tensor at the faces
    VertexSFM - a list of 2x2 np arrays (n = number of vertices) second fundamental tensor at the vertices

    Other Parameters
    wfp     : mx3 array of vertex voronoi cell area/Mixed area weights as given in Meyer 2002
    up,vp   : local coordinate system at each vertex
    e0,e1,e2       : edge vectors
    """
    #list of 2x2 arrays for each vertex
    VertexSFM = [np.zeros([2,2]) for i in vertices]
    up        = np.zeros(vertices.shape)

    e0=vertices[faces[:,2]]-vertices[faces[:,1]]
    e1=vertices[faces[:,0]]-vertices[faces[:,2]]
    e2=vertices[faces[:,1]]-vertices[faces[:,0]]

    e0_norm=normr(e0)
    e1_norm=normr(e1)
    e2_norm=normr(e2)

    FaceNormals=0.5*fastcross(e1,e2) #not unit length. holds the area which is needed next
    VertNormals,wfp=GetVertexNormalsExtra(vertices,faces,FaceNormals,e0,e1,e2)

    FaceNormals=normr(FaceNormals)

    #Calculate initial coordinate system
    up[faces[:,0]]=e2_norm
    up[faces[:,1]]=e0_norm
    up[faces[:,2]]=e1_norm

    #Calculate initial vertex coordinate system
    up=fastcross(up,VertNormals)
    up=normr(up)
    vp=fastcross(VertNormals,up)

    B=normr(fastcross(FaceNormals,e0_norm))

    nfaces=faces.shape[0]

# Build a least square problem at each face to get the SFM at each face and solve it using the normal equation
    scale=1.0/np.sqrt(np.sum((e0[0,:]**2+e1[0,:]**2+e2[0,:]**2)/3.0))
    AT  =  scale*np.array([[inner1d(e0,e0_norm), inner1d(e0,B), np.zeros(nfaces)],
                        [np.zeros(nfaces), inner1d(e0,e0_norm), inner1d(e0,B)],
                        [inner1d(e1,e0_norm), inner1d(e1,B), np.zeros(nfaces)],
                        [np.zeros(nfaces), inner1d(e1,e0_norm), inner1d(e1,B)],
                        [inner1d(e2,e0_norm), inner1d(e2,B), np.zeros(nfaces)],
                        [np.zeros(nfaces), inner1d(e2,e0_norm), inner1d(e2,B)]]).T

    A  = np.transpose(AT,axes=(0,2,1)).copy()

    dn0=VertNormals[faces[:,2]]-VertNormals[faces[:,1]]
    dn1=VertNormals[faces[:,0]]-VertNormals[faces[:,2]]
    dn2=VertNormals[faces[:,1]]-VertNormals[faces[:,0]]

    b=  scale*np.array([inner1d(dn0,e0_norm),
                        inner1d(dn0,B      ),
                        inner1d(dn1,e0_norm),
                        inner1d(dn1,B      ),
                        inner1d(dn2,e0_norm),
                        inner1d(dn2,B      )]).T[:,:,np.newaxis]

    X1=np.array([np.linalg.pinv(a,-1) for a in A])
    X   = np.matmul(X1,b)

#now calculate curvature per vertex as weighted sum of the face curvature
    for i,f in enumerate(faces):
      for j in [0,1,2]:
       new_ku,new_kuv,new_kv = ProjectCurvatureTensor(e0_norm[i],B[i],FaceNormals[i],X[i][0],X[i][1],X[i][2],up[f[j]],vp[f[j]])
       VertexSFM[f[j]]+=wfp[i,j]*np.array([[new_ku,new_kuv],[new_kuv,new_kv]]).squeeze()
    return VertexSFM,VertNormals

def GetCurvatures(vertices,faces):
    """
    INPUT : vertices,faces
    OUTPUT: Gaussian Curvature, Mean Curvature
    """

    VertexSFM,VertNormals=CalcCurvature(vertices,faces)

    ku       =np.array([VSFM[0,0] for VSFM in VertexSFM])
    kuv      =np.array([VSFM[0,1] for VSFM in VertexSFM])
    kv       =np.array([VSFM[1,1] for VSFM in VertexSFM])

    return (ku*kv-kuv**2),0.5*(ku+kv),VertNormals

