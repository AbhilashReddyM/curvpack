import numpy as np

#  The first two functions are modified from MNE surface project. LIcense follows
#  This software is OSI Certified Open Source Software. OSI Certified is a certification mark of the Open Source Initiative.
#  
#  Copyright (c) 2011-2019, authors of MNE-Python. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#  
#  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  Neither the names of MNE-Python authors nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#  This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

def triangle_neighbors(tris, npts):
    """Efficiently compute vertex neighboring triangles.
    Returns the triangles in the 1-ring of a given vertex
    """
    # this code replaces the following, but is faster (vectorized):
    #
    # this['neighbor_tri'] = [list() for _ in xrange(this['np'])]
    # for p in xrange(this['ntri']):
    #     verts = this['tris'][p]
    #     this['neighbor_tri'][verts[0]].append(p)
    #     this['neighbor_tri'][verts[1]].append(p)
    #     this['neighbor_tri'][verts[2]].append(p)
    # this['neighbor_tri'] = [np.array(nb, int) for nb in this['neighbor_tri']]
    #
    verts = tris.ravel()
    counts = np.bincount(verts, minlength=npts)
    reord = np.argsort(verts)
    tri_idx = np.unravel_index(reord, (len(tris), 3))[0]
    idx = np.cumsum(np.r_[0, counts])
    # the sort below slows it down a bit, but is needed for equivalence
    neighbor_tri = np.array([np.sort(tri_idx[v1:v2])
                    for v1, v2 in zip(idx[:-1], idx[1:])])
    return neighbor_tri

def get_surf_neighbors(tris,neighbor_tri, k):
    """Get vertices of 1-ring
    """
    verts = tris[neighbor_tri[k]]
    verts = np.setdiff1d(verts, [k], assume_unique=False)
    nneighbors = len(verts)
    return verts

def GetVertexNormals(vertices,faces,FaceNormals,e0,e1,e2):
    """
    INPUT:
    Vertices       : vertices
    Faces          : vertex connectivity
    FaceNormals    :     Outer Normal per face, having magnitude equal to area of face
    e0,e1,e2       : edge vectors

    OUTPUT:
    VertNormals    :       Unit normal at the vertex
    """

    VertNormals   =np.zeros(vertices.shape)

    #edge lengths
    de0=np.sqrt(e0[:,0]**2+e0[:,1]**2+e0[:,2]**2)
    de1=np.sqrt(e1[:,0]**2+e1[:,1]**2+e1[:,2]**2)
    de2=np.sqrt(e2[:,0]**2+e2[:,1]**2+e2[:,2]**2)

    L2=np.c_[de0**2,de1**2,de2**2]

    #Calculate weights according to N.Max [1999] for normals
    wfv1=FaceNormals/(L2[:,1]*L2[:,2])[:,np.newaxis]
    wfv2=FaceNormals/(L2[:,2]*L2[:,0])[:,np.newaxis]
    wfv3=FaceNormals/(L2[:,0]*L2[:,1])[:,np.newaxis]

#    #Calculate the weights according to MWA for normals
#    wfv1=FaceNormals*np.arcsin(2*Af/(de1*de2))[:,np.newaxis]
#    wfv2=FaceNormals*np.arcsin(2*Af/(de2*de0))[:,np.newaxis]
#    wfv3=FaceNormals*np.arcsin(2*Af/(de0*de1))[:,np.newaxis]


    verts=faces.T[0]
    for j in [0,1,2]:
      VertNormals[:,j]+=np.bincount(verts,minlength=vertices.shape[0],weights=wfv1[:,j])
    verts=faces.T[1]
    for j in [0,1,2]:
      VertNormals[:,j]+=np.bincount(verts,minlength=vertices.shape[0],weights=wfv2[:,j])
    verts=faces.T[2]
    for j in [0,1,2]:
      VertNormals[:,j]+=np.bincount(verts,minlength=vertices.shape[0],weights=wfv3[:,j])

    VertNormals=normr(VertNormals)

    return VertNormals

def fastcross(x, y):
    """Compute cross product between list of 3D vectors
    Input
    x       :  Mx3 array
    y       :  Mx3 array

    Output
    z       : Mx3 array Cross product of x and y.
    """
    if max([x.shape[0], y.shape[0]]) >= 500:
        return np.c_[x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1],
                        x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2],
                        x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]]
    else:
        return np.cross(x, y)

def normr(vec):
    """
    Normalizes an array of vectors. e.g. to convert a np array of vectors to unit vectors
    """
    return vec/np.sqrt((vec**2).sum(axis=1))[:,np.newaxis]
