'''
    Abhilash Reddy Malipeddi, December 2015.
    Laplace Beltrami operator on a function sampled at the vertices using the direct 
    discretization via Gauss divergence theorem from the paper Xu 2004[1].
    This scheme is named  $nabla^D_S$ in the paper
    [1]Xu, Guoliang. "Convergent discrete laplace-beltrami operators over
    triangular surfaces." Geometric Modeling and Processing, 2004. Proceedings. IEEE, 2004.
'''
import numpy
import scipy.sparse
import math

def GetTopologyAroundVertex(vertices,faces):
    '''
    Calculates the topology around the vertices:
    Returns
    1.  vertex_surr_faces   :  indices of the faces that surround each vertex.
    2.  vertex_across_edge  :  for vertex 'i' and triangle 'j' gives vertex 'k'
    3.  face_across_edge    :  for vertex 'i' and triangle 'j' gives face 'f' -- refer to fig 
    #
    #                                i
    #                               / \
    #                              / j \
    #                             /_____\
    #                             \     /
    #                              \ f /
    #                               \ / 
    #                                k
    '''
    MaxValence=15
    npt=vertices.shape[0]
    vertex_across_edge  =-numpy.ones([npt,MaxValence],dtype=int)
    face_across_edge    =-numpy.ones([npt,MaxValence],dtype=int)   
    vertex_surr_faces   =-numpy.ones([npt,MaxValence],dtype=int)   # index of the faces that surrounds each vertex
    vertexValence       = numpy.zeros([npt,1],dtype=int)


    FaceNeighbors=GetFaceNeighbors(vertices,faces)

#This loop will error if the valency becomes greater than MaxValence
    for ke,f in enumerate(faces):
        vertexValence[f]+=1 # valence/number of neighbours of f'th point
        # store the index of the elements surrounding each vertex pointed to by f
        vertex_surr_faces[f[0],vertexValence[f[0]]-1]=ke
        vertex_surr_faces[f[1],vertexValence[f[1]]-1]=ke
        vertex_surr_faces[f[2],vertexValence[f[2]]-1]=ke

    #vertex across edge
    for i,vsf in enumerate(vertex_surr_faces):
# vertex_surr_faces is initialized with -1 throughout. The actual count of faces will usually
# be different for different vertices. There is no way to know this count before hand.
# The below line keeps only face indices that are greater than -1, which are what 
# actually exist.
      vsf=vsf[vsf>-1]
      for j,vj in enumerate(vsf):
         for f in FaceNeighbors[vj]:
            t=faces[f]
            if t[0]!=i and t[1]!=i and t[2]!=i:
               vertex_across_edge[i,j]=numpy.setdiff1d(t,faces[vj])
               face_across_edge[i,j]=f
    return vertex_surr_faces,vertex_across_edge,face_across_edge

def GetFaceNeighbors(vertices,faces):
    '''
    build neighbouring element information: method 1
    '''
    n2e=scipy.sparse.lil_matrix((vertices.shape[0],faces.shape[0]),dtype=int)
    FaceNeighbor=numpy.zeros(faces.shape,dtype=int)

#build adjcency matrix
    for i,t in enumerate(faces):
        n2e[t,i]=numpy.ones([3,1],dtype=int)

    n2e=n2e.tocsr()
    for i,t in enumerate(faces):
      nb=numpy.intersect1d(numpy.nonzero(n2e[t[1]])[1],numpy.nonzero(n2e[t[2]])[1])
      nb=numpy.setdiff1d(nb,i)
      if nb.shape[0]==1:
          FaceNeighbor[i,0]=nb[0]
      nb=numpy.intersect1d(numpy.nonzero(n2e[t[2]])[1],numpy.nonzero(n2e[t[0]])[1])
      nb=numpy.setdiff1d(nb,i)
      if nb.shape[0]==1:
          FaceNeighbor[i,1]=nb[0]
      nb=numpy.intersect1d(numpy.nonzero(n2e[t[0]])[1],numpy.nonzero(n2e[t[1]])[1])
      nb=numpy.setdiff1d(nb,i)
      if nb.shape[0]==1:
          FaceNeighbor[i,2]=nb[0]
    return FaceNeighbor

def DirectDiscreteLB(vertices,faces,inpfunc):
    '''
    Laplace Beltrami operator of a function sampled at the vertices using the direct 
    discretization via gauss formula from "Convergent Discrete Laplace-Beltrami 
    Operators over Triangular Surfaces" by Guoliang Xu
    INPUTS:
    - vertices           :  coordinates of the vertices of the meshj
    - faces              :  vertex number of each face
    - inpfunc            :  a function that is sampled at the vertices of the mesh
    - vertex_surr_faces  :  indices of the faces that surround each vertex.
    - face_across_edge   :  for vertex 'i' and triangle 'j' gives face 'f' -- refer to fig 
    - vertex_across_edge :  for vertex 'i' and triangle 'j' gives vertex 'k'
    #                                i
    #                               / \
    #                              / j \
    #                             /_____\
    #                             \     /
    #                              \ f /
    #                               \ / 
    #                                k
    '''
    LB=numpy.zeros(inpfunc.shape)
    vertex_surr_faces,vertex_across_edge,face_across_edge=GetTopologyAroundVertex(vertices,faces)

    #calculate the area of each triangle in the mesh
    e1=vertices[faces[:,1]]-vertices[faces[:,0]]
    e2=vertices[faces[:,2]]-vertices[faces[:,0]]
    Area=numpy.cross(e1,e2)
    Area=0.5*numpy.sqrt((Area*Area).sum(axis=1))[:,numpy.newaxis]

    for i,pi in enumerate(vertices):
      fi=inpfunc[i]
      vsf=vertex_surr_faces[i]
      vsf=vsf[vsf>-1]
      Lsum=0.
      AP=0.
      for j,vj in enumerate(vsf):
        Aj=Area[vj]
        Adj=Area[face_across_edge[i,j]]

        tj,tjp=numpy.setdiff1d(faces[vj],i)#vertices that are not 'i'
        pj =vertices[tj]
        pjp=vertices[tjp]
        pdj=vertices[vertex_across_edge[i,j]]

        fj =inpfunc[tj]
        fjp=inpfunc[tjp]
        fdj=inpfunc[vertex_across_edge[i,j]]

        nj     = -(0.5/Aj )*( numpy.dot(pi-pj,pj-pjp) *(pjp-pi)  + numpy.dot(pi-pjp,pjp-pj) *(pj-pi)  )
        ndj    = -(0.5/Adj)*( numpy.dot(pdj-pj,pj-pjp)*(pjp-pdj) + numpy.dot(pdj-pjp,pjp-pj)*(pj-pdj) )
        fbarj  = -(0.5/Aj )*( numpy.dot(pi-pj,pj-pjp) *(fjp-fi)  + numpy.dot(pi-pjp,pjp-pj) *(fj-fi)  )
        fbardj = -(0.5/Adj)*( numpy.dot(pdj-pj,pj-pjp)*(fjp-fdj) + numpy.dot(pdj-pjp,pjp-pj)*(fj-fdj) )
        normp  = numpy.sqrt(numpy.sum((pj-pjp)**2))
        normn  = numpy.sqrt(numpy.sum((nj-ndj)**2))
        Lsum  += (fbarj-fbardj)*normp/normn
        AP    += Aj
      LB[i]=Lsum/AP
    return LB