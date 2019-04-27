'''
=======================================================================
 2016 Changes by ARM (abhilashreddy.com)
  - made to work with Python 3+
  - made to work with recent versions of matplotlib
=======================================================================

 Author: William G.K. Martin (wgm2111@cu where cu=columbia.edu)
 copyright (c) 2010
 liscence: BSD style

 ======================================================================
'''
import numpy as np
import matplotlib.tri as tri

def get_points():
    '''
    Create the coordinates of the vertices of the base icosahedron

    '''
    # Define the verticies with the golden ratio
    a = (1. + np.sqrt(5.))/2.0   # golden ratio
    p = np.array([[a,-a,-a, a, 1, 1,-1,-1, 0, 0, 0, 0],
                  [0, 0, 0, 0, a,-a,-a, a, 1, 1,-1,-1],
                  [1, 1,-1,-1, 0, 0, 0, 0, a,-a,-a, a]]).transpose()
    p = p / np.sqrt((p**2).sum(1))[0]

    # rotate top point to the z-axis
    ang = np.arctan(p[0,0] / p[0,2])
    ca, sa = np.cos(ang), np.sin(ang)
    rotation = np.array([[ca, 0, -sa], [0, 1.0, 0], [sa, 0, ca]])
    p = np.inner(rotation, p).transpose()
    # reorder in a downward spiral
    reorder_index = [0, 3, 4, 8, -1, 5,-2, -3, 7, 1, 6, 2]
    return p[reorder_index, :]

def get_barymat(n):
    """
    Define the matrix that will refine points on a triangle

    """
    numrows = n*(n+1)//2

    # define the values that will be needed
    ns = np.arange(n)
    vals = ns / float(n-1)

    # initialize array
    bcmat = np.zeros((numrows, 3))#np.arange(numrows*3).reshape((numrows, 3))

    # loop over blocks to fill in the matrix
    shifts = np.arange(n,0,-1)
    starts = np.zeros(n, dtype=int);
    starts[1:] += np.cumsum(shifts[:-1]) # starts are the cumulative shifts
    stops = starts + shifts
    for n_, start, stop, shift in zip(ns, starts, stops, shifts):
        bcmat[start:stop, 0] = vals[shift-1::-1]
        bcmat[start:stop, 1] = vals[:shift]
        bcmat[start:stop, 2] = vals[n_]
    return bcmat


class icosahedron(object):
    """
    The verticies of an icosahedron, together with triangles, edges, and
    triangle midpoints and edge midpoints.  The class data stores the
    """

    # define points (verticies)
    p = get_points()
    px, py, pz = p[:,0], p[:,1], p[:,2]

    # define triangles (faces)
    tri = np.array([[1,2,3,4,5,6,2,7,2,8,3, 9,10,10,6, 6, 7, 8, 9,10],
                    [0,0,0,0,0,1,7,2,3,3,4, 4, 4, 5,5, 7, 8, 9,10, 6],
                    [2,3,4,5,1,7,1,8,8,9,9,10, 5, 6,1,11,11,11,11,11]]).transpose()

    trimids = (p[tri[:,0]] + p[tri[:,1]] + p[tri[:,2]]) / 3.0

    # define bars (edges)
    bar = list()
    for t in tri:
        bar += [np.array([i,j]) for i, j in [t[0:2], t[1:], t[[2, 0]]] if j>i]
    bar = np.array(bar)
    barmids = (p[bar[:,0]] + p[bar[:,1]]) / 2.0


def triangulate_bary(bary):
    """
    triangulate a barycentric triangle using matplotlib.
    return (bars, triangles)
    """
    x, y = np.cos(-np.pi/4.)*bary[:,0] + np.sin(-np.pi/4.)*bary[:,1] , bary[:,2]
    dely = tri.Triangulation(x,y)
    return dely.edges, dely.triangles


def get_triangulation(n, ico=icosahedron()):
    """
    Compute the triangulation of the sphere by refineing each face of the
    icosahedron to an nth order barycentric triangle.  There are two key issues
    that this routine addresses.

    1) calculate the triangles (unique by construction)
    2) remove non-unique nodes and edges
    """
    verts = np.array([ico.p[ico.tri[:,0]],
                      ico.p[ico.tri[:,1]],
                      ico.p[ico.tri[:,2]]])
    bary = get_barymat(n)
    newverts = np.tensordot(verts, bary,  axes=[(0,), (-1,)]).transpose(0,2,1)
    numverts = newverts.shape[1]
    if newverts.size/3 > 1e6: print("newverts.size/3 is high: {0}".format(
            newverts.size/3))
    flat_coordinates = np.arange(newverts.size/3).reshape(20, numverts)

    barbary, tribary = triangulate_bary(bary)

    newtri = np.zeros((20, tribary.shape[0], 3), dtype=int)
    newbar = np.zeros((20, barbary.shape[0], 2), dtype=int)

    for i in range(20):
        for j in range(3):
            newtri[i, :, j] = flat_coordinates[i, tribary[:,j]]
            if j < 2: newbar[i, :, j] = flat_coordinates[i, barbary[:,j]]

    newverts = newverts.reshape(newverts.size//3, 3)
    newtri = newtri.reshape(newtri.size//3, 3)
    newbar = newbar.reshape(newbar.size//2, 2)
    # normalize verticies
    scalars = np.sqrt((newverts**2).sum(-1))
    newverts = (newverts.T / scalars).T
    # remove repeated verticies
    aux, iunique, irepeat = np.unique(np.dot(newverts//1e-8, 100*np.arange(1,4,1)),
                                      return_index=True, return_inverse=True)

    univerts = newverts[iunique]
    unitri = irepeat[newtri]
    unibar = irepeat[newbar]
    mid = .5 * (univerts[unibar[:,0]] + univerts[unibar[:,1]])
    aux, iu  = np.unique(np.dot(mid//1e-8, 100*np.arange(1,4,1)), return_index=True)
    unimid = mid[iu]
    unibar = unibar[iu,:]
    return univerts, unitri, unibar

class icosphere(icosahedron):
    """
    """
    def __init__(self, n):
        """
        define an icosahedron based discretization of the sphere
        n is the order of barycentric triangles used to refine each
        face of the icosaheral base mesh.
        """
        self.p, self.tri, self.bar = get_triangulation(n+1, icosahedron)


'''
if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot

    from mpl_toolkits.mplot3d import Axes3D

    isph=icosphere(15)
    npt=isph.p.shape[0]
    nelem=isph.tri.shape[0]
    vertices=isph.p
    faces=isph.tri

    fig = pyplot.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],cmap='viridis', triangles=faces, linewidth=0.10,edgecolor="black",alpha=1.0)
    pyplot.show()
'''