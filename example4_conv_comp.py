import numpy as np
import sympy as sy
from curvpack import icosphere,curvature1,curvature2,curvature3,curvature4,curvature5,curvature6

def f2():
  '''
  Sympy stuff for geometry of Red Blood Cell. We start with the equation
  for the geometry and apply differential operators(surface gradient and surface divergence)
  to get the Mean and Gaussian curvatures. This is an axisymmetric surface. We use 
  cylindrical coordinate system
  '''
  t  = sy.symbols('t')

  c0 = sy.Rational(' 0.2072')
  c1 = sy.Rational(' 2.0026')
  c2 = sy.Rational('-1.1228')

  #equation of the surface
  r=sy.sin(t)
  z=(1/2)*sy.sqrt(1-r*r)*(c0+c1*r*r+c2*r*r*r*r)

  ds = sy.simplify(sy.diff(z,t))
  dr = sy.diff(r,t)

  b=sy.sqrt(dr**2+ds**2)
  #normal components in the r and z directions
  nr  = ds/b
  nz  = dr/b

  d2s = sy.simplify(sy.diff(ds,t))
  d2r = sy.simplify(sy.diff(dr,t))

  k1  = (-d2r*ds +dr*d2s)/b**3

  k2  = ds/(r*b)

  G=k1*k2
  H=(k1+k2)/2

  dH  = r*sy.simplify( sy.diff(H,t)/b)
  d2H =-(sy.diff(dH,t)/(r*b))

  return sy.lambdify(t,[z,nr,nz,H,G,d2H],"numpy")

MC_max_error =[]
MC_avg_error =[]

g=open('Total_Errors.dat','wb')

for case in range(1,7):
  nr_max_error = []
  nr_avg_error = []
  MC_max_error = []
  MC_avg_error = []
  GC_max_error = []
  GC_avg_error = []
  number_tri   = []
  for n in range(6):
  
  # read-in the mesh data
    fname1='meshes/mesh'+str(n)+'.txt'
    f=open(fname1,'r')
    fields = f.readline().strip().split(',')
    npt=int(fields[0])
    nelem=int(fields[1])
    data=np.genfromtxt((f),usecols=(0,1,2))
    f.close()
    vertices=data[:npt,:]
    faces=data[npt:,:].astype(int)
    data=0
  
    #project mesh on to a RBC. not really. the mesh is already RBC. just get the analytical values
    theta=np.arcsin(np.clip(np.sqrt(vertices[:,0]**2+vertices[:,1]**2),1e-15,1.0-1e-15))
    phi = np.arctan2(vertices[:,1],vertices[:,0])
    v=np.zeros([npt,3])
    rbc=f2()
    v[:,0],v[:,1],v[:,2],MCX,GCX,LBX=rbc(theta)
    NormalsX=np.c_[-v[:,1]*np.cos(phi),-v[:,1]*np.sin(phi),np.sign(vertices[:,2])*v[:,2]]
    MCX=-MCX
  
    if(case==1):  
      GC,MC,Normals=curvature1(vertices,faces)
    elif(case==2):
      GC,MC,Normals=curvature2(vertices,faces)
    elif(case==3):
      GC,MC,Normals=curvature3(vertices,faces)
    elif(case==4):
      GC,MC,Normals=curvature4(vertices,faces)
    elif(case==5):
      GC,MC,Normals=curvature5(vertices,faces)
    elif(case==6):
      GC,MC,Normals=curvature6(vertices,faces)

    ne  = NormalsX-Normals
    nen = np.sqrt(ne[:,0]**2+ne[:,1]**2+ne[:,2]**2)
  
    errorMC = np.abs(MC-MCX)/np.max(np.abs(MCX))
    errorGC = np.abs(GC-GCX)/np.max(np.abs(GCX))
  
    nr_max_error.append([np.max (    nen)])
    nr_avg_error.append([np.mean(    nen)])
    MC_max_error.append([np.max (errorMC)])
    MC_avg_error.append([np.mean(errorMC)])
    GC_max_error.append([np.max (errorGC)])
    GC_avg_error.append([np.mean(errorGC)])
    number_tri.append([nelem])

  dx=1.0/np.sqrt(np.array(number_tri))
  error_array=np.c_[dx,number_tri,np.array(nr_max_error),np.array(nr_avg_error),
                    np.array(MC_max_error),np.array(MC_avg_error),
                    np.array(GC_max_error),np.array(GC_avg_error),]

  g.write(bytes(f'ZONE T=curvature{case}\n','utf-8'))
  np.savetxt(g,error_array,fmt='%16.9E '*8)

g.close()
