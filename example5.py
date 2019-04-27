import numpy as np
import sympy as sy
from curvpack import curvature1,LB1

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


normals_max_error =[]
normals_avg_error =[]
MC_max_error =[]
MC_avg_error =[]
GC_max_error =[]
GC_avg_error =[]
LB_max_error =[]
LB_avg_error =[]
Fb_max_error =[]
Fb_avg_error =[]
number_tri=[]


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

  #get analytical results
  theta=np.arcsin(np.clip(np.sqrt(vertices[:,0]**2+vertices[:,1]**2),1e-15,1.0-1e-15)) #clipped to avoid singularity
  phi = np.arctan2(vertices[:,1],vertices[:,0])
  v=np.zeros([npt,3])
  rbc=f2()
  v[:,0],v[:,1],v[:,2],MCX,GCX,LBX=rbc(theta)
  NormalsX=np.c_[-v[:,1]*np.cos(phi),-v[:,1]*np.sin(phi),np.sign(vertices[:,2])*v[:,2]]

  MCX=-MCX
  FbX=2*(2*MCX*(MCX**2-GCX)+LBX)[:,np.newaxis]*NormalsX

  GC,MC,Normals=curvature1(vertices,faces)
  LB=LB1(vertices,faces,MC)

  Fb=2*(2*MC*(MC**2-GC)+LB)[:,np.newaxis]*Normals

  ne  = NormalsX-Normals
  nen = np.sqrt(ne[:,0]**2+ne[:,1]**2+ne[:,2]**2)
  errorMC = np.abs(MC-MCX)/np.max(np.abs(MCX))
  errorGC = np.abs(GC-GCX)/np.max(np.abs(GCX))
  errorLB   =  np.abs(LB-LBX)/np.max(np.abs(LBX))
  errorFb   =  (Fb-FbX)
  Fben      =  np.sqrt( (errorFb**2).sum(axis=1))/np.max(np.sqrt((FbX**2).sum(axis=1)))

  normals_max_error.append([np.max (nen)])
  normals_avg_error.append([np.mean(nen)])

  MC_max_error.append([np.max (errorMC)])
  MC_avg_error.append([np.mean(errorMC)])
  GC_max_error.append([np.max (errorGC)])
  GC_avg_error.append([np.mean(errorGC)])
  LB_max_error.append([np.max (errorLB)])
  LB_avg_error.append([np.mean(errorLB)])
  Fb_max_error.append([np.max (Fben)])
  Fb_avg_error.append([np.mean(Fben)])
  number_tri.append([nelem])

  print(f'mesh{n}, npt={npt}, nelem={nelem}, done')


dx=1.0/np.sqrt(np.array(number_tri))
error_array=np.c_[dx,number_tri,np.array(normals_max_error),np.array(normals_avg_error),
                  np.array(MC_max_error),np.array(MC_avg_error),
                  np.array(GC_max_error),np.array(GC_avg_error),
                  np.array(LB_max_error),np.array(LB_avg_error),
                  np.array(Fb_max_error),np.array(Fb_avg_error)]
                  
g=open('Errors_C1L1.dat','wb')
np.savetxt(g,error_array,fmt='%16.9E '*12)
g.close()

