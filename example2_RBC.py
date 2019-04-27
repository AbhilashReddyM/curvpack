import numpy as np
import sympy as sy
from curvpack import icosphere,curvature1,curvature2,curvature3,curvature4,curvature5,LB1

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

#get a triangulated spherical surface
isph=icosphere(20)
npt=isph.p.shape[0]
nelem=isph.tri.shape[0]
vertices=isph.p
faces=isph.tri
isph=[]

#get the z coordinate for each point and project the sphere. Also get analytical
#curvatures and Laplace-Beltrami of curvature

#clip theta values to slightly below 1 and slightly above 0 to avoid numerical issues down the line
theta=np.arcsin(np.clip(np.sqrt(vertices[:,0]**2+vertices[:,1]**2),1e-15,1.0-1e-15))
phi = np.arctan2(vertices[:,1],vertices[:,0])

RBC_Analytical=f2()
z,nr,nz,MCX,GCX,LBX=RBC_Analytical(theta)

NormalsX=np.c_[-nr*np.cos(phi),-nr*np.sin(phi),np.sign(vertices[:,2])*nz]

#project the Z coordinate of the sphere to the Z coordinate of the analytical RBC shape
#the analytical function always returns a positive value. so we use the original position of the mesh to decide
# whether this point should be `projected up` or `projected down`

vertices[:,2]=np.sign(vertices[:,2])*z

MCX=-MCX
#FbX=2*(2*MCX*(MCX**2-GCX)+LBX)[:,np.newaxis]*NormalsX

GC,MC1,Normals=curvature1(vertices,faces)
GC,MC2,Normals=curvature2(vertices,faces)
GC,MC3,Normals=curvature3(vertices,faces)
GC,MC4,Normals=curvature4(vertices,faces)
GC,MC5,Normals=curvature5(vertices,faces)

errMC1 = np.abs(MC1-MCX)/np.max(np.abs(MCX))
errMC2 = np.abs(MC2-MCX)/np.max(np.abs(MCX))
errMC3 = np.abs(MC3-MCX)/np.max(np.abs(MCX))
errMC4 = np.abs(MC4-MCX)/np.max(np.abs(MCX))
errMC5 = np.abs(MC5-MCX)/np.max(np.abs(MCX))

with open(f'ex2_out.dat','wb') as f:
  f.write(bytes('Variables = "X" "Y" "Z" "MC1" "MC2" "MC3" "MC4" "MC5" \n',"utf-8"))
  f.write(bytes('ZONE F=FEPOINT,ET=TRIANGLE,N='+str(npt)+',E='+str(nelem)+'SOLUTIONTIME=0 \n','utf-8'))
  np.savetxt(f,np.c_[vertices,errMC1,errMC2,errMC3,errMC4,errMC5],fmt='%16.9E '*8)
  np.savetxt(f,1+faces,fmt='%i %i %i')
