import numpy as np
from curvpack import icosphere,curvature1,curvature2,curvature3,curvature4,curvature5,LB1

def cart2sph(xyz):
    ptsnew = np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

#get a triangulated spherical surface
isph=icosphere(20)
npt=isph.p.shape[0]
nelem=isph.tri.shape[0]
vertices=isph.p
faces=isph.tri
isph=[]

spvert=cart2sph(vertices)

#ellipsoid a=2, b=1,c=1
c=3;b=2;a=1.5;
                              #u                   v
vertices[:,0]=a*np.cos(spvert[:,2])*np.sin(spvert[:,1])
vertices[:,1]=b*np.sin(spvert[:,2])*np.sin(spvert[:,1])
vertices[:,2]=c*np.cos(spvert[:,1])

#Exact mean curvature of ellipsoid 2*H
denom=4*(a**2*b**2*np.cos(spvert[:,1])**2 + c**2*( b**2*np.cos(spvert[:,2])**2 + a**2*np.sin(spvert[:,2])**2)*np.sin(spvert[:,1])**2 )**(3./2.)
MCX = a*b*c*( 3*(a**2+b**2) + 2*c**2 + (a**2+b**2-2*c**2)*np.cos(2*spvert[:,1]) - 2*(a**2-b**2)*np.cos(2*spvert[:,2])*np.sin(spvert[:,1])**2)

MCX/=denom  #this is 2*H

MCX/=2.0   #this is H now 

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

with open(f'ex3_out.dat','wb') as f:
  f.write(bytes('Variables = "X" "Y" "Z" "MCX" "MC4" "errMC1" "errMC2" "errMC3" "errMC4" "errMC5" \n',"utf-8"))
  f.write(bytes('ZONE F=FEPOINT,ET=TRIANGLE,N='+str(npt)+',E='+str(nelem)+'SOLUTIONTIME=0 \n','utf-8'))
  np.savetxt(f,np.c_[vertices,MCX,MC4,errMC1,errMC2,errMC3,errMC4,errMC5],fmt='%16.9E '*10)
  np.savetxt(f,1+faces,fmt='%i %i %i')



