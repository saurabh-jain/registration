import numpy as np
import pointSets
import surfaces
from surfaces import *
from kernelFunctions import *
from secondOrderFiberMatching import *

def compute(createSurfaces=True):

    if createSurfaces:
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        y = y-1
        z = z-1
        s2 = np.sqrt(2)

        I1 = np.minimum(.06 - ((x-.50)**2 + 0.5*y**2 + z**2), np.minimum(((x-.50)**2 + 0.5*y**2 + z**2)-0.045, 0.2-y)) 
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)


        #return fv1

        s = 1.375
        I1 = np.minimum(.05/s - (((x-.50)**2 + 0.5*y**2 + z**2)), np.minimum((2*(x-.50)**2 + 0.5*y**2 + 2*z**2)-0.03/s, 0.15/s-y))  
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv2.vertices[:,1] += 20 - 15/s

        fv1.saveVTK('/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Fibers/fv2.vtk')
    else:
        fv1 = Surface(filename='/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2  = Surface(filename='/Users/younes/Development/Results/Fibers/fv2.vtk')

    ## Object kernel
    V1 = fv1.vertices/100 - [0,1,1]
    N1 = fv1.computeVertexNormals()
    sel0 = V1[:,1] < 0.9*V1[:,1].max()
    I0 = np.nonzero(sel0)
    I0 = I0[0]
    V1 = V1[I0, :]
    N1 = N1[I0, :]


    QNN = (N1[:,0]-0.5)**2 + 0.5*N1[:,1]**2 + N1[:,2]**2
    QVV = (V1[:,0]-0.5)**2 + 0.5*V1[:,1]**2 + V1[:,2]**2
    QNV = (N1[:,0]-0.5)*(V1[:,0]-0.5) + 0.5*N1[:,1]*V1[:,1] + N1[:,2]*V1[:,2]
    sel1 = np.logical_and(np.fabs(QVV-0.06) < 0.01, (np.fabs(N1[:,1]) < 0.8))
    I1 = np.nonzero(sel1)
    I1= I1[0]

    V11 = V1[I1, :]
    N11 = N1[I1, :]
    a = np.zeros(len(I1))
    a = (-QNV[I1] - np.sqrt(QNV[I1]**2 - QNN[I1]*(QVV[I1]-0.0525)))/QNN[I1]
    N = V1.shape[0]
    M = V1.shape[0] + len(I1)
    y0 = np.zeros([M,3])
    v0 = np.zeros([M,3])
    nz = np.zeros([N, 3])
    nz2 = np.zeros([N, 3])
    y0[0:N, :] = V1 
    y0[N:M, :] = V11 + a[:,np.newaxis]*N11
    nz[:, 0] = -N1[:, 2]
    nz[:, 2] = N1[:, 0]
    nz2[:,0] = N1[:,0]*N1[:,1]
    nz2[:,1] = -(N1[:,0]**2 + N1[:,2]**2)
    nz2[:,2] = N1[:,2]*N1[:,1]
    nz = nz / (1e-5 + np.sqrt((nz**2).sum(axis=1)[:,np.newaxis]))
    nz2 = nz2 / (1e-5 + np.sqrt((nz2**2).sum(axis=1)[:,np.newaxis]))
    theta = np.pi/4
    psi = np.pi/6
    c = np.cos(theta)
    s = np.sin(theta)
    c0 = np.cos(psi)
    s0 = np.sin(psi)
    v0[0:N, :] = -c*nz - s*nz2 
    #v0[I1,:] *= -1
    v0[I1, :] += 2*s*nz2[I1,:]
    v0[N:M, :] = v0[I1, :]
    v0[0:N,:] = c0*v0[0:N, :] - s0*N1
     
    K1 = Kernel(name='gauss', sigma = 10.0)
    y0 = 100*(y0+[0,1,1])
    pointSets.savePoints('/Users/younes/Development/Results/Fibers/Ellipses/fibers.vtk', y0, vector=v0)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=5, sigmaError=1., errorType='measure')
    f = SurfaceMatching(Template=fv1, Target=fv2, Fiber=(y0,v0), outputDir='/Users/younes/Development/Results/Fibers/Ellipses',param=sm, testGradient=False,
                        #subsampleTargetSize = 500,
                         maxIter=1000)

    f.optimizeMatching()


    return f
