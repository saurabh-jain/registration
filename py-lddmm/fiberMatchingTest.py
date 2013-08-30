import numpy as np
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
        
        I1 = np.minimum(.05 - ((x-.50)**2 + 0.5*y**2 + z**2), np.minimum((2*(x-.50)**2 + 0.5*y**2 + 2*z**2)-0.03, 0.15-y))  
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)

        fv1.saveVTK('/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Fibers/fv2.vtk')
    else:
        fv1 = Surface(filename='/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2  = Surface(filename='/Users/younes/Development/Results/Fibers/fv2.vtk')

    ## Object kernel
    V1 = fv1.vertices/100 - [0,1,1]
    N1 = fv1.computeVertexNormals()
    QNN = (N1[:,0]-0.5)**2 + 0.5*N1[:,1]**2 + N1[:,2]**2
    QVV = (V1[:,0]-0.5)**2 + 0.5*V1[:,1]**2 + V1[:,2]**2
    QNV = (N1[:,0]-0.5)*(V1[:,0]-0.5) + 0.5*N1[:,1]*V1[:,1] + N1[:,2]*V1[:,2]
    I1 = np.nonzero(np.fabs(QVV-0.06) < 0.01)
    print V1
    I1= I1[0]
    a = np.zeros(len(I1))
    a = (-QNV[I1] + np.sqrt(QNV[I1]**2 - QNN[I1]*(QVV[I1]-0.0525)))/QNN[I1]
    N = V1.shape[0]
    M = V1.shape[0] + len(I1)
    y0 = np.zeros([M,3])
    v0 = np.zeros([M,3])
    nz = np.zeros([N, 3])
    y0[0:N, :] = V1 
    y0[N:M, :] = V1[I1,:] + a[:,np.newaxis]*N1[I1,:]
    nz[:, 0] = N1[:, 1]
    nz[:, 1] = -N1[:, 0]
    nz = nz / np.sqrt((nz**2).sum(axis=1)[:,np.newaxis])
    v0[N:M, :] = nz[I1, :]
    theta = np.pi/12
    c = np.cos(theta)
    s = np.sin(theta)
    v0[0:N, 0] = c*nz[:,0] 
    v0[0:N, 1] = c*nz[:,1] 
    v0[0:N, 2] = s 
    K1 = Kernel(name='gauss', sigma = 10.0)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=5, sigmaError=1., errorType='current')
    f = SurfaceMatching(Template=fv1, Target=fv2, Fiber=(y0,v0), outputDir='/Users/younes/Development/Results/Fibers/Ellipses',param=sm, testGradient=True,
                        #subsampleTargetSize = 500,
                         maxIter=1000)

    f.optimizeMatching()


    return f
