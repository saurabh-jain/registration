import numpy as np
import surfaces
from surfaces import *
from kernelFunctions import *
from gaussianDiffeonsSurfaceMatching import *

def compute():

    [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
    y = y-1
    z = z-1
    s2 = np.sqrt(2)

    I1 = .06 - ((x-.50)**2 + 0.5*y**2 + z**2)  
    fv1 = Surface() ;
    fv1.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1])

    u = (z + y)/s2
    v = (z - y)/s2
    I1 = .05 - np.minimum((x-.7)**2 + 0.5*y**2 + z**2, (x-.30)**2 + 0.5*y**2 + z**2)  
    #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
    fv2 = Surface() ;
    fv2.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1])

    ## Object kernel
    sm = SurfaceMatchingParam(timeStep=0.1, sigmaKernel=10.0, sigmaDist=20, sigmaError=10., errorType='current')
    f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results/Diffeons',param=sm, testGradient=True, DiffeonRatio=0.0125,
                        maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100., zeroVar=False)
    # f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results',param=sm, testGradient=True, Diffeons=(fv1.vertices.copy(),np.zeros([fv1.vertices.shape[0],3,3])),
    #                      maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()


    return f
