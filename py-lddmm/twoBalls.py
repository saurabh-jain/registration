import numpy as np
import surfaces
from surfaces import *
from kernelFunctions import *
from surfaceMultiPhase import *

def compute():

    ## Build Two colliding ellipses
    [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
    y = y-1
    z = z-1
    s2 = np.sqrt(2)

    I1 = .06 - ((x-.30)**2 + 0.5*y**2 + z**2)  
    fv1 = Surface() ;
    fv1.Isosurface(I1, value = 0, target=750, scales=[1, 1, 1])

    I1 = .06 - ((x-1.70)**2 + 0.5*y**2 + z**2) 
    fv2 = Surface() ;
    fv2.Isosurface(I1, value=0, target=750, scales=[1, 1, 1])

    u = (z + y)/s2
    v = (z - y)/s2
    I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
    fv3 = Surface() ;
    fv3.Isosurface(I1, value = 0, target=750, scales=[1, 1, 1])

    u = (z + y)/s2
    v = (z - y)/s2
    I1 = .095 - ((x-1.3)**2 + v**2 + 0.5*u**2) 
    fv4 = Surface() ;
    fv4.Isosurface(I1, value=0, target=750, scales=[1, 1, 1])

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 100.0)
    ## Background kernel
    K2 = Kernel(name='gauss', sigma = 10.0)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=20., sigmaError=10., errorType='measure')
    f = (SurfaceMatching(Template=(fv1,fv2), Target=(fv3,fv4), outputDir='/Users/younes/Development/Results/tight_Sliding_rigid2_10',param=sm, mu=1.,regWeightOut=1., testGradient=True,
                         typeConstraint='slidingV2', maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f
