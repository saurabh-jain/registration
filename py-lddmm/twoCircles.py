import numpy as np
import curves
from curves import *
from kernelFunctions import *
from curveMultiPhase import *

def compute():

    ## Build Two colliding ellipses
    [x,y] = np.mgrid[0:200, 0:200]/100.
    y = y-1
    s2 = np.sqrt(2)

    I1 = .06 - ((x-.30)**2 + 0.5*y**2)  
    fv1 = Curve() ;
    fv1.Isocontour(I1, value = 0, target=750, scales=[1, 1])
    #return

    I1 = .06 - ((x-1.70)**2 + 0.5*y**2) 
    fv2 = Curve() ;
    fv2.Isocontour(I1, value=0, target=750, scales=[1, 1])

    I1 = 0.16 - ((x-.7)**2 + (y+0.25)**2) 
    fv3 = Curve() ;
    fv3.Isocontour(I1, value = 0, target=750, scales=[1, 1])

    I1 = 0.16 - ((x-1.3)**2 + (y-0.25)**2) 
    fv4 = Curve() ;
    fv4.Isocontour(I1, value=0, target=750, scales=[1, 1])

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 100.0)
    ## Background kernel
    K2 = Kernel(name='gauss', sigma = 10.0)

    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=20., sigmaError=.1, errorType='measure')
    f = (CurveMatching(Template=(fv1,fv2), Target=(fv3,fv4), outputDir='/Users/younes/Development/Results/curveMatching2',param=sm, mu=1.,regWeightOut=1., testGradient=False,
                       typeConstraint='stitched', maxIter_cg=10000, maxIter_al=100, affine='none', rotWeight=10))
    f.optimizeMatching()


    return f


if __name__=="__main__":
    compute()
