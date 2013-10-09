import numpy as np
import curves
from curves import *
from kernelFunctions import *
from curveMultiPhase import *

def compute():

    ## Build Two colliding ellipses
    t = np.arange(0, 2*np.pi, 0.02)

    p = np.zeros([len(t), 2])
    p[:,0] = -1 + (1 - 0.25* np.cos(2*t)) * np.cos(t)
    p[:,1] = -1 + (1 - 0.25* np.cos(2*t)) * np.sin(t)
    fv1 = Curve(pointSet = p) ;

    p[:,0] = 1. + 0.75 * (1 - 0.15* np.cos(6*t)) * np.cos(t)
    p[:,1] = 1 + 0.75*(1 - 0.25* np.cos(6*t)) * np.sin(t)
    #p[:,0] = 1 +  np.cos(t)
    #p[:,1] = 1 + 0.75 * np.sin(t)
    fv2 = Curve(pointSet = p) ;

    p[:,0] = -1.0 + 0.75 * np.cos(t)
    p[:,1] = np.sin(t)
    fv3 = Curve(pointSet = p) ;

    p[:,0] = 1.0 + (1 - 0.25* np.cos(6*t)) * np.cos(t)
    p[:,1] =  (1 - 0.25* np.cos(6*t)) * np.sin(t)
    fv4 = Curve(pointSet = p) ;


    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 2)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = .2)

    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=0.5, sigmaError=.01, errorType='current')
    f = (CurveMatching(Template=(fv1,fv2), Target=(fv3,fv4), outputDir='/Users/younes/Development/Results/Curves_Sliding_tmp',param=sm, mu=.001,regWeightOut=1., testGradient=True,
                       typeConstraint='sliding', maxIter_cg=10000, maxIter_al=100, affine='none', rotWeight=10))
    f.optimizeMatching()


    return f
