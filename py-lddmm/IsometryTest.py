import numpy as np
from numpy import *
import surfaces
from surfaces import *
from kernelFunctions import *
import surfaceMatching
from surfaceWithIsometries import *


def branches(angles, lengths = .75, radii=.1, center = [0,0,0],
             ImageSize = 100, npt = 100):

    [x,y,z] = np.mgrid[-ImageSize:ImageSize+1, -ImageSize:ImageSize+1, -ImageSize:ImageSize+1]/float(ImageSize)

    if (type(lengths) is int) | (type(lengths) is float):
        lengths = tile(lengths, len(angles))
    if (type(radii) is int) | (type(radii) is float):
        radii = tile(radii, len(angles))

    t = np.mgrid[0:npt+1]/float(npt)
    img = np.zeros(x.shape)
    dst = np.zeros(x.shape)

    for kk,th in enumerate(angles):
        #print kk, th
        u = [cos(th[0])*sin(th[1]), sin(th[0])*sin(th[1]), cos(th[1])]
        s = (x-center[0])*u[0] + (y-center[1])*u[1] + (z-center[2])*u[2]
        dst = sqrt((x-center[0] - s*u[0])**2 + (y-center[1] - s*u[1])**2 + (z-center[2] - s*u[2])**2)
        end = center + lengths[kk]*np.array([cos(th[0])*sin(th[1]), sin(th[0])*sin(th[1]), cos(th[1])])
        dst1 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
        dst2 = (x-end[0])**2 + (y-end[1])**2 + (z-end[2])**2
        (I1, I2, I3) = nonzero(s<0)
        dst[I1, I2, I3] = sqrt(dst1[I1, I2, I3]) 
        (I1, I2, I3) = nonzero(s>lengths[kk])
        dst[I1, I2, I3] = sqrt(dst2[I1, I2, I3])
        (I1, I2, I3) = nonzero(dst <= radii[kk])
        #print dst.min(), dst.max()
        img[I1, I2, I3] = 1
    fv = Surface()
    print 'computing isosurface'
    fv.Isosurface(img, value = 0.5, target=1000, scales=[1, 1, 1])
    return fv

def compute():

    fv1 = branches([[0,0], [pi/4, pi/2.5], [-pi/4, -3*pi/4]])
    fv2 = branches([[0,0], [pi/2.5, pi/4], [-pi/3, -0.7*pi]])
    K1 = Kernel(name='gauss', sigma = 20.0)

    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=10, sigmaError=1., errorType='measure')
    f = SurfaceWithIsometries(Template=fv1, Target=fv2, outputDir='Results/IsometriesShort', centerRadius = [100., 100., 100., 30.],
                               param=sm, mu=.0001, testGradient=False, maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=1.,
    transWeight=1.)
    print f.gradCoeff
    f.optimizeMatching()


    return f

