import numpy as np
import surfaces
from surfaces import *
from kernelFunctions import *
from gaussianDiffeonsSurfaceMatching import *

def compute(createSurfaces=True):

    if createSurfaces:
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        y = y-1
        z = z-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x-.50)**2 + 0.5*y**2 + z**2)  
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1])

        #return fv1
        
        u = (z + y)/s2
        v = (z - y)/s2
        I1 = .05 - np.minimum((x-.7)**2 + 0.5*y**2 + z**2, (x-.30)**2 + 0.5*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.1)

        fv1.saveVTK('/Users/younes/Development/Results/Diffeons/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Diffeons/fv2.vtk')
    else:
        path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
        sub1 = '0186193_1_6'
        #sub2 = '1449400_1_L'
        sub2 = 'LU027_R_sumNCBC20100628'
        fv1 = surfaces.Surface(filename = path+'5_population_template_qc/newTemplate.byu')
        #    f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
        fv2 = surfaces.Surface(filename = path+'2_qc_flipped_registered/'+sub2+'_registered.byu')
        #f1.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub2+'_amyg_L.byu'))
        #fv1 = Surface(filename='/Users/younes/Development/Results/Diffeons/fv1.vtk')
        #fv2  = Surface(filename='/Users/younes/Development/Results/Diffeons/fv2.vtk')

        #return fv1, fv2

    ## Object kernel
    r0 = 10./fv1.vertices.shape[0]
    sm = SurfaceMatchingParam(timeStep=0.1, sigmaKernel=6.5, sigmaDist=20, sigmaError=1., errorType='measure')
    f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results/Diffeons2/Scale1',param=sm, testGradient=False, DiffeonEpsForNet = r0,
                        #DiffeonSegmentationRatio=r0,
                        maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100., zeroVar=False)
    # f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results',param=sm, testGradient=True, Diffeons=(fv1.vertices.copy(),np.zeros([fv1.vertices.shape[0],3,3])),
    #                      maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    f.maxIter = 100
    f.setOutputDir('/Users/younes/Development/Results/Diffeons2/Scale2')
    f.restart(DiffeonEpsForNet = 2*r0)
    #f.restart(DiffeonSegmentationRatio = 0.025)
    f.maxIter = 100
    f.setOutputDir('/Users/younes/Development/Results/Diffeons2/Scale3')
    f.restart(DiffeonEpsForNet = 4*r0)
    #f.restart(DiffeonSegmentationRatio = 0.05)

    return f
