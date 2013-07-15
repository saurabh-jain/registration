import numpy as np
import surfaces
from surfaces import *
from kernelFunctions import *
from surfaceMatching import *

def compute(createSurfaces=True):

    if createSurfaces:
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        y = y-1
        z = z-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x-.50)**2 + 0.5*y**2 + z**2)  
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        #return fv1
        
        u = (z + y)/s2
        v = (z - y)/s2
        I1 = .05 - np.minimum((x-.7)**2 + 0.5*y**2 + z**2, (x-.30)**2 + 0.5*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=2000, scales=[1, 1, 1], smooth=0.01)

        fv1.saveVTK('/Users/younes/Development/Results/Diffeons/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Diffeons/fv2.vtk')
    else:
        if False:
            path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
            #sub1 = '0186193_1_6'
            #sub2 = '1449400_1_L'
            sub2 = 'LU027_R_sumNCBC20100628'
            fv1 = surfaces.Surface(filename = path+'5_population_template_qc/newTemplate.byu')
            v1 = fv1.surfVolume()
            #f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
            fv2 = surfaces.Surface(filename = path+'2_qc_flipped_registered/'+sub2+'_registered.byu')
            v2 = fv2.surfVolume()
            if (v2*v1 < 0):
                fv2.faces = fv2.faces[:, [0,2,1]]
        else:
            if False:
                #path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
                path = '/Volumes/CIS/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
                #sub1 = '0186193_1_6'
                #sub2 = '1449400_1_L'
                sub2 = 'LU027_R_sumNCBC20100628'
                fv1 = surfaces.Surface(filename = path+'5_population_template_qc/newTemplate.byu')
                v1 = fv1.surfVolume()
                #f0.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
                fv2 = surfaces.Surface(filename = path+'2_qc_flipped_registered/'+sub2+'_registered.byu')
                v2 = fv2.surfVolume()
                if (v2*v1 < 0):
                    fv2.faces = fv2.faces[:, [0,2,1]]
            else:
                #f1.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub2+'_amyg_L.byu'))
                fv1 = Surface(filename='/Users/younes/Development/Results/Diffeons/fv1.vtk')
                fv2  = Surface(filename='/Users/younes/Development/Results/Diffeons/fv2.vtk')

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 5.0)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=5, sigmaError=1., errorType='current')
    f = SurfaceMatching(Template=fv1, Target=fv2, outputDir='/Users/younes/Development/Results/Surface/Balls',param=sm, testGradient=False,
                        #subsampleTargetSize = 500,
                         maxIter=1000, affine= 'none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()


    return f
