import numpy as np
import logging
import loggingUtils
import surfaces
from surfaces import *
from kernelFunctions import *
import surfaceMatching
#import surfaceTimeSeries as match
import secondOrderMatching as match

def compute():

    outputDir = '/Users/younes/Development/Results/biocardTS/spline2'
    #outputDir = '/Users/younes/Development/Results/biocardTS/geodesic'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()

    rdir = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
    #sub = '2840698'
    sub = '2729611'
    if sub == '2840698':
        fv1 = surfaces.Surface(filename=rdir+'2840698_2_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv2 = surfaces.Surface(filename=rdir+'2840698_3_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv3 = surfaces.Surface(filename=rdir+'2840698_4_8_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv4 = surfaces.Surface(filename=rdir+'2840698_5_6_hippo_L_reg.byu_10_6.5_2.5.byu')
    if sub == '2729611':
        fv1 = surfaces.Surface(filename=rdir+sub+'_1_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv2 = surfaces.Surface(filename=rdir+sub+'_2_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv3 = surfaces.Surface(filename=rdir+sub+'_3_7_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv4 = surfaces.Surface(filename=rdir+sub+'_4_6_hippo_L_reg.byu_10_6.5_2.5.byu')

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)


    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=10., errorType='varifold')
    f = (match.SurfaceMatching(Template=fv1, Targets=(fv2,fv3,fv4), outputDir=outputDir, param=sm,
                               typeRegression='spline',
                          affine='euclidean', testGradient=True, affineWeight=.1,  maxIter=1000, controlWeight=1.))
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()

