import glob
import numpy as np
import surfaces
from surfaces import *
from kernelFunctions import *
from surfaceMatching import *
from surfaceTemplate import *

def main():
    files = glob.glob('/Volumes/project/biocard/data/phase_1_surface_mapping_new_structure/hippocampus/2_qc_flipped_registered/*_reg.byu')
    print len(files)
    fv1 = []
    for k in range(10):
        fv1.append(surfaces.Surface(filename = files[k]))
    K1 = Kernel(name='gauss', sigma = 6.5)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='current')
    f = SurfaceTemplate(HyperTmpl=fv1[0], Targets=fv1, outputDir='Results/surfaceTemplateBiocard',param=sm, testGradient=True, lambdaPrior = 0.01,
                         maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.computeTemplate()

