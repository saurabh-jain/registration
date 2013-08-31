import numpy as np
from scipy.ndimage import *
import diffeo
from diffeo import *
from kernelFunctions import *
from gaussianDiffeonsImageMatching import *

def compute(createImages=True):

    if createImages:
        [x,y] = np.mgrid[0:100, 0:100]/50.
        x = x-1
        y = y-1


        I1 = filters.gaussian_filter(255*np.array(.06 - ((x)**2 + 1.5*y**2) > 0), 1)
        im1 = gridScalars(data = I1, dim=2) 

        #return fv1
        
        I2 = filters.gaussian_filter(255*np.array(.05 - np.minimum((x-.2)**2 + 1.5*y**2, (x+.20)**2 + 1.5*y**2) > 0), 1)  
        im2 = gridScalars(data = I2, dim=2) 
        #I1 = .06 - ((x-.50)**2 + 0.75*y**2 + z**2)  
        #I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2) 

        im1.saveImg('/Users/younes/Development/Results/Diffeons/Images/im1.png', normalize=True)
        im2.saveImg('/Users/younes/Development/Results/Diffeons/Images/im2.png', normalize=True)
    else:
        if True:
            path = '/Users/younes/IMAGES/'
            im1 = gridScalars(fileName = path+'database/camel07.pgm', dim=2)
            im2 = gridScalars(fileName = path+'database/camel08.pgm', dim=2)
            im1.data = filters.gaussian_filter(im1.data, .5)
            im2.data = filters.gaussian_filter(im2.data, .5)
            #im1 = gridScalars(fileName = path+'image_0031.jpg', dim=2)
            #im2 = gridScalars(fileName = path+'image_0043.jpg', dim=2)
            #im2.saveImg('/Users/younes/Development/Results/Diffeons/Images/imTest.png', normalize=True)
            print im2.data.max()
        else:
            #f1.append(surfaces.Surface(filename = path+'amygdala/biocardAmyg 2/'+sub2+'_amyg_L.byu'))
            im1 = gridScalars(fileName='/Users/younes/Development/Results/Diffeons/Images/im1.png', dim=2)
            im2  = gridScalars(fileName='/Users/younes/Development/Results/Diffeons/Images/im2.png', dim=2)

        #return fv1, fv2

    ## Object kernel

    sm = ImageMatchingParam(timeStep=0.1, sigmaKernel=10., sigmaError=1.)
    f = ImageMatching(Template=im1, Target=im2, outputDir='/Users/younes/Development/Results/Diffeons/Images/Camel07_Camel08',param=sm, testGradient=False,
                      subsampleTemplate = 1,
                        zeroVar=False,
                        targetMargin=10,
                        templateMargin=10,
                        DecimationTarget=10,
                        maxIter=10000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()
    return f
