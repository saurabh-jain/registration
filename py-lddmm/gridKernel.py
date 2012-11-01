import numpy as np
import scipy as sp


class GridKernel:
    def __init__(dimension = None, name = 'gauss', sigma = 6.5, order = 3, lddmm_order = 1, alpha = 0.01, gamma = 1, w1 = 1.0, w2 = 1.0, dim = 3, center = [0,0,0]):
        self.name = name
        self.sigma = sigma
        self.order = order
        self.alpha = alpha
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.dim = dim
        self.center = center
        self.dimension = dimension
        self.image_dimension = None
        self.initialized = False
        if name == 'gauss' || name == 'laplacian':
            center = self.dimension/2 ;
            if len(self.dimension)==3:
                (x,y,z) = np.mgrid(0:self.dimension[0], 0:self.dimension[1], 0:self.dimension[2])
                u = (x-self.center[0])**2 + (y-self.center[1])**2 + (z-self.center[2])**2
            elif len(self.dimension)==2:
                (x,y) = np.mgrid(0:self.dimension[0], 0:self.dimension[1])
                u = (x-self.center[0])**2 + (y-self.center[1])**2 
            elif len(self.dimension)==1:
                (x,y) = np.mgrid(0:self.dimension[0])
                u = (x-self.center[0])**2
            else:
                print 'use lddmm kernel in dimension 1 , 2 or 3'
                return
            if name == 'gauss':
                self.values = np.exp(-u/(2*sigma**2))
            elif name=='laplacian':
                u = np.sqrt(u) / sigma
                if order == 0:
                    pol = np.ones(u.shape)
                elif order == 1:
                    pol = 1 + u
                elif order == 2:
                    pol = 1 + u + u*u/3 
                elif order == 3:
                    pol = 1 + u + 0.4*u*u + u**3/15
                elif order == 4:
                    pol = 1 + u + (3./7)*(u*u) + (u**3)/10.5 + (u**4)/105
                else:
                    print "Laplacian kernel defined up to order 4"
                    return
                self.values = pol * np.exp(-u)
                

    def apply(self, img):
        if not(self.initialized and (img.shape == self.image_dimension)):
            if name == 'LDDMM':
                fimg = np.rfftn(img)
                if len(self.dimension)==3:
                    (x,y,z) = np.mgrid(0:img.shape[0], 0:img.shape[1], 0:img.shape[2])
                    u = (self.gamma + self.alpha *(x**2 +y**2+z**2))**(2*self.lddmm_order)
                elif len(self.dimension)==2:
                    (x,y) = np.mgrid(0:img.shape[0], 0:img.shape[1])
                    u = (self.gamma + self.alpha *(x**2 +y**2))**(-2*self.lddmm_order)
                elif len(self.dimension)==1:
                    x = np.mgrid(0:img.shape[0])
                    u = (self.gamma + self.alpha *(x**2))**(2*self.lddmm_order)
                else:
                    print 'use lddmm kernel in dimension 1 , 2 or 3'
                    return
                self.image_dimension = copy(img.shape)
                self.fft = np.rfftn(u)
                
            else:
                self.image_dimension = img.shape + self.dimension/2 + 1
                fimg = np.rfftn(img, self.image_dimension)
                self.fft = np.ifftshift(np.rfftn(self.values, self.image_dimension))

                
            print 'kernel is not implemented (yet...)'
            return

        u = np.multiply(self.fft, fimg)
        res = np.ifftn(u)
        return res

