import numpy as np
from scipy.spatial import distance as dfun

#import kernelMatrix_fort

## Computes matrix associated with Gaussian kernel
# par[0] = width
# if y=None, computes K(x,x), otherwise computes K(x,y)
def kernelMatrixGauss(x, y=None, grid=None, par=[1], diff = False, diff2 = False, constant_plane=False, precomp=None):
    sig = par[0]
    sig2 = 2*sig*sig
    if precomp == None:
        if y==None:
            if grid == None:
                u = np.exp(-dfun.pdist(x,'sqeuclidean')/sig2)
                #        K = np.eye(x.shape[0]) + np.mat(dfun.squareform(u))
                K = dfun.squareform(u, checks=False)
                np.fill_diagonal(K, 1)
                precomp = np.copy(K)
                if diff:
                    K = -K/sig2
                elif diff2:
                    K = K/(sig2*sig2)
            else:
                dst = ((grid[..., newaxis, :] - x)**2).sum(axis=-1)
                K = np.exp(-dst/sig2)
                if diff:
                    K = -K/sig2
                elif diff2:
                    K = K/(sig2*sig2)
        else:
            K = np.exp(-dfun.cdist(y, x, 'sqeuclidean')/sig2)
            precomp = np.copy(K)
            if diff:
                K = -K/sig2
            elif diff2:
                K = K/(sig2*sig2)
    else:
        K = np.copy(precomp)
        if diff:
            K = -K/sig2
        elif diff2:
            K = K/(sig2*sig2)

    if constant_plane:
        K2 = np.exp(-dfun.pdist(x[:,x.shape[1]-1],'sqeuclidean')/sig2)
        np.fill_diagonal(K2, 1)
        if diff:
            K2 = -K2/sig2
        elif diff2:
            K2 = K2/(sig2*sig2)
        return K, K2, precomp
    else:
        return K, precomp

# Polynomial factor for Laplacian kernel
def lapPol(u, ord):
    if ord == 0:
        pol = 1.
    elif ord == 1:
        pol = 1. + u
    elif ord == 2:
        pol = (3. + 3*u + u*u)/3.
    elif ord == 3:
        pol = (15. + 15 * u + 6*u*u + u*u*u)/15.;
    elif ord == 4:
        pol = (105. + 105*u + 45*u*u + 10*u*u*u + u*u*u*u)/105.
    else:
        exit(0,'Invalid Laplacian kernel order\n')
    return pol


# Polynomial factor for Laplacian kernel (first derivative)
def lapPolDiff(u, ord):
    if ord == 0:
        exit(0, 'Order zero Laplacian is not differentiable')
    elif ord == 1:
        pol = 1.
    elif ord == 2:
        pol = (1 + u)/3.
    elif ord == 3:
        pol = (3 + 3*u + u*u)/15.
    elif ord == 4:
        pol = (15 + 15 * u + 6*u*u + u*u*u)/105.
    else:
        exit(0,'Invalid Laplacian kernel order\n')
    return pol


# Polynomial factor for Laplacian kernel (second derivative)
def lapPolDiff2(u, ord):
    if ord <= 1:
        exit(0, 'Order zero or 1 Laplacian are not twice differentiable')
    elif ord == 2:
        pol = 1.0/3.
    elif ord == 3:
        pol = (1 + u)/15.
    elif ord == 4:
        pol = (3 + 3 * u + u*u)/105.
    else:
        exit(0,'Invalid Laplacian kernel order\n')
    return pol


# computes matrix associated with polynomial kernel
# par[0] = width, par[1] =  order
#def kernelMatrixLaplacian(x, y=None, par=[1., 3], diff=False, diff2 = False, constant_plane=False, precomp = None):
#    sig = par[0]
#    ord=par[1]
#    if precomp == None:
#        if y==None:
#            u = dfun.pdist(x)/sig
#        else:
#            u = dfun.cdist(x, y)/sig
#        precomp = u
#    else:
#        u = precomp
#
#    #if diff==False & diff2==False:
#    if diff==False and diff2==False:
#        if y == None:
#            K = dfun.squareform(np.multiply(lapPol(u,ord), np.exp(-u)))
#            np.fill_diagonal(K, 1)
#        else:
#            K = np.multiply(lapPol(u,ord), np.exp(-u))
#    elif diff2==False:
#        if y == None:
#            K = dfun.squareform(np.multiply(-lapPolDiff(u, ord), np.exp(-u)/(2*sig*sig)))
#            np.fill_diagonal(K, -1./((2*ord-1)*2*sig*sig))
#        else:
#            K = np.multiply(-lapPolDiff(u, ord), np.exp(-u)/(2*sig*sig))
#    else:
#        if y == None:
#            K = dfun.squareform(np.multiply(lapPolDiff2(u, ord), np.exp(-u)/(4*sig**4)))
#            #np.fill_diagonal(K, 1./((2*ord-1)*4*sig**4))
#            np.fill_diagonal(K, 1./((35)*4*sig**4))
#        else:
#            K = np.multiply(lapPolDiff2(u, ord), np.exp(-u)/(4*sig**4))
#
#
#    if constant_plane:
#        uu = dfun.pdist(x[:,x.shape[1]-1])/sig
#        K2 = dfun.squareform(lapPol(uu,ord)*np.exp(-uu))
#        np.fill_diagonal(K2, 1)
#        return K,K2,precomp
#    else:
#        return K,precomp

# computes matrix associated with polynomial kernel
# par[0] = width, par[1] =  order
def kernelMatrixLaplacian(x, y=None, grid=None, par=[1., 3], diff=False, diff2 = False, constant_plane=False, precomp = None):
    sig = par[0]
    ord=par[1]
    if precomp == None:
        precomp = kernelMatrixLaplacianPrecompute(x, y, grid, par)

    u = precomp[0]
    expu = precomp[1]

    if y == None and grid==None:
        if diff==False and diff2==False:
            K = dfun.squareform(lapPol(u,ord) *expu)
            np.fill_diagonal(K, 1)
        elif diff2==False:
            K = dfun.squareform(-lapPolDiff(u, ord) * expu/(2*sig*sig))
            np.fill_diagonal(K, -1./((2*ord-1)*2*sig*sig))
        else:
            K = dfun.squareform(lapPolDiff2(u, ord) *expu /(4*sig**4))
            np.fill_diagonal(K, 1./((35)*4*sig**4))
    else:
        if diff==False and diff2==False:
            K = lapPol(u,ord) * expu
        elif diff2==False:
            K = -lapPolDiff(u, ord) * expu/(2*sig*sig)
        else:
            K = lapPolDiff2(u, ord) *expu/(4*sig**4)

    if constant_plane:
        uu = dfun.pdist(x[:,x.shape[1]-1])/sig
        K2 = dfun.squareform(lapPol(uu,ord)*np.exp(-uu))
        np.fill_diagonal(K2, 1)
        return K,K2,precomp
    else:
        return K,precomp

def kernelMatrixLaplacianPrecompute(x, y=None, grid=None, par=[1., 3], diff=False, diff2 = False, constant_plane=False):
    sig = par[0]
    ord=par[1]
    if y==None:
        if grid==None:
            u = dfun.pdist(x)/sig
        else:
            u = np.sqrt(((grid[..., newaxis, :] - x)**2).sum(axis=-1))/sig
    else:
        u = dfun.cdist(y, x)/sig
    precomp = [u, exp(-u)]
    return precomp

# Wrapper for kernel matrix computation
def  kernelMatrix(Kpar, x, y=None, grid=None, diff = False, diff2=False, constant_plane = False):
    # [K, K2] = kernelMatrix(Kpar, varargin)
    # creates a kernel matrix based on kernel parameters Kpar
    # if varargin = z

    #if (Kpar.prev_x is x) and (Kpar.prev_y is y):
    if Kpar._hold:
        #print 'Kernel: not computing'
        precomp = np.copy(Kpar.precomp)
    else:
        precomp = None


    if Kpar.name == 'gauss':
        res = kernelMatrixGauss(x,y=y, grid=grid, par = [Kpar.sigma], diff=diff, diff2=diff2, constant_plane = constant_plane, precomp=precomp)
    elif Kpar.name == 'laplacian':
        res = kernelMatrixLaplacian(x,y=y, grid=grid, par = [Kpar.sigma, Kpar.order], diff=diff, diff2=diff2, constant_plane = constant_plane, precomp=precomp)
    else:
        print 'unknown Kernel type'
        return []

    #Kpar.prev_x = x
    #Kpar.prev_y = y
    Kpar.precomp = res[-1]
    if constant_plane:
        return res[0:2]
    else:
        return res[0]


# Kernel specification
# name = 'gauss' or 'laplacian'
# affine = 'affine' or 'euclidean' (affine component)
# sigma: width
# order: order for Laplacian kernel
# w1: weight for linear part; w2: weight for translation part; center: origin
# dim: dimension
class KernelSpec:
    def __init__(self, name='gauss', affine = 'none', sigma = 6.5, order = 3, w1 = 1.0, w2 = 1.0, dim = 3, center = [0,0,0], weight = 1.0):
        self.name = name
        self.sigma = sigma
        self.order = order
        self.weight=weight
        self.w1 = w1
        self.w2 = w2
        self.constant_plane=False
        self.center = np.array(center)
        self.affine_basis = [] ;
        self.dim = dim
        #self.prev_x = []
        #self.prev_y = []
        self.precomp = []
        self._hold = False
        self.affine = affine
        if name == 'laplacian':
            self.kernelMatrix = kernelMatrixLaplacian
            self.par = [sigma, order]
        elif name == 'gauss':
            self.kernelMatrix = kernelMatrixGauss
            self.par = [sigma]
        else:
            self.name = 'none'
            self.kernelMatrix = None
            self.par = []
        if self.affine=='euclidean':
            if self.dim == 3:
                s2 = np.sqrt(2.0)
                self.affine_basis.append(np.mat([ [0,1,0], [-1,0,0], [0,0,0]])/s2)
                self.affine_basis.append(np.mat([ [0,0,1], [0,0,0], [-1,0,0]])/s2)
                self.affine_basis.append(np.mat([ [0,0,0], [0,0,1], [0,-1,0]])/s2)
            elif self.dim==2:
                s2 = np.sqrt(2.0)
                self.affine_basis.append(np.mat([ [0,1], [-1,0]])/s2)
            else:
                print 'euclidian kernels only available in dimension 2 or 3'
                return


# Main class for kernel definition
class Kernel(KernelSpec):
    def precompute(self, x,  y=None, grid=None, diff=False, diff2=False):
        if not (self.kernelMatrix == None):
            if self._hold:
                precomp = self.precomp
            else:
                precomp = None

            #precomp = None
            r = self.kernelMatrix(x, y=y, grid = grid, par = self.par, precomp=precomp, diff=diff, diff2=diff2)
            #self.prev_x = x
            #self.prev_y = y
            self.precomp = r[1]
            return r[0] * self.weight

    def hold(self):
        self._hold = True
    def release(self):
        self._hold = False

    # Computes K(x,x)a or K(x,y)a
    def applyK(self, x, a, y = None, grid=None):
        if not (self.kernelMatrix == None):
            r = self.precompute(x, y=y, grid=grid, diff=False)
            z = np.dot(r, a)
        else:
            z = np.zeros([x.shape[0],a.shape[1]])
        if self.affine == 'affine':
            xx = x-self.center
            #aa = np.mat(a)
            if y == None:
                if grid==None:
                    z += self.w1 * np.dot(xx, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
                else:
                    yy = grid -self.center
                    #yy = grid - self.center.reshape(np.concatenate([[self.center.size], np.ones(self.center.size)]))
                    z += self.w1 * np.dot(grid, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
            else:
                yy = y-self.center
                z += self.w1 * np.dot(yy, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
        elif self.affine == 'euclidean':
            xx = x-self.center
            if not (y==None):
                yy = y-self.center
            if not (grid==None):
                gg = grid - self.center
                #self.center.reshape(np.concatenate([[self.center.size], np.ones(self.center.size)]))
                #aa = np.mat(a)
            z += self.w2 * a.sum(axis=0)
            for E in self.affine_basis:
                xE = np.dot(xx, E.T)
                if y==None:
                    if grid == None:
                        z += self.w1 * (xE * a).sum() * xE
                    else:
                        gE = np.dot(gg, E.T)
                        z += self.w1 * (xE * a).sum() * gE
                else:
                    yE = np.dot(yy, E.T)
                    z += self.w1 * np.multiply(xE, a).sum() * yE
                #print 'In kernel: ', self.w1 * np.multiply(yy, aa).sum()

        return z

    # Computes A(j) = D_1[K(x(j), x)a2]a1(j)
    def applyDiffK(self, x, a1, a2):
        zpx = np.zeros(x.shape)
        v = np.dot(a1, x.T)
        if not (self.kernelMatrix == None):
            r = self.precompute(x, diff=True)
            u = -v + np.multiply(x, a1).sum(axis=1)
            zpx +=  2* np.dot(np.multiply(r, u), a2)
        if self.affine == 'affine':
            xx = x-self.center
            zpx += self.w1 * np.dot(v, a2)
        elif self.affine == 'euclidean':
            xx = x-self.center
            for E in self.affine_basis:
                yy = np.dot(xx,E.T)
                bb = np.dot(a1,E.T)
                zpx += self.w1 * np.multiply(yy, a2).sum() * bb
        return zpx

    # Computes A(i) = sum_j D_2[K(x(i), x(j))a2(j)]a1(j)
    def applyDiffK2(self, x, a1, a2):
        zpx = np.zeros(x.shape)
        v = np.dot(x,a1.T)
        if not (self.kernelMatrix == None):
            r = self.precompute(x, diff=True)
            u = (v - np.multiply(x, a1).sum(axis=1)).T
            zpx =  -2* np.dot(np.multiply(r, u), a2)
        if self.affine == 'affine':
            xx = x-self.center
            zpx += self.w1 * np.dot(v,a2)
            #np.multiply(xx, a1).sum(axis = 1) * a2.sum(axis=0)
        elif self.affine == 'euclidean':
            xx = x-self.center
            for E in self.affine_basis:
                yy = np.dot(xx,E.T)
                bb = np.dot(a1,E.T)
                zpx += self.w1 * np.dot(np.dot(xx,a1.T), a2)
                #np.multiply(a2.T, bb.sum(axis=0).T) * yy
        return zpx

    # Computes array A(j) = sum_(k) nabla_1[a1(k, j). K(x(j), x)a2(k)]
    def applyDiffKT(self, x, a1, a2, y=None):
        zpx = np.zeros(x.shape)
        a = np.dot(a1[0], a2[0].T)
        for k in range(1,len(a1)):
            a += np.dot(a1[k], a2[k].T)
        if not (self.kernelMatrix == None):
            r = self.precompute(x, diff=True, y=y)
            g1 =  np.multiply(r, a)
            #print a.shape, r.shape, g1.shape
            if y==None:
                zpx = 2*(np.multiply(x, g1.sum(axis=1).reshape([x.shape[0],1])) - np.dot(g1,x))
            else:
                zpx = 2*(np.multiply(x, g1.sum(axis=1).reshape([x.shape[0], 1])) - np.dot(g1,y))
        if self.affine == 'affine':
            if y==None:
                xx = x-self.center
            else:
                xx = y-self.center
            zpx += self.w1 * np.dot(a, xx)
        elif self.affine == 'euclidean':
            if y==None:
                xx = x-self.center
            else:
                xx = y-self.center
            for E in self.affine_basis:
                yy = np.dot(xx, E.T)
                for k in range(len(a1)):
                     bb = np.dot(a1[k], E)
                     zpx += self.w1 * np.multiply(yy, a2[k]).sum() * bb
        return zpx


    # Computes sum_(l) D_11[n(k)^T K(x(k), x(l))a(l)]p(k)
    def applyDDiffK11(self, x, n, a, p):
        zpx = np.zeros(x.shape)
        if not (self.kernelMatrix == None):
            r1 = self.precompute(x, diff=True)
            r2 = self.precompute(x, diff2=True)
            #xxp = -np.dot(p, x.T) + np.multiply(x, p).sum(axis=1)
            xxp = -np.dot(p, x.T) + np.multiply(x,p).sum(axis=1).\
                                reshape([x.shape[0],1])
            na = np.dot(n, a.T)
            xpna = np.multiply(xxp, na)
            #u = np.multiply(xpna, x) - np.mutiply(xpna, x.T)
            u = np.multiply(r2, xpna)
            zpx = 4 * (np.multiply(u.sum(axis=1).reshape([x.shape[0],1]), x) - np.dot(u, x))
            u = np.multiply(r1, na)
            zpx += 2*np.multiply(u.sum(axis=1).reshape([x.shape[0],1]), p)

        return zpx

    # Computes sum_(l) D_12[n(k)^T K(x(k), x(l))a(l)]p(l)
    def applyDDiffK12(self, x, n, a, p):
        zpx = np.zeros(x.shape)
        na = np.dot(n, a.T)
        if not (self.kernelMatrix == None):
            r1 = self.precompute(x, diff=True)
            r2 = self.precompute(x, diff2=True)
            #xxp = (np.dot(p, x.T) - np.multiply(x, p).sum(axis=1)).T
            xxp = np.dot(x, p.T) - np.multiply(x,p).sum(axis=1)
            na = np.dot(n, a.T)
            xpna = np.multiply(xxp, na)
            #u = np.multiply(xpna, x) - np.mutiply(xpna, x.T)
            u = np.multiply(r2, xpna)
            zpx = - 4 * (np.multiply(u.sum(axis=1).reshape([x.shape[0],1]), x) - np.dot(u, x))
            u = np.multiply(r1, na)
            zpx -= 2* np.dot(u, p)
        if self.affine == 'affine':
            zpx += self.w1 * np.dot(na, p)
        elif self.affine == 'euclidean':
            for E in self.affine_basis:
                zpx += self.w1 * np.multiply(a*E, p).sum() * np.dot(n, E)
        return zpx


    # Computes sum_l div_1(K(x_k, x_l)a_l)
    def applyDivergence(self, x, a, y=None):
        zJ = np.zeros([y.shape[0],1])
        if not (self.kernelMatrix == None):
            r = self.precompute(x, y=y,  diff=True)
            if y==None:
                zJ += 2 * (np.multiply(np.dot(r,a), x).sum(axis=1) - np.dot(r, np.multiply(a,x).sum(axis=1)))
                if self.affine == 'affine':
                    xx = x-self.center
                    zJ += self.w1 * np.multiply(xx,a).sum(axis=1)
            else:
                #print r.shape, a.shape, y.shape, x.shape, zJ.shape
                zJ += 2 * (np.multiply(np.dot(r,a), y).sum(axis=1) - np.dot(r, np.multiply(a,x).sum(axis=1)))
                if self.affine == 'affine':
                    xx = y-self.center
                    zJ += self.w1 * np.multiply(xx,a).sum(axis=1)
        return zJ.T
