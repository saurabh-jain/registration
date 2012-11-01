import numpy as np

class AffineBasis:
    def __init__(self, dim=3, affine='affine'):
        u = 1.0/np.sqrt(2.0)
        dimSym = (dim * (dim-1))/2
        affCode = 0
        self.dim = dim
        self.rotComp = []
        self.simComp = []
        self.affComp = []
        self.transComp = []
        if affine == 'affine':
            affCode = 1
            self.affineDim = 2* (dimSym + dim)
        elif affine == 'similitude':
            affCode = 2
            self.affineDim = dimSym + 1 + dim
        elif affine == 'euclidean':
            affCode = 3
            self.affineDim = dimSym + dim
        elif affine == 'translation':
            affCode = 4
            self.affineDim = dim
        else:
            affCode = 5
            self.basis = []
            self.affineDim = 0
        if affCode <= 4:
            self.basis = np.zeros([2 * (dimSym + dim), self.affineDim])

        k = 0 ;
        if affCode <= 1:
            for i in range(dimSym):
                for j in range(i+1, dim):
                    self.basis[i*dim + j, k] = u
                    self.basis[j*dim + i, k] = u
                    k+=1
            for i in range(dim-1):
                uu = np.sqrt((1 - 1.0/(i+2)))/(i+1.)
                self.basis[(i+1)*dim + (i+1.), k] = np.sqrt(1 - 1.0/(i+2))
                for j in range(i+1):
                    self.basis[j*dim + j, k] = -uu
                k += 1
            self.affComp = range(k)
        if affCode <= 2:
            k0=k
            for i in range(dim):
                self.basis[i*dim+i,k] = 1./np.sqrt(dim)
            k += 1
            self.simComp = range(k0, k)
        if affCode <= 3:
            k0 = k
            for i in range(dim):
                for j in range(i+1, dim):
                    self.basis[i*dim + j, k] = u
                    self.basis[j*dim + i, k] = -u
                    k+=1
            self.rotComp = range(k0,k)
        if affCode <= 4:
            k0 = k
            for i in range(dim):
                self.basis[i+dim**2, k] = 1
                k += 1
            self.transComp = range(k0, k)

    def getTransforms(self, Afft):
        Tsize = Afft.shape[0]
        dim2 = self.dim**2
        A = [np.zeros([Tsize, self.dim, self.dim]), np.zeros([Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(Tsize):
                AB = np.dot(self.basis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        return A
