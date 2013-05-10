import numpy as np
import numpy.linalg as LA
import scipy.linalg as spLA
import kernelFunctions as kfun
from pointSets import epsilonNet


def generateDiffeonsFromSegmentation(fv, rate):
    nc = int(np.floor(fv.vertices.shape[0] * rate))
    (idx, c) = fv.laplacianSegmentation(nc)
    for k in range(c.shape[0]):
        dst = ((c[k, :] - fv.vertices)**2).sum(axis=1)
        I = np.argmin(dst)
        c[k, :] = fv.vertices[I]
    return generateDiffeons(fv, c, idx)

def generateDiffeonsFromNet(fv, eps):
    (L, AA) =  fv.laplacianMatrix()
    (D, y) = spLA.eigh(L, AA, eigvals= (L.shape[0]-10, L.shape[0]-1))
    (net, idx) = epsilonNet(y, eps)
    c = fv.vertices[net, :]
    #print c
    return generateDiffeons(fv, c, idx)

def generateDiffeons(fv, c, idx):
    a, foo = fv.computeVertexArea()
    #print idx
    nc = idx.max()
    print 'Computed', nc+1, 'diffeons'
    S = np.zeros([nc+1, 3, 3])
    #C = np.zeros([nc, 3])
    for k in range(nc):
        I = np.flatnonzero(idx==k)
	#print I
        nI = len(I)
        aI = a[I]
        ak = aI.sum()
	#C[k, :] = (fv.vertices[I, :]*a[I]).sum(axis=0)/ak ; 
        y = (fv.vertices[I, :] - c[k, :])
        S[k, :, :] = (y.reshape([nI, 3, 1]) * aI.reshape([nI, 1, 1]) * y.reshape([nI, 1, 3])).sum(axis=0)/ak
        [D,V] = LA.eig(S[k, :, :])
        D = np.sort(D, axis=None)
        S[k, :, :] = S[k, :, :] * np.sqrt(ak/(np.pi * (D[1]*D[2])))
        #print np.pi * (D[1]*D[2]), ak
    return c, S, idx

        

# Saves in .vtk format
def saveDiffeons(fileName, c, S):
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
        fvtkout.write('\nPOINTS {0: d} float'.format(c.shape[0]))
        for ll in range(c.shape[0]):
            fvtkout.write('\n{0: f} {1: f} {2: f}'.format(c[ll,0], c[ll,1], c[ll,2]))
        fvtkout.write(('\nPOINT_DATA {0: d}').format(c.shape[0]))
        d,v = multiMatEig(S)
        fvtkout.write('\nSCALARS first_eig float 1\nLOOKUP_TABLE default')
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f}'.format(d[ll,2]))
        fvtkout.write('\nSCALARS second_eig float 1\nLOOKUP_TABLE default')
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f}'.format(d[ll,1]))
        fvtkout.write('\nSCALARS third_eig float 1\nLOOKUP_TABLE default')
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f}'.format(d[ll,0]))
        fvtkout.write('\nVECTORS first_dir float')
        #print v.shape, d.shape
        v[:, :, 2] = v[:,:,2] * np.sqrt(d[:,2]).reshape([d.shape[0], 1])
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(v[ll, 0, 2], v[ll, 1, 2], v[ll, 2, 2]))

        fvtkout.write('\nVECTORS second_dir float')
        v[:, :, 1] *= np.sqrt(d[:,1]).reshape([d.shape[0], 1])
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(v[ll, 0, 2], v[ll, 1, 2], v[ll, 2, 2]))

        fvtkout.write('\nVECTORS third_dir float')
        v[:, :, 0] *= np.sqrt(d[:,2]*d[:,1]).reshape([d.shape[0], 1])
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(v[ll, 0, 2], v[ll, 1, 2], v[ll, 2, 2]))

        # fvtkout.write('\nTENSORS tensors float')
        # for ll in range(S.shape[0]):
        #     for kk in range(3):
        #         fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(S[ll, kk, 0], S[ll, kk, 1], S[ll, kk, 2]))
        fvtkout.write('\n')




def multiMatDet1(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[1]
    if (dim==1):
        detR = S
    elif (dim == 2):
        detR = np.multiply(S[:,0, 0], S[:, 1, 1]) - np.multiply(S[:,0, 1], S[:, 1, 0])
    elif (dim==3):
        detR = (S[:, 0, 0] * S[:, 1, 1] * S[:, 2, 2] 
                -S[:, 0, 0] * S[:, 1, 2] * S[:, 2, 1]
                -S[:, 0, 1] * S[:, 1, 0] * S[:, 2, 2]
                -S[:, 0, 2] * S[:, 1, 1] * S[:, 2, 0]
                +S[:, 0, 1] * S[:, 1, 2] * S[:, 2, 0]
                +S[:, 0, 2] * S[:, 1, 0] * S[:, 2, 1])
    return detR

def multiMatEig(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[1]
    d = np.zeros([N,dim])
    v = np.zeros([N,dim,dim])
    for k in range(N):
        D, V = LA.eig(S[k,:,:])
        idx = D.argsort()
        d[k,:] = D[idx]
        v[k,:, :] = V[:,idx]
    return d,v
        
def multiMatInverse1(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[1]
    if (dim==1):
        R = np.divide(1, S)
        detR = S
    elif (dim == 2):
        detR = np.multiply(S[:,0, 0], S[:, 1, 1]) - np.multiply(S[:,0, 1], S[:, 1, 0])
	R = np.zeros(S.shape)
        R[:, 0, 0] = S[:, 1, 1].copy()
        R[:, 1, 1] = S[:, 0, 0].copy()
        R[:, 0, 1] = -S[:, 0, 1]
        R[:, 1, 0] = -S[:, 1, 0]
        R = R / detR.reshape([N, 1, 1])
    elif (dim==3):
        detR = (S[:, 0, 0] * S[:, 1, 1] * S[:, 2, 2] 
                -S[:, 0, 0] * S[:, 1, 2] * S[:, 2, 1]
                -S[:, 0, 1] * S[:, 1, 0] * S[:, 2, 2]
                -S[:, 0, 2] * S[:, 1, 1] * S[:, 2, 0]
                +S[:, 0, 1] * S[:, 1, 2] * S[:, 2, 0]
                +S[:, 0, 2] * S[:, 1, 0] * S[:, 2, 1])
            #detR = np.divide(1, detR)
	R = np.zeros(S.shape)
        R[:, 0, 0] = S[:, 1, 1] * S[:, 2, 2] - S[:, 1, 2] * S[:, 2, 1]
        R[:, 1, 1] = S[:, 0, 0] * S[:, 2, 2] - S[:, 0, 2] * S[:, 2, 0]
        R[:, 2, 2] = S[:, 1, 1] * S[:, 0, 0] - S[:, 1, 0] * S[:, 0, 1]
        R[:, 0, 1] = -S[:, 0, 1] * S[:, 2, 2] + S[:, 2, 1] * S[:, 0, 2]
        R[:, 0, 2] = S[:, 0, 1] * S[:, 1, 2] - S[:, 0, 2] * S[:, 1, 1]
        R[:, 1, 2] = -S[:, 0, 0] * S[:, 1, 2] + S[:, 0, 2] * S[:, 1, 0]
        if isSym:
            R[:, 1, 0] = R[:, 0, 1].copy()
            R[:, 2, 0] = R[:, 0, 2].copy()
            R[:, 2, 1] = R[:, 1, 2].copy()
        else:
            R[:, 1, 0] = -S[:, 1, 0] * S[:, 2, 2] + S[:, 1, 2] * S[:, 2, 0]
            R[:, 2, 0] = S[:, 1, 0] * S[:, 2, 1] - S[:, 2, 0] * S[:, 1, 1]
            R[:, 2, 1] = -S[:, 0, 0] * S[:, 2, 1] + S[:, 2, 0] * S[:, 0, 1]
        R = R / detR.reshape([N, 1, 1])
    return R, detR
        
def multiMatInverse2(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[2] ;
    if (dim==1):
        R = np.divide(1, S)
        detR = S
    elif (dim == 2):
	R = np.zeros([N, N, dim, dim])
        detR = np.multiply(S[:, :,0, 0], S[:, :, 1, 1]) - np.multiply(S[:, :,0, 1], S[:, :, 1, 0])
        R[:, :, 0, 0] = S[:, :, 1, 1].copy()
        R[:, :, 1, 1] = S[:, :, 0, 0].copy()
        R[:, :, 0, 1] = -S[:, :, 0, 1]
        R[:, :, 1, 0] = -S[:, :, 1, 0]
        R = R / detR.reshape([N, N, 1, 1])
    elif (dim==3):
	R = np.zeros([N, N, dim, dim])
        detR = (S[:, :, 0, 0] * S[:, :, 1, 1] * S[:, :, 2, 2] 
                -S[:, :, 0, 0] * S[:, :, 1, 2] * S[:, :, 2, 1]
                -S[:, :, 0, 1] * S[:, :, 1, 0] * S[:, :, 2, 2]
                -S[:, :, 0, 2] * S[:, :, 1, 1] * S[:, :, 2, 0]
                +S[:, :, 0, 1] * S[:, :, 1, 2] * S[:, :, 2, 0]
                +S[:, :, 0, 2] * S[:, :, 1, 0] * S[:, :, 2, 1])
            #detR = np.divide(1, detR)
        R[:, :, 0, 0] = S[:, :, 1, 1] * S[:, :, 2, 2] - S[:, :, 1, 2] * S[:, :, 2, 1]
        R[:, :, 1, 1] = S[:, :, 0, 0] * S[:, :, 2, 2] - S[:, :, 0, 2] * S[:, :, 2, 0]
        R[:, :, 2, 2] = S[:, :, 1, 1] * S[:, :, 0, 0] - S[:, :, 1, 0] * S[:, :, 0, 1]
        R[:, :, 0, 1] = -S[:, :, 0, 1] * S[:, :, 2, 2] + S[:, :, 2, 1] * S[:, :, 0, 2]
        R[:, :, 0, 2] = S[:, :, 0, 1] * S[:, :, 1, 2] - S[:, :, 0, 2] * S[:, :, 1, 1]
        R[:, :, 1, 2] = -S[:, :, 0, 0] * S[:, :, 1, 2] + S[:, :, 0, 2] * S[:, :, 1, 0]
        if isSym:
            R[:, :, 1, 0] = R[:, :, 0, 1].copy()
            R[:, :, 2, 0] = R[:, :, 0, 2].copy()
            R[:, :, 2, 1] = R[:, :, 1, 2].copy()
        else:
            R[:, :, 1, 0] = -S[:, :, 1, 0] * S[:, :, 2, 2] + S[:, :, 1, 2] * S[:, :, 2, 0]
            R[:, :, 2, 0] = S[:, :, 1, 0] * S[:, :, 2, 1] - S[:, :, 2, 0] * S[:, :, 1, 1]
            R[:, :, 2, 1] = -S[:, :, 0, 0] * S[:, :, 2, 1] + S[:, :, 2, 0] * S[:, :, 0, 1]
        R = R / detR.reshape([N, N, 1, 1])
    return R, detR
        



def computeProducts(c, S, sig):
    M = c.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    SS = sigEye.reshape([1,dim,dim]) + S 
    detR = multiMatDet1(SS, isSym=True) 
    SS = sigEye.reshape([1,1,dim,dim]) + S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim])
    (R2, detR2) = multiMatInverse2(SS, isSym=True)
    
    diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
    betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt((detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*np.exp(-dst/2)
    
    return gcc


def gaussianDiffeonsGradientMatrices(x, c, S, a, px, pc, pS, sig, timeStep):
    N = x.shape[0]
    M = c.shape[0]
    dim = x.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    SS = sigEye.reshape([1,dim,dim]) + S 
    (R, detR) = multiMatInverse1(SS, isSym=True) 
    SS = sigEye.reshape([1,1,dim,dim]) + S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim])
    (R2, detR2) = multiMatInverse2(SS, isSym=True)

    diff = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
    betax = (R.reshape([1, M, dim, dim])*diff.reshape([N, M, 1, dim])).sum(axis=3)
    dst = (diff * betax).sum(axis=2)
    fx = np.exp(-dst/2)

    diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
    betac = (R.reshape([1, M, dim, dim])*diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (diffc * betac).sum(axis=2)
    fc = np.exp(-dst/2)
    betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt((detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*np.exp(-dst/2)

    Dv = -((fc.reshape([M,M,1])*betac).reshape([M, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
    IDv = np.eye(dim).reshape([1,dim,dim]) + timeStep * Dv ;
    pSS = (pS.reshape([M,dim,dim,1]) * (IDv.reshape([M,dim,dim, 1]) * S.reshape([M, 1, dim ,dim])).sum(axis=2).reshape([M,1,dim,dim])).sum(axis=2)
    
    fS = (pSS.reshape([M, 1, dim, dim])*betac.reshape([M,M,1,dim])).sum(axis=3)
    #fS = (pSS.reshape([M, 1, dim, dim])*betac.reshape([M,M,dim, 1])).sum(axis=2)
    #fS = (pS.reshape([M, 1, dim,dim])* fS.reshape([M,M,1,dim])).sum(axis=3)
    grx = np.dot(fx.T, px)
    grc = np.dot(fc.T, pc)
    grS = -2 * (fc.reshape([M,M,1]) * fS).sum(axis=0)
    return grx, grc, grS, gcc

