import numpy as np
import numpy.linalg as LA
import kernelFunctions as kfun

def generateDiffeons(fv, rate):
    nc = int(np.floor(fv.vertices.shape[0] * rate))
    (idx, c) = fv.laplacianSegmentation(nc)
    #print idx
    #print c.shape
    nc = idx.max()
    S = np.zeros([nc, 3, 3])
    C = np.zeros([nc, 3])
    for k in range(nc):
        I = np.flatnonzero(idx==k)
	#print I
	C[k, :] = fv.vertices[I, :].sum(axis=0)/len(I) ; 
        y = fv.vertices[I, :] - C[k, :]
        S[k, :, :] = np.dot(y.T, y)/len(I)
    return C, S
        
    
    

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
    #fS = (pS.reshape([M, 1, dim,dim])* fS.reshape([M,M,1,dim])).sum(axis=3)
    grx = np.dot(fx.T, px)
    grc = np.dot(fc.T, pc)
    grS = -2 * (fc.reshape([M,M,1]) * fS).sum(axis=0)
    return grx, grc, grS, gcc

