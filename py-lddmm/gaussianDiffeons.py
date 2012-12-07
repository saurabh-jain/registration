import numpy as np
import numpy.linalg as LA
import kernelFunctions as kfun

def computeDiffeons(fv, rate):
    nc = np.floor(fv.vertices.shape[0] * rate)
    (idx, c) = fv.laplacianSegmentation(nc)
    S = np.zeros([nc, 3, 3])
    for k in range(nc):
        I = np.nonzero(idx==k)
        y = fv.vertices[I, :] - c[k, :]
        S[k, :, :] = np.dot(y.T, y)/len(I)
        
    
    

def multiMatInverse1(S, isSym=False):
    dim = S.shape[1] ;
    if (dim==1):
        R = np.divide(1, S)
        detR = R
    elif (dim == 2):
        detR = np.divide(1, np.multiply(S[:,0, 0], S[:, 1, 1]) - np.multiply(S[:,0, 1], S[:, 1, 0]))
        R[:, 0, 0] = S[:, 1, 1].copy()
        R[:, 1, 1] = S[:, 0, 0].copy()
        R[:, 0, 1] = -S[:, 0, 1]
        R[:, 1, 0] = -S[:, 1, 0]
        R = np.multiply(R, detR)
    elif (dim==3):
        detR = (S[:, 0, 0] * S[:, 1, 1] * S[:, 2, 2] 
                -S[:, 0, 0] * S[:, 1, 2] * S[:, 2, 1]
                -S[:, 0, 1] * S[:, 1, 0] * S[:, 2, 2]
                -S[:, 0, 2] * S[:, 1, 1] * S[:, 2, 0]
                +S[:, 0, 1] * S[:, 1, 2] * S[:, 2, 0]
                +S[:, 0, 2] * S[:, 1, 0] * S[:, 2, 1])
        detR = np.divide(1, detR)
        R[:, 0, 0] = S[:, 1, 1] * S[:, 2, 2] - S[:, 1, 2] * S[:, 2, 1]
        R[:, 1, 1] = S[:, 0, 0] * S[:, 2, 2] - S[:, 0, 2] * S[:, 2, 0]
        R[:, 2, 2] = S[:, 1, 1] * S[:, 0, 0] - S[:, 1, 0] * S[:, 0, 1]
        R[:, 0, 1] = -S[:, 0, 1] * S[:, 2, 2] + S[:, 2, 1] * S[:, 0, 2]
        R[:, 0, 2] = S[:, 0, 1] * S[:, 1, 2] - S[:, 0, 2] * S[:, 1, 1]
        R[:, 1, 2] = -S[:, 0, 0] * S[:, 1, 2] + S[:, 0, 2] * S[:, 1, 0]
        if isSym:
            R[:, 1, 0] = R[:, 0, 1]
            R[:, 2, 0] = R[:, 0, 2]
            R[:, 2, 1] = R[:, 1, 2]
        else:
            R[:, 1, 0] = -S[:, 1, 0] * S[:, 2, 2] + S[:, 1, 2] * S[:, 2, 0]
            R[:, 2, 0] = S[:, 1, 0] * S[:, 2, 1] - S[:, 2, 0] * S[:, 1, 1]
            R[:, 2, 1] = -S[:, 0, 0] * S[:, 2, 1] + S[:, 2, 0] * S[:, 0, 1]
        R = np.multiply(R, detR)
    return R, detR
        
def multiMatInverse2(S, isSym=False):
    dim = S.shape[2] ;
    if (dim==1):
        R = np.divide(1, S)
        detR = R
    elif (dim == 2):
        detR = np.divide(1, np.multiply(S[:, :,0, 0], S[:, :, 1, 1]) - np.multiply(S[:, :,0, 1], S[:, :, 1, 0]))
        R[:, :, 0, 0] = S[:, :, 1, 1].copy()
        R[:, :, 1, 1] = S[:, :, 0, 0].copy()
        R[:, :, 0, 1] = -S[:, :, 0, 1]
        R[:, :, 1, 0] = -S[:, :, 1, 0]
        R = np.multiply(R, detR)
    elif (dim==3):
        detR = (S[:, :, 0, 0] * S[:, :, 1, 1] * S[:, :, 2, 2] 
                -S[:, :, 0, 0] * S[:, :, 1, 2] * S[:, :, 2, 1]
                -S[:, :, 0, 1] * S[:, :, 1, 0] * S[:, :, 2, 2]
                -S[:, :, 0, 2] * S[:, :, 1, 1] * S[:, :, 2, 0]
                +S[:, :, 0, 1] * S[:, :, 1, 2] * S[:, :, 2, 0]
                +S[:, :, 0, 2] * S[:, :, 1, 0] * S[:, :, 2, 1])
        detR = np.divide(1, detR)
        R[:, :, 0, 0] = S[:, :, 1, 1] * S[:, :, 2, 2] - S[:, :, 1, 2] * S[:, :, 2, 1]
        R[:, :, 1, 1] = S[:, :, 0, 0] * S[:, :, 2, 2] - S[:, :, 0, 2] * S[:, :, 2, 0]
        R[:, :, 2, 2] = S[:, :, 1, 1] * S[:, :, 0, 0] - S[:, :, 1, 0] * S[:, :, 0, 1]
        R[:, :, 0, 1] = -S[:, :, 0, 1] * S[:, :, 2, 2] + S[:, :, 2, 1] * S[:, :, 0, 2]
        R[:, :, 0, 2] = S[:, :, 0, 1] * S[:, :, 1, 2] - S[:, :, 0, 2] * S[:, :, 1, 1]
        R[:, :, 1, 2] = -S[:, :, 0, 0] * S[:, :, 1, 2] + S[:, :, 0, 2] * S[:, :, 1, 0]
        if isSym:
            R[:, :, 1, 0] = R[:, :, 0, 1]
            R[:, :, 2, 0] = R[:, :, 0, 2]
            R[:, :, 2, 1] = R[:, :, 1, 2]
        else:
            R[:, :, 1, 0] = -S[:, :, 1, 0] * S[:, :, 2, 2] + S[:, :, 1, 2] * S[:, :, 2, 0]
            R[:, :, 2, 0] = S[:, :, 1, 0] * S[:, :, 2, 1] - S[:, :, 2, 0] * S[:, :, 1, 1]
            R[:, :, 2, 1] = -S[:, :, 0, 0] * S[:, :, 2, 1] + S[:, :, 2, 0] * S[:, :, 0, 1]
        R = np.multiply(R, detR)
    return R, detR
        



def computeProducts(c, S, sig):
    M = c.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;

    bigEye = np.tile(sig2*np.eye(dim).reshape([1, dim, dim]),[M, 1, 1])
    bigEye2 = np.tile(sig2*np.eye(dim).reshape([1, 1, dim, dim]),[M, M, 1, 1])
    SS = S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim]) 
    (R, detR) = multiMatInverse1(bigEye + S, isSym=True) 
    (R2, detR2) = multiMatInverse2(bigEye + SS, isSym=True) 

    diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
    betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt(np.multiply(detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*exp(-dst/2)
    
    return gcc


def gaussianDiffeonsGradientMatrices(x, c, cS, px, pc, pS, sig):
    # R = np.zeros([M, dim, dim])
    # R2 = np.zeros([M, M, dim, dim])
    # detR = np.zeros(M)
    # detR2 = np.zeros([M, M])
    
    # fx = np.zeros([N, M])
    # fc = np.zeros([M, M])
    # fS = np.zeros([M, M, dim])
    N = x.shape[0]
    M = c.shape[0]
    dim = x.shape[1]
    sig2 = sig*sig ;

    bigEye = np.tile(sig2*np.eye(dim).reshape([1, dim, dim]),[M, 1, 1])
    bigEye2 = np.tile(sig2*np.eye(dim).reshape([1, 1, dim, dim]),[M, M, 1, 1])
    SS = S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim]) 
    (R, detR) = multiMatInverse1(bigEye + S, isSym=True) 
    (R2, detR2) = multiMatInverse2(bigEye + SS, isSym=True) 

    # for k in range(M):
    #     R[k, :, :] = LA.inv(sig2*np.eye(dim) + np.squeeze(S[k, :, :]))

    # for k in range(M):
    #     for l in range(k, M):
    #         R2[k,l, :, :] = LA.inv(sig2 * np.eye(dim) + np.squeeze(S[k, :, :]) + np.squeeze(S[l, :, :]))
    #         detR2[k, l] = np.det(R2[k,l,:,:])
    #     for l in range(k):
    #         R2[k,l,:,:] = R2[l,k,:,:]
    #         detR2[k,l] = detR2[l,k]


    diff = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
    betax = R.reshape([1, M, dim, dim])*diff.reshape([N, M, 1, dim]).sum(axis=3)
    dst = (diff * betax).sum(axis=2)
    fc = exp(-dst/2)

    diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
    betac = R.reshape([1, M, dim, dim])*diff.reshape([M, M, 1, dim]).sum(axis=3)
    dst = (diff * betac).sum(axis=2)
    fc = exp(-dst/2)
    # pxa = np.dot(px, a.T)
    # aa = np.dot(a, a.T)
    # pca = np.dot(pc, a.T)
    betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt(np.multiply(detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*exp(-dst/2)

    # for j in range(N):
    #     for l in range(M):
    #         tmp = x[j, :] - c[l, :]
    #         betax = np.dot(R[l], tmp)
    #         dst = np.multiply(betax, tmp).sum()
    #         fx[j,l] = exp(-dst/2)

    # for k in range(M):
    #     for l in range(M):
    #         tmp = c[k, :] - c[l, :]
    #         betac = np.dot(R[l], tmp)
    #         dst = np.multiply(betac, tmp).sum()
    #         fc[k,l] = exp(-dst/2)
    #         fS[k, l, :] = np.dot(pS[k, :, :], np.dot(S[k, :, :], betac))
    #         if (l<=k):
    #             betacc = np.dot(R2[k,l,:,:], tmp)
    #             dst = np.multiply(tmp.betacc).sum()
    #             gcc[k][l] = np.sqrt((detR[k]*detR[l])/((sig2**dim)*detR2[k,l]))*exp(-dst/2)

    # for k in range(M):
    #     for l in range(k+1, M):
    #         gcc[k,l] = gcc[l,k]
    
    
    grx = np.dot(fx.T, px) 
    grc = np.dot(fc.T, pc)
    grS = -2 * np.multiply(fc.reshape([M,M,1]), fS).sum(axis=0)
    return grc, grc, grS, gcc

