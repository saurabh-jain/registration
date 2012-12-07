import numpy as np
import kernelFunctions as kfun
from gaussianDiffeons import *

# Solves dx/dt = K(x,x) a(t) + A(t) x + b(t) with x(0) = x0
# affine: affine component [A, b] or None
# if withJacobian =True: return Jacobian determinant
# if withNormal = nu0, returns nu(t) evolving as dnu/dt = -Dv^{T} nu
def landmarkDirectEvolutionEuler(x0, at, KparDiff, affine = None, withJacobian=False, withNormals=None, withPointSet=None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0] + 1
    timeStep = 1.0/(M-1)
    xt = np.zeros([M, N, dim])
    xt[0, :, :] = x0
    simpleOutput = True
    if not (withNormals==None):
        simpleOutput = False
        nt = np.zeros([M, N, dim])
        nt[0, :, :] = withNormals
    if withJacobian:
        simpleOutput = False
        Jt = np.zeros([M, N])
    if not(affine == None):
        A = affine[0]
        b = affine[1]
    if not (withPointSet==None):
        simpleOutput = False
        K = withPointSet.shape[0]
        yt = np.zeros([M,K,dim])
        yt[:, :, :] = withPointSet

    for k in range(M-1):
        z = np.squeeze(xt[k, :, :])
        # gzz = kfun.kernelMatrix(KparDiff, z)
        a = np.mat(np.squeeze(at[k, :, :]))
        # xt[k+1, :, :] = z + timeStep * (gzz*a)
        xt[k+1, :, :] = z + timeStep * KparDiff.applyK(z, a)
        if not (affine == None):
            xt[k+1, :, :] += timeStep * (np.dot(z, A[k].T) + b[k])
        if not (withPointSet == None):
            zy = np.squeeze(yt[k, :, :])
            yt[k+1, :, :] = zy + timeStep * KparDiff.applyK(zy, a, y=z)
            if not (affine == None):
                yt[k+1, :, :] += timeStep * (np.dot(zy, A[k].T) + b[k])

        if not (withNormals==None):
            zn = np.squeeze(nt[k, :, :])        
            nt[k+1, :, :] = zn - timeStep * KparDiff.applyDiffKT(z, [np.mat(zn)], [a]) 
            if not (affine == None):
                nt[k+1, :, :] += timeStep * np.dot(zn, A[k])
        if withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a)
            if not (affine == None):
                Jt[k+1, :] += timeStep * (np.trace(A[k]))
        #print x0.sum()*11, xt.sum()
    if simpleOutput:
        return xt
    else:
        output = [xt]
        if not (withPointSet==None):
            output.append(yt)
        if not (withNormals==None):
            output.append(nt)
        if withJacobian:
            output.append(Jt)
        return output
    # if withJacobian:
    #     if not (withNormals==None):
    #         return xt, nt, Jt
    #     else:
    #         return xt, Jt
    # else:
    #     if not (withNormals==None):
    #         return xt, nt
    #     else:
    #         return xt






def gaussianDiffeonsEvolutionEuler(x0, c0, S0, at, sigma, affine = None, withJacobian=False, withPointSet=None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = c0.shape[0]
    T = at.shape[0] + 1
    timeStep = 1.0/(M-1)
    xt = np.zeros([T, N, dim])
    ct = np.zeros([T, M, dim])
    St = np.zeros([T, M, dim, dim])
    xt[0, :, :] = x0
    ct[0, :, :] = c0
    St[0, :, :, :] = S0
    simpleOutput = True
    sig2 = sigma*sigma

    if withJacobian:
        simpleOutput = False
        Jt = np.zeros([T, N])
        
    if not(affine == None):
        A = affine[0]
        b = affine[1]
        
    if not (withPointSet==None):
        simpleOutput = False
        K = withPointSet.shape[0]
        yt = np.zeros([T,K,dim])
        yt[:, :, :] = withPointSet

    bigEye = np.tile(sig2*np.eye(dim).reshape([1, dim, dim]),[M, 1, 1])
    for t in range(T-1):
        x = np.squeeze(xt[t, :, :])
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, :, :, :])
        a = np.mat(np.squeeze(at[t, :, :]))

        (R, detR) = multiMatInverse1(bigEye + S, isSym=True) 

        #        for k in range(M):
        #   R[k, :, :] = LA.inv(sig2*np.eye(dim) + np.squeeze(S[k, :, :]))

        diff = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
        dst = (diff * (R.reshape([1, M, dim, dim])*diff.reshape[N, M, 1, dim]).sum(axis=3)).sum(axis=2)
        fx = exp(-dst/2)
        # for j in range(N):
        #     for l in range(M):
        #         tmp = x[j,:] - c[l,:]
        #         dst = np.multiply(tmp, np.dot(R[l, :, :], tmp))
        #         fx[j,l] = exp(-dst/2)
        zx = np.dot(fx, a)

        diff = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
        beta = R.reshape([1, M, dim, dim])*diff.reshape([M, M, 1, dim]).sum(axis=3)
        dst = (diff * beta).sum(axis=2)
        fc = exp(-dst/2)
        # for k in range(M):
        #     for l in range(M):
        #         tmp = c[k, :] - c[l, :]
        #         dst = np.multiply(tmp, np.dot(R[l, :, :], tmp))
        #         fc[k,l] = exp(-dst/2) ;
        zc = np.dot(fc, a)

        Dv = -(np.multiply(fc, beta).reshape[M, M, 1, dim]*a.reshape([1, M, dim, 1])).sum(axis=1)
        SDvT = (S*Dv).sum(axis=2)
        zS = (Dv.reshape([M,dim, dim, 1])*S.reshape([M, 1, dim, dim])).sum(axis=2) + SDvT
        zScorr = (SDvT.reshape([M,dim, dim, 1])*S.reshape([M, 1, dim, dim])).sum(axis=2) 

        # for k in range(M):
        #     Dv[k, :, :] = np.zeros([dim, dim]) ;
        #     for l in range(M):
        #         tmp = c[k, :] - c[l, :]
        #     beta = np.dot(R[l, :], tmp)
        #     Dv[k] = np.dot(np.multiply(fc[k, :].reshape([M,1]), a).T, beta)
        #     ## need to check
        #     print Dv[k].shape

        #     zS[k] = -np.dot(Dv[k, :, :], S[k, :, :]) - np.dot(S[k, :, :], Dv[k,:, :].T)
        #     Zscorr[k] = np.dot(Dv[k, :, :], np.dot(S[k, :, :], Dv[k,:, :].T))

        xt[t+1, :, :] = x + timeStep * zx
        if not (affine == None):
            xt[t+1, :, :] += timeStep * (np.dot(x, A[t].T) + b[t])

        ct[t+1, :, :] = c + timeStep * zc
        if not (affine == None):
            ct[t+1, :, :] += timeStep * (np.dot(c, A[t].T) + b[t])

        St[t+1, :, :] = S + timeStep * zS + (timeStep**2) * zScorr
        if not (affine == None):
            ## Probably wrong
            AS = (A[t].reshape([1, dim, dim, 1]) * S.reshape([M,1, dim, dim])).sum(axis = 2)
            SAT = (A[t].reshape([1,dim,dim]) * S).sum(axis=1)
            S[t+1, :, :] += timeStep * (AS + SAT) + (timeStep**2) * (SAT.reshape([M,dim, dim, 1])*S.reshape([M, 1, dim, dim])).sum(axis=2)
                  
            # S1 = np.multiply(S, A[t].T.reshape([1,dim,dim])).sum(axis=1) 
            # S2 = np.multiply(A[t].reshape([1, dim, dim]), S).sum(axis=1)
            # St[t+1, :, :] += timeStep * (S1 + S2) +  (timeStep**2) np.multiply(S2, A[t].T.reshape([1,dim,dim])).sum(axis=1) 

        if not (withPointSet == None):
            y = np.squeeze(yt[t, :, :])
            K = y.shape[0]
            diff = y.reshape([K, 1, dim]) - c.reshape([1, M, dim])
            dst = (diff * (R.reshape([1, M, dim, dim])*diff.reshape[K, M, 1, dim]).sum(axis=3)).sum(axis=2)
            fy = exp(-dst/2)
            # for j in range(y.shape[0]):
            #     for l in range(M):
            #         tmp = y[j, :] - c[l, :]
            #         dst = np.multiply(tmp, np.dot(R[l, :, :], tmp))
            #         fy[j,l] = exp(-dst/2)
            zy = np.dot(fy, a)
            yt[t+1, :, :] = y + timeStep * zy
            if not (affine == None):
                yt[t+1, :, :] += timeStep * (np.dot(y, A[t].T) + b[t])

        if withJacobian:
            Div = np.multiply(fx, np.multiply(a,beta).sum(axis=1).reshape([1, M])).sum(axis=1)
            # np.zeros(N) ;
            # for k in range(N):
            #     #Dv[k] = np.zeros([dim, dim]) ;
            #     for l in range(M):
            #         tmp = x[k, :] - c[l, :]
            #         beta[l,:] = np.dot(R[l, :], tmp)
            #     Dv[k] = np.multiply(fx[k, :], np.multiply(a, beta).sum(axis=1)).sum()
            Jt[t+1, :] = Jt[t, :] + timeStep * Div
            if not (affine == None):
                Jt[t+1, :] += timeStep * (np.trace(A[t]))

    if simpleOutput:
        return xt
    else:
        output = [xt]
        if not (withPointSet==None):
            output.append(yt)
        if not (withNormals==None):
            output.append(nt)
        if withJacobian:
            output.append(Jt)
        return output



# backwards covector evolution along trajectory associated to x0, at
def gaussianDiffeonsCovector(x0, c0, S0,  at, px1, pc1, pS1, sigma, regweight, affine = None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0/M
    xt = gaussianDiffeonsEvolutionEuler(x0, c0, S0, at, sigma, affine=affine)
    pxt = np.zeros([T, N, dim])
    pct = np.zeros([T, M, dim])
    pSt = np.zeros([T, M, dim, dim])
    pxt[T-1, :, :] = px1
    pct[T-1, :, :] = pc1
    pSt[T-1, :, :] = pS1
    sig2 = sigma*sigma

    if not(affine == None):
        A = affine[0]

    bigEye = np.tile(sig2*np.eye(dim).reshape([1, dim, dim]),[M, 1, 1])
    bigEye2 = np.tile(sig2*np.eye(dim).reshape([1, 1, dim, dim]),[M, M, 1, 1])
    for t in range(T-1):
        px = np.mat(np.squeeze(pxt[T-t-1, :, :]))
        pc = np.mat(np.squeeze(pct[T-t-1, :, :]))
        pS = np.mat(np.squeeze(pSt[T-t-1, :, :]))
        x = np.mat(np.squeeze(xt[T-t-1, :, :]))
        c = np.mat(np.squeeze(ct[T-t-1, :, :]))
        S = np.mat(np.squeeze(St[T-t-1, :, :]))
        a = np.mat(np.squeeze(at[T-t-1, :, :]))

        SS = S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim]) 
        (R, detR) = multiMatInverse1(bigEye + S, isSym=True) 
        (R2, detR2) = multiMatInverse2(bigEye + SS, isSym=True) 

        
        # R = np.zeros([M, dim, dim])
        # R2 = np.zeros([M, M, dim, dim])
        # detR = np.zeros(M)
        # detR2 = np.zeros([M, M])
    
        diff = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
        betax = R.reshape([1, M, dim, dim])*diff.reshape([N, M, 1, dim]).sum(axis=3)
        dst = (diff * betax).sum(axis=2)
        fc = exp(-dst/2)

        diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
        betac = R.reshape([1, M, dim, dim])*diff.reshape([M, M, 1, dim]).sum(axis=3)
        dst = (diff * betac).sum(axis=2)
        fc = exp(-dst/2)
        pxa = np.dot(px, a.T)
        aa = np.dot(a, a.T)
        pca = np.dot(pc, a.T)

        # fx = np.zeros([N, M])
        # fc = np.zeros([M, M])
        # pxa = np.zeros([N, M])
        # pca = np.zeros([M, M])
        # aa = np.zeros([M, M])


        # betax = np.zeros([N, M, dim])
        # betac = np.zeros([M, M, dim])
        # betaSym = np.zeros([M, M, dim, dim])
        # spsa = np.zeros([M, M, dim])
        # gcc = np.zeros([M, M])
        # betacc = np.zeros([M, M, dim])
        # betaSymcc = np.zeros([M, M, dim, dim])

        # For k in range(M):
        #     R[k, :, :] = LA.inv(sig2*np.eye(dim) + np.squeeze(S[k, :, :]))
        #     #        detR[k] = invSym(dim, (const double*) tmpS, R[k]) ;

        # for k in range(M):
        #     for l in range(k, M):
        #         R2[k,l, :, :] = LA.inv(sig2 * np.eye(dim) + np.squeeze(S[k, :, :]) + np.squeeze(S[l, :, :]))
        #         detR2[k, l] = np.det(R2[k,l,:,:])
        #     for l in range(k):
        #         R2[k,l,:,:] = R2[l,k,:,:]
        #         detR2[k,l] = detR2[l,k]

        # for j in range(N):
        #     for l in range(M):
        #         tmp = x[j, :] - c[l, :]
        #         betax[j, l. :] = np.dot(R[l], tmp)
        #         dst = np.multiply(betax[j,l,:], tmp).sum()
        #         fx[j,l] = exp(-dst/2)

        betaSym = betac.reshape([M, M, dim, 1]) * betac.reshape([M, M, 1, dim])
        betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
        betaSymcc = betacc.reshape([M, M, dim, 1]) * betacc.reshape([M, M, 1, dim])
        dst = (betacc * diffc).sum(axis=2)
        gcc = np.sqrt(np.multiply(detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*exp(-dst/2)
        spsa = (S.reshape([M, 1, dim, dim]) * (pS.reshape([M,1,dim, dim]) * a.reshape([1, M, 1, dim])).sum(axis=3).reshape([M, M, 1, dim])).sum(axis=3)

        # for k in range(M):
        #     for l in range(M):
        #         tmp = c[k, :] - c[l, :]
        #         betac[k, l. :] = np.dot(R[l], tmp)
        #         betaSym[k, l, :, :] = np.multiply(betac[k,l,:].reshape([dim,1]), betac[k,l,:].reshape([1,dim])) 
        #         dst = np.multiply(betac[k,l,:], tmp).sum()
        #         fc[k,l] = exp(-dst/2)
        #         if (l<= k):
        #             betacc[k, l, :] = np.dot(R2[k,l,:,:], tmp)
        #             betaSymcc[k, l, :, :] = np.multiply(betacc[k,l,:].reshape([dim,1]), betacc[k,l,:].reshape([1,dim])) 
        #             dst = np.multiply(betacc[k,l,:], tmp).sum()
        #             gcc[k][l] = np.sqrt((detR[k]*detR[l])/((sig2**dim)*detR2[k,l]))*exp(-dst/2)

        #         spsa[k,l,:] = np.dot(S[k, :, :], np.dot(pS[k, : :], a[l, :]))
        # for k in range(M):
        #     for l in range(k+1, M):
        #         betacc[k, l, :] = - betacc[l,k,:]
        #         betaSymcc[k, l, :, :] = betaSymcc[l,k,:, :]
        #         gcc[k,l] = gcc[l,k]

        u = pxa.reshape([N,M,1]) * (fx.reshape([N, M, 1]) * betax)
        zpx = u.sum(axis=1)
        zpc = - u.sum(axis=0)
        u = pca.reshape([M,M,1]) * (fc.reshape([M, M, 1]) * betac)
        zpc += u.sum(axis=1) - u.sum(axis=0)

        BmA = betaSym - R.reshape([1, M, dim, dim])
        u = fc.reshape([M, M, 1]) * (Bma *spsa.reshape([M, M, 1, dim])).sum(axis=3)
        zpc -= 2 * (u.sum(axis=1) - u.sum(axis=0))
        zpc -= 2 * (np.multiply(gcc, aa).reshape([M, M, 1]) * betacc) 

        # for k in range(M):
        #     for l in range(M):
        #         tmp = np.dot(betaSym[k,l,:, :] - R[l, :, :], spsa[k,l,:])
        #         zpc[k, :] -= 2*fc[k,l]*tmp
        #         tmp = np.dot(betaSym[l,k,:, :] - R[k, :, :], spsa[l,k,:])
        #         zpc[k, :] += 2*fc[l,l]*tmp
        #         zpc[k, :] -= 2*gcc[k,l] * aa[k,l] * betacc[k, l, :]


        zpS = - (np.multiply(fx,pxa).reshape([N,M,1,1]) * (betax.reshape([N,M,dim,1]) * betax.reshape([N,M,1,dim]))).sum(axis=0)/2
        zpS = - (np.multiply(fc,pca).reshape([M,M,1,1]) * betaSym).sum(axis=0)/2
        abeta = a.reshape([1, M, dim, 1]) * betac.reshape([M,M,1,dim])
        abeta = abeta + np.transpose(abeta, (0,1,3,2))
        zpS += (fc.reshape(M, M, dim, dim) * abeta).sum(axis=1)
        pSa = (pS.reshape([M, 1, dim, dim]) * a.reshape([1, M, 1,dim])).sum(axis=3)
        Sbeta = (S.reshape([M, 1, dim, dim]) * beta.reshape([M, M, 1,dim])).sum(axis=3)
        zpS += (np.multiply(fc, (Sbeta * pSa).sum(axis=2)).reshape([M,M,1,1]) * betaSym).sum(axis=0)
        u = R.reshape([1,M,dim,dim]) * (spsa.reshape([M, M, dim,1]) * betac.reshape([M,M,1,dim]))
        zpS -= (u + np.transpose(u, (0,1,3,2))).sum(axis=0)
        zpS += (np.multiply(fc, aa).reshape([M,M,1,1]) *(betaSymcc - R2 + R.reshape([M,1,dim,dim]))).sum(axis=1)
        # for k in range(M):
        #     for l in range(M):
        #         TS = np.multiply(a[l, :].reshape([dim,1]), betac[k,l, :].reshape([1,dim]))
        #         tmpS = np.dot(pS[k, :, :], TS)
        #         zpS[k, :, :] += 2*fc[k.l]*tmpS
        #         u = np.multiply(spsa[l,k,:], betac[l,k,:]).sum()
        #         zpS[k, :, :] += fc[l,k]*u*betaSym[l,k,:]
        #         tmp = np.dot(R[k,:,:], spsa[l,k,:])
        #         zpS[k, :, :] -= fc[l,k] * (np.multiply(tmp.reshape([dim,1]), betac[l,k,:].reshape(1, dim)) + np.multiply(betac[l,k,:].reshape(1, dim), tmp.reshape([dim,1])))
        #         zpS[k, :, :] += gcc[k, l] * aa[k,l] * (betaSymcc[k, l, :, :] - R2[k, l, :, :] + R[k, :, :])

        pxt[T-t-2, :, :] = np.squeeze(pxt[T-t-1, :, :]) + timeStep * zpx
        pct[T-t-2, :, :] = np.squeeze(pct[T-t-1, :, :]) + timeStep * zpc
        pSt[T-t-2, :, :] = np.squeeze(pSt[T-t-1, :, :]) + timeStep * zpS
        if not (affine == None):
            pxt[T-t-2, :, :] += timeStep * np.dot(np.squeeze(pxt[T-t-1, :, :]), A[T-t-1])
            pct[T-t-2, :, :] += timeStep * np.dot(np.squeeze(pct[T-t-1, :, :]), A[T-t-1])
            pSt[T-t-2, :, :] = timeStep * (np.dot(A[T-t-1], np.squeeze(pSt[T-t-1, :, :])) + np.dot(np.squeeze(pSt[T-t-1, :, :]), A[T-t-1].T))
    return pxt, pct, pSt, xt, ct, St

    
# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def gaussianDiffeonsGradient(x0, c0, S0, at, px1, pc1, pS1, sigma, regweight, getCovector = False, affine = None):
    (pxt, pct, pSt, xt, ct, St) = gaussianDiffeonsCovector(x0, c0, S0, at, px1, pc1, pS1, sigma, regweight, affine=affine)
    dat = np.zeros(at.shape)
    if not (affine == None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for t in range(at.shape[0]):
        a = np.squeeze(at[t, :, :])
        x = np.squeeze(xt[t, :, :])
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, :, :])
        px = np.squeeze(pxt[t, :, :])
        pc = np.squeeze(pct[t, :, :])
        pS = np.squeeze(pSt[t, :, :])
        [grx, grc, grS, gcc] = gaussianDiffeonsGradientMatrices(x, c, S, px, pc, pS, sigma)
        da = 2*np.multiply(gcc,a) - grx - grc - grS
        if not (affine == None):
            dA[t] = np.dot(px.T, x) + np.dot(pc.T, c) - 2*np.multiply(pS.reshape([M, dim, dim, 1]), S.reshape([M, dim, 1, dim])).sum(axis=1).sum(axis=0)
            db[t] = px.sum(axis=0) + pc.sum(axis=0)

        (L, W) = LA.eigh(gcc)
        dat[t, :, :] = LA.solve(gcc+(L.max()/10000)*np.eye(M), da)

    if affine == None:
        if getCovector == False:
            return dat, xt, ct, St
        else:
            return dat, xt, ct, St, pxt, pct, pSt
    else:
        if getCovector == False:
            return dat, dA, db, xt, ct, St
        else:
            return dat, dA, db, xt, ct, St, pxt, pct, pSt



def landmarkHamiltonianCovector(x0, at, px1, KparDiff, regweight, affine = None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0]
    timeStep = 1.0/M
    xt = landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
    pxt = np.zeros([M, N, dim])
    pxt[M-1, :, :] = px1
    if not(affine == None):
        A = affine[0]

    for t in range(M-1):
        px = np.mat(np.squeeze(pxt[M-t-1, :, :]))
        z = np.mat(np.squeeze(xt[M-t-1, :, :]))
        a = np.mat(np.squeeze(at[M-t-1, :, :]))
        # dgzz = kfun.kernelMatrix(KparDiff, z, diff=True)
        # if (isfield(KparDiff, 'zs') && size(z, 2) == 3)
        #     z(:,3) = z(:,3) / KparDiff.zs ;
        # end
        a1 = [px, a, -2*regweight*a]
        a2 = [a, px, a]
        pxt[M-t-2, :, :] = np.squeeze(pxt[M-t-1, :, :]) + timeStep * KparDiff.applyDiffKT(z, a1, a2)
        if not (affine == None):
            pxt[M-t-2, :, :] += timeStep * np.dot(np.squeeze(pxt[M-t-1, :, :]), A[M-t-1])
    return pxt, xt

# Same, adding adjoint evolution for normals
def landmarkAndNormalsCovector(x0, n0, at, px1, pn1, KparDiff, regweight):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0]
    timeStep = 1.0/M
    (xt, nt) = landmarkAndNormalsEvolutionEuler(x0, at, KparDiff)
    pxt = np.zeros([M, N, dim])
    pxt[M-1, :, :] = px1
    pnt = np.zeros([M, N, dim])
    pnt[M-1, :, :] = pn1
    for t in range(M-1):
        px = np.mat(np.squeeze(pxt[M-t-1, :, :]))
        z = np.mat(np.squeeze(xt[M-t-1, :, :]))
        a = np.mat(np.squeeze(at[M-t-1, :, :]))
        # dgzz = kfun.kernelMatrix(KparDiff, z, diff=True)
        # if (isfield(KparDiff, 'zs') && size(z, 2) == 3)
        #     z(:,3) = z(:,3) / KparDiff.zs ;
        # end
        a1 = [px, a, -2*regweight*a]
        a2 = [a, px, a]
        pxt[M-t-2, :, :] = np.squeeze(pxt[M-t-1, :, :]) + timeStep * KparDiff.applyDiffKT(z, a1, a2)

    return pxt, xt

# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def landmarkHamiltonianGradient(x0, at, px1, KparDiff, regweight, getCovector = False, affine = None):
    (pxt, xt) = landmarkHamiltonianCovector(x0, at, px1, KparDiff, regweight, affine=affine)
    dat = np.zeros(at.shape)
    if not (affine == None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for k in range(at.shape[0]):
        a = np.squeeze(at[k, :, :])
        px = np.squeeze(pxt[k, :, :])
        dat[k, :, :] = (2*regweight*a-px)
        if not (affine == None):
            dA[k] = np.dot(pxt[k].T, xt[k])
            db[k] = pxt[k].sum(axis=0)

    if affine == None:
        if getCovector == False:
            return dat, xt
        else:
            return dat, xt, pxt
    else:
        if getCovector == False:
            return dat, dA, db, xt
        else:
            return dat, dA, db, xt, pxt

