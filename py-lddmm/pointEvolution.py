import numpy as np
import kernelFunctions as kfun
import gaussianDiffeons as gd
import numpy.linalg as LA

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
        a = np.mat(np.squeeze(at[k, :, :]))
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



def gaussianDiffeonsEvolutionEuler(x0, c0, S0, at, sigma, affine = None, withJacobian=False, withPointSet=None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = c0.shape[0]
    T = at.shape[0] + 1
    timeStep = 1.0/(T-1)
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
        yt[0, :, :] = withPointSet

    sigEye = sig2*np.eye(dim)
    for t in range(T-1):
        x = np.squeeze(xt[t, :, :])
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, :, :, :])
        a = np.squeeze(at[t, :, :])

        (R, detR) = gd.multiMatInverse1(sigEye.reshape([1,dim,dim]) + S, isSym=True)

        diff = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
        betax = (R.reshape([1, M, dim, dim])*diff.reshape([N, M, 1, dim])).sum(axis=3)
        dst = (betax * diff).sum(axis=2)
        fx = np.exp(-dst/2)
        zx = np.dot(fx, a)

        diff = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
        betac = (R.reshape([1, M, dim, dim])*diff.reshape([M, M, 1, dim])).sum(axis=3)
        dst = (diff * betac).sum(axis=2)
        fc = np.exp(-dst/2)
        zc = np.dot(fc, a)

        Dv = -((fc.reshape([M,M,1])*betac).reshape([M, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
        if not (affine == None):
            Dv = Dv + A[t].reshape([1,dim,dim])
        SDvT = (S.reshape([M,dim,1,dim])*Dv.reshape([M,1,dim, dim])).sum(axis=3)
        zS = SDvT.transpose([0,2,1]) + SDvT
        zScorr = (SDvT.reshape([M, 1, dim, dim])*Dv.reshape([M, dim, dim, 1])).sum(axis=2) 

        xt[t+1, :, :] = x + timeStep * zx
        if not (affine == None):
            xt[t+1, :, :] += timeStep * (np.dot(x, A[t].T) + b[t])

        ct[t+1, :, :] = c + timeStep * zc
        if not (affine == None):
            ct[t+1, :, :] += timeStep * (np.dot(c, A[t].T) + b[t])

        St[t+1, :, :, :] = S + timeStep * zS + (timeStep**2) * zScorr

        if not (withPointSet == None):
            y = np.squeeze(yt[t, :, :])
            K = y.shape[0]
            diff = y.reshape([K, 1, dim]) - c.reshape([1, M, dim])
            dst = (diff * (R.reshape([1, M, dim, dim])*diff.reshape[K, M, 1, dim]).sum(axis=3)).sum(axis=2)
            fy = np.exp(-dst/2)
            zy = np.dot(fy, a)
            yt[t+1, :, :] = y + timeStep * zy
            if not (affine == None):
                yt[t+1, :, :] += timeStep * (np.dot(y, A[t].T) + b[t])

        if withJacobian:
            Div = np.multiply(fx, (betax * a.reshape(1,M, dim)).sum(axis=2)).sum(axis=1)
            Jt[t+1, :] = Jt[t, :] + timeStep * Div
            if not (affine == None):
                Jt[t+1, :] += timeStep * (np.trace(A[t]))

    if simpleOutput:
        return xt, ct, St
    else:
        output = [xt, ct, St]
        if not (withPointSet==None):
            output.append(yt)
        # if not (withNormals==None):
        #     output.append(nt)
        if withJacobian:
            output.append(Jt)
        return output



# backwards covector evolution along trajectory associated to x0, at
def gaussianDiffeonsCovector(x0, c0, S0,  at, px1, pc1, pS1, sigma, regweight, affine = None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0/T
    (xt, ct, St) = gaussianDiffeonsEvolutionEuler(x0, c0, S0, at, sigma, affine=affine)
    pxt = np.zeros([T, N, dim])
    pct = np.zeros([T, M, dim])
    pSt = np.zeros([T, M, dim, dim])
    pxt[T-1, :, :] = px1
    pct[T-1, :, :] = pc1
    pSt[T-1, :, :, :] = pS1
    sig2 = sigma*sigma

    if not(affine == None):
        A = affine[0]

    sigEye = sig2*np.eye(dim)
    for t in range(T-1):
        px = np.squeeze(pxt[T-t-1, :, :])
        pc = np.squeeze(pct[T-t-1, :, :])
        pS = np.squeeze(pSt[T-t-1, :, :])
        x = np.squeeze(xt[T-t-1, :, :])
        c = np.squeeze(ct[T-t-1, :, :])
        S = np.squeeze(St[T-t-1, :, :, :])
        a = np.squeeze(at[T-t-1, :, :])

        SS = S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim]) 
        (R, detR) = gd.multiMatInverse1(sigEye.reshape([1,dim,dim]) + S, isSym=True) 
        (R2, detR2) = gd.multiMatInverse2(sigEye.reshape([1,1,dim,dim]) + SS, isSym=True) 

        diff = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
        betax = (R.reshape([1, M, dim, dim])*diff.reshape([N, M, 1, dim])).sum(axis=3)
        dst = (diff * betax).sum(axis=2)
        fx = np.exp(-dst/2)

        diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
        betac = (R.reshape([1, M, dim, dim])*diffc.reshape([M, M, 1, dim])).sum(axis=3)
        dst = (diffc * betac).sum(axis=2)
        fc = np.exp(-dst/2)

        aa = np.dot(a, a.T)
        pxa = np.dot(px, a.T)
        pca = np.dot(pc, a.T)

        betaSym = betac.reshape([M, M, dim, 1]) * betac.reshape([M, M, 1, dim])
        betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
        betaSymcc = betacc.reshape([M, M, dim, 1]) * betacc.reshape([M, M, 1, dim])
        dst = (betacc * diffc).sum(axis=2)
        gcc = np.sqrt((detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*np.exp(-dst/2)
        psa = (pS.reshape([M,1,dim, dim]) * a.reshape([1, M, 1, dim])).sum(axis=3)
        spsa = (S.reshape([M, 1, dim, dim]) * psa.reshape([M, M, 1, dim])).sum(axis=3)
        #print np.fabs(betacc + betacc.transpose([1,0,2])).sum()

        u = (pxa * fx).reshape([N, M, 1]) * betax
        zpx = u.sum(axis=1)
        zpc = - u.sum(axis=0)
        u2 = (pca * fc).reshape([M, M, 1]) * betac
        zpc += u2.sum(axis=1) - u2.sum(axis=0)

        BmA = betaSym - R.reshape([1, M, dim, dim])
        u = fc.reshape([M, M, 1]) * (BmA *spsa.reshape([M, M, 1, dim])).sum(axis=3)
        zpc -= 2 * (u.sum(axis=1) - u.sum(axis=0))
        zpc -= 2 * (np.multiply(gcc, aa).reshape([M, M, 1]) * betacc).sum(axis=1)

        zpS = - 0.5 * (np.multiply(fx,pxa).reshape([N,M,1,1]) * (betax.reshape([N,M,dim,1]) * betax.reshape([N,M,1,dim]))).sum(axis=0)
        zpS -= 0.5 * (np.multiply(fc,pca).reshape([M,M,1,1]) * betaSym).sum(axis=0)
        abeta = ((fc.reshape([M,M,1])*betac).reshape([M, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
        #abeta = (fc.reshape([M,M,1,1]) * (a.reshape([1, M, dim, 1]) * betac.reshape([M,M,1,dim]))).sum(axis=1)
        abeta = (pS.reshape([M, dim, dim, 1])*abeta.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS += abeta + np.transpose(abeta, (0, 2, 1))
        u = np.multiply(fc, (spsa * betac).sum(axis=2))
        zpS += (u.reshape([M,M,1,1])*betaSym).sum(axis=0)
        u = (fc.reshape([M,M,1,1]) * spsa.reshape([M, M, dim,1]) * betac.reshape([M,M,1,dim])).sum(axis=0)
        u = (R.reshape([M,dim, dim, 1]) * u.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS -= u + u.transpose((0,2,1))
        zpS += (np.multiply(gcc, aa).reshape([M,M,1,1]) *(betaSymcc - R2 + R.reshape([M,1,dim,dim]))).sum(axis=1)

        pxt[T-t-2, :, :] = px - timeStep * zpx
        pct[T-t-2, :, :] = pc - timeStep * zpc
        pSt[T-t-2, :, :, :] = pS - timeStep * zpS

        if not (affine == None):
            pxt[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pxt[T-t-1, :, :]), A[T-t-1])
            pct[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pct[T-t-1, :, :]), A[T-t-1])
            pSt[T-t-2, :, :, :] -= timeStep * ((A[T-t-1].reshape([1,dim,dim,1])*pSt[T-t-1, :, :, :].reshape([M,1,dim,dim])).sum(axis=2) + (pSt[T-t-1, :, :, :].reshape([M,dim,1,dim])*A[T-t-1].reshape([1,1,dim,dim])).sum(axis=3))
    return pxt, pct, pSt, xt, ct, St

    
# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def gaussianDiffeonsGradient(x0, c0, S0, at, px1, pc1, pS1, sigma, regweight, getCovector = False, affine = None):
    (pxt, pct, pSt, xt, ct, St) = gaussianDiffeonsCovector(x0, c0, S0, at, px1, pc1, pS1, sigma, regweight, affine=affine)

    dat = np.zeros(at.shape)
    M = c0.shape[0]
    dim = c0.shape[1]
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
        [grx, grc, grS, gcc] = gd.gaussianDiffeonsGradientMatrices(x, c, S, px, pc, pS, sigma)
        
        da = 2*np.dot(gcc,a) - grx - grc - grS
        if not (affine == None):
            dA[t] = np.dot(px.T, x) + np.dot(pc.T, c) - 2*np.multiply(pS.reshape([M, dim, dim, 1]), S.reshape([M, dim, 1, dim])).sum(axis=1).sum(axis=0)
            db[t] = px.sum(axis=0) + pc.sum(axis=0)

        (L, W) = LA.eigh(gcc)
        #dat[t, :, :] = LA.solve(gcc+(L.max()/10000)*np.eye(M), da)
        dat[t, :, :] = LA.solve(gcc, da)

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
        px = np.squeeze(pxt[M-t-1, :, :])
        z = np.squeeze(xt[M-t-1, :, :])
        a = np.squeeze(at[M-t-1, :, :])
        # dgzz = kfun.kernelMatrix(KparDiff, z, diff=True)
        # if (isfield(KparDiff, 'zs') && size(z, 2) == 3)
        #     z(:,3) = z(:,3) / KparDiff.zs ;
        # end
        a1 = [px, a, -2*regweight*a]
        a2 = [a, px, a]
        #print 'test', px.sum()
        zpx = KparDiff.applyDiffKT(z, a1, a2)
        pxt[M-t-2, :, :] = np.squeeze(pxt[M-t-1, :, :]) + timeStep * zpx
        #print 'zpx', np.fabs(zpx).sum(), np.fabs(px).sum(), z.sum()
        #print 'pxt', np.fabs((pxt)[M-t-2]).sum()
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
        px = np.squeeze(pxt[M-t-1, :, :])
        z = np.squeeze(xt[M-t-1, :, :])
        a = np.squeeze(at[M-t-1, :, :])
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
        #print 'testgr', (2*a-px).sum()
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

