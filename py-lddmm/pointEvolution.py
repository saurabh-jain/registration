import numpy as np
import kernelFunctions as kfun
import gaussianDiffeons as gd
import numpy.linalg as LA
import affineBasis

def gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine = None, withJacobian=False, withPointSet=None, withNormals=None,
                                   withDiffeonSet=None):

    dim = c0.shape[1]
    M = c0.shape[0]
    T = at.shape[0] + 1
    timeStep = 1.0/(T-1)
    ct = np.zeros([T, M, dim])
    St = np.zeros([T, M, dim, dim])
    ct[0, :, :] = c0
    St[0, :, :, :] = S0
    simpleOutput = True
    sig2 = sigma*sigma


    if (withPointSet == None) and (withDiffeonSet==None):
        withJacobian = False
        withNormals = None

    if not(withPointSet == None):
        simpleOutput = False
        x0 = withPointSet
        N = x0.shape[0]
        xt = np.zeros(np.concatenate([[T], x0.shape]))
        xt[0, ...] = x0
        if type(withJacobian) == bool:
            if withJacobian:
                Jt = np.zeros(np.insert(x0.shape[0:-1], 0, T))
        elif not(withJacobian==None):
            #print withJacobian
            J0 = withJacobian
            Jt = np.zeros(np.insert(J0.shape, 0, T))
            Jt[0, ...] = J0
            withJacobian = True
        else:
            withJacobian=False
        withPointSet = True
                              
        if not(withNormals == None):
            b0 = withNormals
            bt = np.zeros(np.concatenate([[T], b0.shape]))
            bt[0, ...] = b0
            
    if not(affine == None):
        A = affine[0]
        b = affine[1]
        
    if not (withDiffeonSet==None):
        simpleOutput = False
        K = withDiffeonSet[0].shape[0]
        yt = np.zeros([T,K,dim])
        Ut = np.zeros([T,K,dim,dim])
        yt[0, :, :] = withDiffeonSet[0]
        Ut[0, :, :] = withDiffeonSet[1]
        if withJacobian:
            Jt = np.zeros([T, K])
        if not(withNormals == None):
            b0 = withNormals
            bt = np.zeros([T, K, dim])
            bt[0, :, :] = b0
            withNormals = True
        withDiffeonSet = True


    sigEye = sig2*np.eye(dim)
    for t in range(T-1):
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, :, :, :])
        a = np.squeeze(at[t, :, :])

        (R, detR) = gd.multiMatInverse1(sigEye.reshape([1,dim,dim]) + S, isSym=True)

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

        ct[t+1, :, :] = c + timeStep * zc
        if not (affine == None):
            ct[t+1, :, :] += timeStep * (np.dot(c, A[t].T) + b[t])

        St[t+1, :, :, :] = S + timeStep * zS + (timeStep**2) * zScorr
        #St[t+1, :, :, :] = S


        if withPointSet:
            x = np.squeeze(xt[t, ...])
            diffx = x[..., np.newaxis, :] - c.reshape([M, dim])
            betax = (R*diffx[..., np.newaxis, :]).sum(axis=-1)
            dst = (betax * diffx).sum(axis=-1)
            fx = np.exp(-dst/2)
            zx = np.dot(fx, a)
            xt[t+1, ...] = x + timeStep * zx
            if not (affine == None):
                xt[t+1, ...] += timeStep * (np.dot(x, A[t].T) + b[t])
            if withJacobian:
                #print Jt.shape
                Div = -(fx * (betax * a).sum(axis=-1)).sum(axis=-1)
                Jt[t+1, ...] = Jt[t, ...] + timeStep * Div
                if not (affine == None):
                    Jt[t+1, ...] += timeStep * (np.trace(A[t]))
            if withNormals:
                bb = np.squeeze(bt[t, ...])
                zb = ((fx * np.dot(bb, a.T))[..., np.newaxis] * betax).sum(axis=-2)
                zb -= (fx * (betax * a).sum(axis=-1)).sum(axis=-1)[..., np.newaxis] *bb
                bt[t+1,:,:] = bb + timeStep*zb

        # if not(withPointSet==None):
        #     x = np.squeeze(xt[t, ...])
        #     diffx = x.[N, 1, dim]) - c.reshape([1, M, dim])
        #     betax = (R.reshape([1, M, dim, dim])*diffx.reshape([N, M, 1, dim])).sum(axis=3)
        #     dst = (betax * diffx).sum(axis=2)
        #     fx = np.exp(-dst/2)
        #     zx = np.dot(fx, a)
        #     xt[t+1, :, :] = x + timeStep * zx
        #     if not (affine == None):
        #         xt[t+1, :, :] += timeStep * (np.dot(x, A[t].T) + b[t])
        #     if withJacobian:
        #         Div = -(fx * (betax * a.reshape(1,M, dim)).sum(axis=2)).sum(axis=1)
        #         Jt[t+1, :] = Jt[t, :] + timeStep * Div
        #         if not (affine == None):
        #             Jt[t+1, :] += timeStep * (np.trace(A[t]))
        #     if not(withNormals==None):
        #         bb = np.squeeze(bt[t, :, :])
        #         zb = ((fx * np.dot(bb, a.T)).reshape([N,M,1]) * betax).sum(axis=1)
        #         zb -= (fx * (betax * a.reshape([1,M,dim])).sum(axis=2)).sum(axis=1).reshape([N,1]) *bb
        #         bt[t+1,:,:] = bb + timeStep*zb


        if withDiffeonSet:
            y = np.squeeze(yt[t, :, :])
            U = np.squeeze(Ut[t, :, :, :])
            K = y.shape[0]
            diffy = y.reshape([K, 1, dim]) - c.reshape([1, M, dim])
            betay = (R.reshape([1, M, dim, dim])*diffy.reshape([K, M, 1, dim])).sum(axis=3)
            dst = (diffy * betay).sum(axis=2)
            fy = np.exp(-dst/2)
            zy = np.dot(fy, a)
            yt[t+1, :, :] = y + timeStep * zy
            if not (affine == None):
                yt[t+1, :, :] += timeStep * (np.dot(y, A[t].T) + b[t])
            Dvy = -((fy.reshape([K,M,1])*betay).reshape([K, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
            if not (affine == None):
                Dvy = Dvy + A[t].reshape([1,dim,dim])
            UDvT = (U.reshape([K,dim,1,dim])*Dvy.reshape([K,1,dim, dim])).sum(axis=3)
            zU = UDvT.transpose([0,2,1]) + UDvT
            zUcorr = (UDvT.reshape([K, 1, dim, dim])*Dvy.reshape([K, dim, dim, 1])).sum(axis=2) 
            Ut[t+1, :, :, :] = U + timeStep * zU + (timeStep**2) * zUcorr
            if withJacobian:
                Div = -(fy * (betay * a.reshape(1,M, dim)).sum(axis=2)).sum(axis=1)
                Jt[t+1, :] = Jt[t, :] + timeStep * Div
                if not (affine == None):
                    Jt[t+1, :] += timeStep * (np.trace(A[t]))
            if withNormals:
                bb = np.squeeze(bt[t, :, :])
                zb = ((fy * np.dot(bb, a.T)).reshape([K,M,1]) * betay).sum(axis=1)
                zb -= (fy * (betay * a.reshape([1,M,dim])).sum(axis=2)).sum(axis=1).reshape([K,1]) *bb
                bt[t+1,:,:] = bb + timeStep*zb


    if simpleOutput:
        return ct, St
    else:
        output = [ct, St]
        if withNormals:
            output.append(bt)
        if withPointSet:
            output.append(xt)
            if withJacobian:
                output.append(Jt)
        if withDiffeonSet:
            output.append(yt)
            output.append(Ut)
            if withJacobian:
                output.append(Jt)
        # if not (withNormals==None):
        #     output.append(nt)
        return output



# backwards covector evolution along trajectory associated to x0, at
def gaussianDiffeonsCovectorPset(c0, S0, x0,  at, pc1, pS1, px1, sigma, regweight, affine = None, withJacobian=None):
    dim = c0.shape[1]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0/T
    #print c0.shape, x0.shape
    if not(withJacobian==None):
        #print withJacobian
        J0 = withJacobian[0]
        pJ1 = withJacobian[1]
        withJacobian = True
    else:
        J0 = None
        withJacobian = False
    if withJacobian:
        (ct, St, xt, Jt) = gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=affine, withPointSet=x0, withJacobian=J0)
    else:
        (ct, St, xt) = gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=affine, withPointSet=x0)
    pct = np.zeros([T, M, dim])
    pSt = np.zeros([T, M, dim, dim])
    pxt = np.zeros(np.insert(x0.shape, 0, T))
    pxt[T-1, ...] = px1
    pct[T-1, :, :] = pc1
    pSt[T-1, :, :, :] = pS1
    if withJacobian:
        pJt = np.tile(pJ1, np.insert(np.ones(J0.ndim), 0, T))
        #pJt = np.zeros(np.insert(J0.shape, 0, T))
        #pJt[T-1, ...] = pJ1
    sig2 = sigma*sigma

    if not(affine == None):
        A = affine[0]

    sigEye = sig2*np.eye(dim)
    for t in range(T-1):
        px = np.squeeze(pxt[T-t-1, ...])
        pc = np.squeeze(pct[T-t-1, :, :])
        pS = np.squeeze(pSt[T-t-1, :, :])
        x = np.squeeze(xt[T-t-1, ...])
        c = np.squeeze(ct[T-t-1, :, :])
        S = np.squeeze(St[T-t-1, :, :, :])
        a = np.squeeze(at[T-t-1, :, :])

        SS = S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim]) 
        (R, detR) = gd.multiMatInverse1(sigEye.reshape([1,dim,dim]) + S, isSym=True) 
        (R2, detR2) = gd.multiMatInverse2(sigEye.reshape([1,1,dim,dim]) + SS, isSym=True) 

        diffx = x[...,np.newaxis,:] - c
        betax = (R*diffx[...,np.newaxis,:]).sum(axis=-1)
        dst = (diffx * betax).sum(axis=-1)
        fx = np.exp(-dst/2)

        diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
        betac = (R.reshape([1, M, dim, dim])*diffc.reshape([M, M, 1, dim])).sum(axis=3)
        dst = (diffc * betac).sum(axis=2)
        fc = np.exp(-dst/2)

        Dv = -((fc.reshape([M,M,1])*betac).reshape([M, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
        IDv = np.eye(dim).reshape([1,dim,dim]) + timeStep * Dv ;
        SpS = (S.reshape([M,dim,dim,1]) * (IDv.reshape([M,dim,dim, 1]) * pS.reshape([M, dim, 1 ,dim])).sum(axis=1).reshape([M,1,dim,dim])).sum(axis=2)

        aa = np.dot(a, a.T)
        pxa = np.dot(px, a.T)
        pca = np.dot(pc, a.T)

        betaxSym = betax[...,np.newaxis] * betax[...,np.newaxis,:]
        betaxa = (betax*a).sum(axis=-1)
        betaSym = betac.reshape([M, M, dim, 1]) * betac.reshape([M, M, 1, dim])
        betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
        betaSymcc = betacc.reshape([M, M, dim, 1]) * betacc.reshape([M, M, 1, dim])
        dst = (betacc * diffc).sum(axis=2)
        gcc = np.sqrt((detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*np.exp(-dst/2)
        spsa = (SpS.reshape([M,1,dim,dim])*a.reshape([1, M, 1, dim])).sum(axis=3)
        #print np.fabs(betacc + betacc.transpose([1,0,2])).sum()

        #print pxa.shape, fx.shape, betax.shape
        u = (pxa * fx)[...,np.newaxis] * betax
        zpx = u.sum(axis=-2)
        zpc = - u.sum(axis=tuple(range(x0.ndim-1)))
        u2 = (pca * fc)[...,np.newaxis] * betac
        zpc += u2.sum(axis=1) - u2.sum(axis=0)

        #BmA = betaSym - R.reshape([1, M, dim, dim])
        Ra = (R * a[:,np.newaxis,:]).sum(axis=2)
        BmA = betaSym - R
        u = fc.reshape([M, M, 1]) * (BmA *spsa.reshape([M, M, 1, dim])).sum(axis=3)
        zpc -= 2 * (u.sum(axis=1) - u.sum(axis=0))
        zpc -= 2 * (np.multiply(gcc, aa).reshape([M, M, 1]) * betacc).sum(axis=1)

        zpS = - 0.5 * ((fx*pxa)[..., np.newaxis,np.newaxis] * betaxSym).sum(axis=tuple(range(x0.ndim-1)))
        zpS -= 0.5 * (np.multiply(fc,pca).reshape([M,M,1,1]) * betaSym).sum(axis=0)
        pSDv = (pS.reshape([M,dim, dim, 1]) * Dv.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS += -pSDv - pSDv.transpose((0,2,1)) - timeStep * (Dv.reshape([M,dim, dim, 1]) * pSDv.reshape([M, dim, 1, dim])).sum(axis=1)
        u = np.multiply(fc, (spsa * betac).sum(axis=2))
        zpS += (u.reshape([M,M,1,1])*betaSym).sum(axis=0)
        u = (fc.reshape([M,M,1,1]) * spsa.reshape([M, M, dim,1]) * betac.reshape([M,M,1,dim])).sum(axis=0)
        u = (R.reshape([M,dim, dim, 1]) * u.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS -= u + u.transpose((0,2,1))
        zpS += (np.multiply(gcc, aa).reshape([M,M,1,1]) *(betaSymcc - R2 + R.reshape([M,1,dim,dim]))).sum(axis=1)

        if withJacobian:
            pJ = np.squeeze(pJt[T-t-1, ...])
            #print betaxa.shape, betax.shape, Ra.shape
            u = (pJ[...,np.newaxis]*fx)[...,np.newaxis] *(betaxa[...,np.newaxis]*betax -Ra)
            zpx -= u.sum(axis=-2)
            zpc += u.sum(axis=tuple(range(x0.ndim-1)))
            zpS += 0.5 * ((pJ[...,np.newaxis]*fx*betaxa)[...,np.newaxis,np.newaxis]*betaxSym).sum(axis=tuple(range(x0.ndim-1)))
            u = ((pJ[...,np.newaxis]*fx)[...,np.newaxis,np.newaxis] *(Ra[...,np.newaxis]*betax[...,np.newaxis,:])).sum(axis=tuple(range(x0.ndim-1)))
            zpS -= 0.5 * (u + u.transpose((0,2,1)))

        pxt[T-t-2, :, :] = px - timeStep * zpx
        pct[T-t-2, :, :] = pc - timeStep * zpc
        pSt[T-t-2, :, :, :] = pS - timeStep * zpS


        if not (affine == None):
            pxt[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pxt[T-t-1, :, :]), A[T-t-1])
            pct[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pct[T-t-1, :, :]), A[T-t-1])
            pSt[T-t-2, :, :, :] -= timeStep * ((A[T-t-1].reshape([1,dim,dim,1])*pSt[T-t-1, :, :, :].reshape([M,1,dim,dim])).sum(axis=2) + (pSt[T-t-1, :, :, :].reshape([M,dim,1,dim])*A[T-t-1].reshape([1,1,dim,dim])).sum(axis=3))
    if withJacobian:
        return pct, pSt, pxt, pJt, ct, St, xt, Jt
    else:
        return pct, pSt, pxt, ct, St, xt

def gaussianDiffeonsCovectorNormals(c0, S0, b0, x0, xS0,  at, pc1, pS1, pb1, px1, pxS1, sigma, regweight, affine = None):
    dim = c0.shape[1]
    N = x0.shape[0]
    M = c0.shape[0]
    T = at.shape[0]
    timeStep = 1.0/T
    (ct, St, bt, xt, xSt) = gaussianDiffeonsEvolutionEuler(c0, S0, at, sigma, affine=affine, withNormals=b0, withDiffeonSet=(x0, xS0))
    #print bt
    pbt = np.zeros([T, N, dim])
    pxt = np.zeros([T, N, dim])
    pxSt = np.zeros([T, N, dim, dim])
    pct = np.zeros([T, M, dim])
    pSt = np.zeros([T, M, dim, dim])
    pbt[T-1, :, :] = pb1
    pxt[T-1, :, :] = px1
    pxSt[T-1, :, :, :] = pxS1
    pct[T-1, :, :] = pc1
    pSt[T-1, :, :, :] = pS1
    sig2 = sigma*sigma

    if not(affine == None):
        A = affine[0]

    sigEye = sig2*np.eye(dim)
    for t in range(T-1):
        pb = np.squeeze(pbt[T-t-1, :, :])
        px = np.squeeze(pxt[T-t-1, :, :])
        pxS = np.squeeze(pxSt[T-t-1, :, :, :])
        pc = np.squeeze(pct[T-t-1, :, :])
        pS = np.squeeze(pSt[T-t-1, :, :])
        x = np.squeeze(xt[T-t-1, :, :])
        xS = np.squeeze(xSt[T-t-1, :, :, :])
        b = np.squeeze(bt[T-t-1, :, :])
        c = np.squeeze(ct[T-t-1, :, :])
        S = np.squeeze(St[T-t-1, :, :, :])
        a = np.squeeze(at[T-t-1, :, :])

        SS = S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim]) 
        (R, detR) = gd.multiMatInverse1(sigEye.reshape([1,dim,dim]) + S, isSym=True) 
        (R2, detR2) = gd.multiMatInverse2(sigEye.reshape([1,1,dim,dim]) + SS, isSym=True) 

        diffx = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
        betax = (R.reshape([1, M, dim, dim])*diffx.reshape([N, M, 1, dim])).sum(axis=3)
        dst = (diffx * betax).sum(axis=2)
        fx = np.exp(-dst/2)


        diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
        betac = (R.reshape([1, M, dim, dim])*diffc.reshape([M, M, 1, dim])).sum(axis=3)
        dst = (diffc * betac).sum(axis=2)
        fc = np.exp(-dst/2)

        Dv = -((fc.reshape([M,M,1])*betac).reshape([M, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
        IDv = np.eye(dim).reshape([1,dim,dim]) + timeStep * Dv ;
        SpS = (S.reshape([M,dim,dim,1]) * (IDv.reshape([M,dim,dim, 1]) * pS.reshape([M, dim, 1 ,dim])).sum(axis=1).reshape([M,1,dim,dim])).sum(axis=2)
        xDv = -((fx.reshape([N,M,1])*betax).reshape([N, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
        xIDv = np.eye(dim).reshape([1,dim,dim]) + timeStep * xDv ;
        #print xS.shape, xIDv.shape, pxS.shape
        xSpxS = (xS.reshape([N,dim,dim,1]) * (xIDv.reshape([N,dim,dim, 1]) * pxS.reshape([N, dim, 1 ,dim])).sum(axis=1).reshape([N,1,dim,dim])).sum(axis=2)

        aa = np.dot(a, a.T)
        pxa = np.dot(px, a.T)
        pca = np.dot(pc, a.T)
        #ba = (b.reshape([M,1,dim])*a.reshape([1,M,dim])).sum(axis=2)
        ba = np.dot(b, a.T) 
        pbbetax = (pb.reshape([N,1,dim])*betax).sum(axis=2)
        pbb = (pb*b).sum(axis=1)
        betaxa = (betax*a.reshape([1, M, dim])).sum(axis=2)
        #betaca = (betac*a.reshape([1, M, dim])).sum(axis=2)

        betaxSym = betax.reshape([N, M, dim, 1]) * betax.reshape([N, M, 1, dim])
        betaSym = betac.reshape([M, M, dim, 1]) * betac.reshape([M, M, 1, dim])
        betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
        betaSymcc = betacc.reshape([M, M, dim, 1]) * betacc.reshape([M, M, 1, dim])
        dst = (betacc * diffc).sum(axis=2)
        gcc = np.sqrt((detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*np.exp(-dst/2)
        spsa = (SpS.reshape([M,1,dim,dim])*a.reshape([1, M, 1, dim])).sum(axis=3)
        xspsa = (xSpxS.reshape([N,1,dim,dim])*a.reshape([1, M, 1, dim])).sum(axis=3)
        #print np.fabs(betacc + betacc.transpose([1,0,2])).sum()
        fpb = fx * pbbetax * ba
        Rpb = (R.reshape(1, M, dim, dim) *pb.reshape(N,1, 1, dim)).sum(axis=3)
        Ra = (R*a.reshape([M,1,dim])).sum(axis=2)
        #print '?', (pbbetac**2).sum(), (Rpb**2).sum()

        zpb = - np.dot(fx*pbbetax, a) + (fx*betaxa).sum(axis=1).reshape([N,1]) * pb

        u = (pxa * fx).reshape([N, M, 1]) * betax
        zpx = u.sum(axis=1)
        zpc = - u.sum(axis=0)

        u = fpb.reshape([N,M,1]) * betax
        zpx += u.sum(axis=1)
        zpc -= u.sum(axis=0)
        u = (fx*ba).reshape(N, M, 1)*Rpb
        zpx -= u.sum(axis=1)
        zpc += u.sum(axis=0)
        u = (fx * pbb.reshape([N,1]) *betaxa).reshape([N,M,1]) * betax
        zpx -= u.sum(axis=1)
        zpc += u.sum(axis=0)
        u = fx.reshape([N,M,1]) * pbb.reshape([N,1,1]) * Ra.reshape([1,M,dim])
        zpx += u.sum(axis=1)
        zpc -= u.sum(axis=0)
        BmA = betaxSym - R.reshape([1, M, dim, dim])
        u = fx.reshape([N, M, 1]) * (BmA *xspsa.reshape([N, M, 1, dim])).sum(axis=3)
        zpx -= 2 * u.sum(axis=1) 
        zpc += 2 * u.sum(axis=0)

        pSDv = (pxS.reshape([N,dim, dim, 1]) * xDv.reshape([N, 1, dim, dim])).sum(axis=2)
        zpxS = -pSDv - np.transpose(pSDv, (0,2,1)) - timeStep * (xDv.reshape([N,dim, dim, 1]) * pSDv.reshape([N, dim, 1, dim])).sum(axis=1)

        u = (pca * fc).reshape([M, M, 1]) * betac
        zpc += u.sum(axis=1) - u.sum(axis=0)

        BmA = betaSym - R.reshape([1, M, dim, dim])
        u = fc.reshape([M, M, 1]) * (BmA *spsa.reshape([M, M, 1, dim])).sum(axis=3)
        zpc -= 2 * (u.sum(axis=1) - u.sum(axis=0))
        zpc -= 2 * (np.multiply(gcc, aa).reshape([M, M, 1]) * betacc).sum(axis=1)

        zpS = -0.5 * (fpb.reshape([N,M,1,1]) * betaxSym).sum(axis=0)
        RpbbT = Rpb.reshape([N,M,dim,1])*betax.reshape([N,M,1,dim])
        bRpbT = Rpb.reshape([N,M,1,dim])*betax.reshape([N,M,dim, 1])
        zpS += 0.5*((fx*ba).reshape([N,M,1,1])*(RpbbT + bRpbT)).sum(axis=0)
        zpS += 0.5*((fx*betaxa*pbb.reshape([N,1])).reshape([N,M,1,1]) * betaxSym).sum(axis=0)
        betaRaT = Ra.reshape([1,M,dim,1]) * betax.reshape([N,M,1,dim])
        RabetaT = Ra.reshape([1,M,1,dim]) * betax.reshape([N,M,dim,1])
        zpS -= 0.5*( (fx*pbb.reshape([N,1])).reshape([N,M,1,1]) * (betaRaT+RabetaT)).sum(axis=0)
        zpS -= 0.5 * (np.multiply(fx,pxa).reshape([N,M,1,1]) * betaxSym).sum(axis=0)
        u = np.multiply(fx, (xspsa * betax).sum(axis=2))
        zpS += (u.reshape([N,M,1,1])*betaxSym).sum(axis=0)
        u = (fx.reshape([N,M,1,1]) * xspsa.reshape([N, M, dim,1]) * betax.reshape([N,M,1,dim])).sum(axis=0)
        u = (R.reshape([M,dim, dim, 1]) * u.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS -= u + u.transpose((0,2,1))
        
        #zpS += 0.5*((fc*ba).reshape([M,M,1,1])*(RpbbT + np.transpose(RpbbT, (0,1,3,2)))).sum(axis=0)
        zpS -= 0.5 * (np.multiply(fc,pca).reshape([M,M,1,1]) * betaSym).sum(axis=0)
        pSDv = (pS.reshape([M,dim, dim, 1]) * Dv.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS += -pSDv - np.transpose(pSDv, (0,2,1)) - timeStep * (Dv.reshape([M,dim, dim, 1]) * pSDv.reshape([M, dim, 1, dim])).sum(axis=1)
        u = np.multiply(fc, (spsa * betac).sum(axis=2))
        zpS += (u.reshape([M,M,1,1])*betaSym).sum(axis=0)
        u = (fc.reshape([M,M,1,1]) * spsa.reshape([M, M, dim,1]) * betac.reshape([M,M,1,dim])).sum(axis=0)
        u = (R.reshape([M,dim, dim, 1]) * u.reshape([M, 1, dim, dim])).sum(axis=2)
        zpS -= u + u.transpose((0,2,1))
        zpS += (np.multiply(gcc, aa).reshape([M,M,1,1]) *(betaSymcc - R2 + R.reshape([M,1,dim,dim]))).sum(axis=1)

        pbt[T-t-2, :, :] = pb - timeStep * zpb
        pxt[T-t-2, :, :] = px - timeStep * zpx
        pxSt[T-t-2, :, :] = pxS - timeStep * zpxS
        pct[T-t-2, :, :] = pc - timeStep * zpc
        pSt[T-t-2, :, :, :] = pS - timeStep * zpS

        if not (affine == None):
            pbt[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pbt[T-t-1, :, :]), A[T-t-1].T)
            pct[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pct[T-t-1, :, :]), A[T-t-1])
            pSt[T-t-2, :, :, :] -= timeStep * ((A[T-t-1].reshape([1,dim,dim,1])*pSt[T-t-1, :, :, :].reshape([M,1,dim,dim])).sum(axis=2) + (pSt[T-t-1, :, :, :].reshape([M,dim,1,dim])*A[T-t-1].reshape([1,1,dim,dim])).sum(axis=3))
            pxt[T-t-2, :, :] -= timeStep * np.dot(np.squeeze(pxt[T-t-1, :, :]), A[T-t-1])
            pxSt[T-t-2, :, :, :] -= timeStep * ((A[T-t-1].reshape([1,dim,dim,1])*pxSt[T-t-1, :, :, :].reshape([N,1,dim,dim])).sum(axis=2) + (pxSt[T-t-1, :, :, :].reshape([N,dim,1,dim])*A[T-t-1].reshape([1,1,dim,dim])).sum(axis=3))
    return pct, pSt, pbt, pxt, pxSt, ct, St, bt, xt, xSt
    
# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def gaussianDiffeonsGradientPset(c0, S0, x0, at, pc1, pS1, px1, sigma, regweight, getCovector = False, affine = None, withJacobian=None):
    if not(withJacobian==None):
        #print withJacobian
        J0 = withJacobian[0]
        pJ1 = withJacobian[1]
        withJacobian = True
    else:
        withJacobian = False

    if withJacobian:
        (pct, pSt, pxt, pJt, ct, St, xt, Jt) = gaussianDiffeonsCovectorPset(c0, S0, x0, at, pc1, pS1, px1, sigma, regweight, affine=affine, withJacobian=(J0, pJ1))
    else:
        (pct, pSt, pxt, ct, St, xt) = gaussianDiffeonsCovectorPset(c0, S0, x0, at, pc1, pS1, px1, sigma, regweight, affine=affine)
    #print (pct**2).sum()**0.5, (pSt**2).sum()**0.5, (pxt**2).sum()**0.5
    

    dat = np.zeros(at.shape)
    M = c0.shape[0]
    dim = c0.shape[1]
    if not (affine == None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for t in range(at.shape[0]):
        a = np.squeeze(at[t, :, :])
        x = np.squeeze(xt[t, ...])
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, ...])
        px = np.squeeze(pxt[t, ...])
        pc = np.squeeze(pct[t, :, :])
        pS = np.squeeze(pSt[t, ...])
        if withJacobian:
            J = np.squeeze(Jt[t, ...])
            pJ = np.squeeze(pJt[t,...])
            [grc, grS, grx, grJ, gcc] = gd.gaussianDiffeonsGradientMatricesPset(c, S, x, a, pc, pS, px, sigma, 1.0/at.shape[0], withJacobian=(J,pJ))
            da = 2*np.dot(gcc,a) - grx - grc - grS - grJ
        else:    
            [grc, grS, grx, gcc] = gd.gaussianDiffeonsGradientMatricesPset(c, S, x, a, pc, pS, px, sigma, 1.0/at.shape[0])
            da = 2*np.dot(gcc,a) - grx - grc - grS

        if not (affine == None):
            #print px.shape, x.shape, pc.shape, c.shape
            dA[t] = (px * x).sum(axis=tuple(range(x.ndim-1))) + np.dot(pc.T, c) - 2*np.multiply(pS.reshape([M, dim, dim, 1]), S.reshape([M, dim, 1, dim])).sum(axis=1).sum(axis=0)
            db[t] = px.sum(axis=tuple(range(x.ndim-1))) + pc.sum(axis=0)

        (L, W) = LA.eigh(gcc)
        dat[t, :, :] = LA.solve(gcc+(L.max()/1000)*np.eye(M), da)
        #dat[t, :, :] = LA.solve(gcc, da)

    if affine == None:
        if getCovector == False:
            if withJacobian:
                return dat, ct, St, xt, Jt
            else:
                return dat, ct, St, xt
        else:
            if withJacobian:
                return dat, ct, St, xt, Jt, pct, pSt, pxt, pJt
            else:
                return dat, ct, St, xt, pct, pSt, pxt
    else:
        if getCovector == False:
            if withJacobian:
                return dat, dA, db, ct, St, xt, Jt
            else:
                return dat, dA, db, ct, St, xt
        else:
            if withJacobian:
                return dat, dA, db, ct, St, xt, Jt, pct, pSt, pxt, pJt
            else:
                return dat, dA, db, ct, St, xt, pct, pSt, pxt

def gaussianDiffeonsGradientNormals(c0, S0, b0, x0, xS0, at, pc1, pS1, pb1, px1, pxS1, sigma, regweight, getCovector = False, affine = None):
    (pct, pSt, pbt, pxt, pxSt, ct, St, bt, xt, xSt) = gaussianDiffeonsCovectorNormals(c0, S0, b0, x0, xS0, at,
                                                                                      pc1, pS1, pb1, px1, pxS1, sigma, regweight, affine=affine)

    #print (pct**2).sum()**0.5, (pSt**2).sum()**0.5, (pbt**2).sum()**0.5

    dat = np.zeros(at.shape)
    M = c0.shape[0]
    dim = c0.shape[1]
    if not (affine == None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for t in range(at.shape[0]):
        a = np.squeeze(at[t, :, :])
        b = np.squeeze(bt[t, :, :])
        x = np.squeeze(xt[t, :, :])
        xS = np.squeeze(xSt[t, :, :])
        c = np.squeeze(ct[t, :, :])
        S = np.squeeze(St[t, :, :])
        pb = np.squeeze(pbt[t, :, :])
        px = np.squeeze(pxt[t, :, :])
        pxS = np.squeeze(pxSt[t, :, :])
        pc = np.squeeze(pct[t, :, :])
        pS = np.squeeze(pSt[t, :, :])
        [grc, grS, grb, grx, grxS, gcc] = gd.gaussianDiffeonsGradientMatricesNormals(c, S, b, x, xS, a, pc, pS, pb, px, pxS, sigma, 1.0/at.shape[0])
        #print t, (grS**2).sum(), (grb**2).sum()
        
        da = 2*np.dot(gcc,a) - grb - grc - grS - grx - grxS
        if not (affine == None):
            dA[t] = -np.dot(b.T, pb) + np.dot(pc.T, c) - 2*np.multiply(pS.reshape([M, dim, dim, 1]), S.reshape([M, dim, 1, dim])).sum(axis=1).sum(axis=0)
            db[t] = pc.sum(axis=0)

        (L, W) = LA.eigh(gcc)
        dat[t, :, :] = LA.solve(gcc+(L.max()/1000)*np.eye(M), da)
        #dat[t, :, :] = LA.solve(gcc, da)

    if affine == None:
        if getCovector == False:
            return dat, ct, St, bt, xt, xSt
        else:
            return dat, ct, St, bt, xt, xSt, pct, pSt, pbt, pxt, pxSt
    else:
        if getCovector == False:
            return dat, dA, db, ct, St, bt, xt, xSt
        else:
            return dat, dA, db, ct, St, bt, xt, xSt, pct, pSt, pbt, pxT, pxSt


def landmarkEPDiff(T, x0, a0, KparDiff, affine = None, withJacobian=False, withNormals=None, withPointSet=None):
    N = x0.shape[0]
    dim = x0.shape[1]
    timeStep = 1.0/T
    at = np.zeros([T, N, dim])
    xt = np.zeros([T+1, N, dim])
    xt[0, :, :] = x0
    at[0, :, :] = a0
    simpleOutput = True
    if not (withNormals==None):
        simpleOutput = False
        nt = np.zeros([T+1, N, dim])
        nt[0, :, :] = withNormals
    if withJacobian:
        simpleOutput = False
        Jt = np.zeros([T+1, N])
    if not(affine == None):
        A = affine[0]
        b = affine[1]
    if not (withPointSet==None):
        simpleOutput = False
        K = withPointSet.shape[0]
        yt = np.zeros([T+1,K,dim])
        yt[0, :, :] = withPointSet

    for k in range(T):
        z = np.squeeze(xt[k, :, :])
        a = np.squeeze(at[k, :, :])
        xt[k+1, :, :] = z + timeStep * KparDiff.applyK(z, a)
        #print 'test', px.sum()
        if k < (T-1):
            at[k+1, :, :] = a - timeStep * KparDiff.applyDiffKT(z, a[np.newaxis,...], a[np.newaxis,...])
        if not (affine == None):
            xt[k+1, :, :] += timeStep * (np.dot(z, A[k].T) + b[k])
        if not (withPointSet == None):
            zy = np.squeeze(yt[k, :, :])
            yt[k+1, :, :] = zy + timeStep * KparDiff.applyK(z, a, firstVar=zy)
            if not (affine == None):
                yt[k+1, :, :] += timeStep * (np.dot(zy, A[k].T) + b[k])

        if not (withNormals==None):
            zn = np.squeeze(nt[k, :, :])        
            nt[k+1, :, :] = zn - timeStep * KparDiff.applyDiffKT(z, zn[np.newaxis,...], a[np.newaxis,...]) 
            if not (affine == None):
                nt[k+1, :, :] += timeStep * np.dot(zn, A[k])
        if withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a)
            if not (affine == None):
                Jt[k+1, :] += timeStep * (np.trace(A[k]))
    if simpleOutput:
        return xt, at
    else:
        output = [xt, at]
        if not (withPointSet==None):
            output.append(yt)
        if not (withNormals==None):
            output.append(nt)
        if withJacobian:
            output.append(Jt)
        return output



# Solves dx/dt = K(x,x) a(t) + A(t) x + b(t) with x(0) = x0
# affine: affine component [A, b] or None
# if withJacobian =True: return Jacobian determinant
# if withNormal = nu0, returns nu(t) evolving as dnu/dt = -Dv^{T} nu
def landmarkDirectEvolutionEuler(x0, at, KparDiff, affine = None, withJacobian=False, withNormals=None, withPointSet=None):
    N = x0.shape[0]
    dim = x0.shape[-1]
    M = at.shape[0] + 1
    timeStep = 1.0/(M-1)
    xt = np.zeros([M, N, dim])
    xt[0, ...] = x0
    simpleOutput = True
    if not (withNormals==None):
        simpleOutput = False
        nt = np.zeros([M, N, dim])
        nt[0, ...] = withNormals
    if not(affine == None):
        A = affine[0]
        b = affine[1]
    if not (withPointSet==None):
        simpleOutput = False
        K = withPointSet.shape[0]
        yt = np.zeros([M,K,dim])
        yt[0,...] = withPointSet
        if withJacobian:
            simpleOutput = False
            Jt = np.zeros([M, K])
    else:
        if withJacobian:
            simpleOutput = False
            Jt = np.zeros([M, N])

    for k in range(M-1):
        z = np.squeeze(xt[k, ...])
        a = np.squeeze(at[k, ...])
        xt[k+1, ...] = z + timeStep * KparDiff.applyK(z, a)
        if not (affine == None):
            xt[k+1, :, :] += timeStep * (np.dot(z, A[k].T) + b[k])
        if not (withPointSet == None):
            zy = np.squeeze(yt[k, :, :])
            yt[k+1, :, :] = zy + timeStep * KparDiff.applyK(z, a, firstVar=zy)
            if not (affine == None):
                yt[k+1, :, :] += timeStep * (np.dot(zy, A[k].T) + b[k])
            if withJacobian:
                Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a, firstVar=zy)
                if not (affine == None):
                    Jt[k+1, :] += timeStep * (np.trace(A[k]))
        else:
            if withJacobian:
                Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a)
                if not (affine == None):
                    Jt[k+1, :] += timeStep * (np.trace(A[k]))

        if not (withNormals==None):
            zn = np.squeeze(nt[k, :, :])        
            nt[k+1, :, :] = zn - timeStep * KparDiff.applyDiffKT(z, zn[np.newaxis,...], a[np.newaxis,...]) 
            if not (affine == None):
                nt[k+1, :, :] += timeStep * np.dot(zn, A[k])
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




def landmarkHamiltonianCovector(x0, at, px1, KparDiff, regweight, affine = None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0]
    timeStep = 1.0/M
    xt = landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
    pxt = np.zeros([M+1, N, dim])
    pxt[M, :, :] = px1
    if not(affine == None):
        A = affine[0]

    for t in range(M):
        px = np.squeeze(pxt[M-t, :, :])
        z = np.squeeze(xt[M-t-1, :, :])
        a = np.squeeze(at[M-t-1, :, :])
        # dgzz = kfun.kernelMatrix(KparDiff, z, diff=True)
        # if (isfield(KparDiff, 'zs') && size(z, 2) == 3)
        #     z(:,3) = z(:,3) / KparDiff.zs ;
        # end
        a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*regweight*a[np.newaxis,...]))
        a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
        #a1 = [px, a, -2*regweight*a]
        #a2 = [a, px, a]
        #print 'test', px.sum()
        zpx = KparDiff.applyDiffKT(z, a1, a2)
        pxt[M-t-1, :, :] = px + timeStep * zpx
        #print 'zpx', np.fabs(zpx).sum(), np.fabs(px).sum(), z.sum()
        #print 'pxt', np.fabs((pxt)[M-t-2]).sum()
        if not (affine == None):
            pxt[M-t-1, :, :] += timeStep * np.dot(px, A[M-t-1])
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
        px = np.squeeze(pxt[k+1, :, :])
        #print 'testgr', (2*a-px).sum()
        dat[k, :, :] = (2*regweight*a-px)
        if not (affine == None):
            dA[k] = np.dot(pxt[k+1].T, xt[k])
            db[k] = pxt[k+1].sum(axis=0)

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

def timeSeriesCovector(x0, at, px1, KparDiff, regweight, affine = None):
    N = x0.shape[0]
    dim = x0.shape[1]
    M = at.shape[0]
    nTarg = len(px1)
    Tsize1 = M/nTarg
    timeStep = 1.0/M
    xt = landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
    pxt = np.zeros([M+1, N, dim])
    pxt[M, :, :] = px1[nTarg-1]
    if not(affine == None):
        A = affine[0]

    for t in range(M):
        px = np.squeeze(pxt[M-t, :, :])
        z = np.squeeze(xt[M-t-1, :, :])
        a = np.squeeze(at[M-t-1, :, :])
        # dgzz = kfun.kernelMatrix(KparDiff, z, diff=True)
        # if (isfield(KparDiff, 'zs') && size(z, 2) == 3)
        #     z(:,3) = z(:,3) / KparDiff.zs ;
        # end
        a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*regweight*a[np.newaxis,...]))
        a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
        #a1 = [px, a, -2*regweight*a]
        #a2 = [a, px, a]
        #print 'test', px.sum()
        zpx = KparDiff.applyDiffKT(z, a1, a2)
        if not (affine == None):
            zpx += np.dot(px, A[M-t-1])
        pxt[M-t-1, :, :] = px + timeStep * zpx
        if (t<M-1) and ((M-1-t)%Tsize1 == 0):
            pxt[M-t-1, :, :] += px1[(M-t-1)/Tsize1 - 1]
        #print 'zpx', np.fabs(zpx).sum(), np.fabs(px).sum(), z.sum()
        #print 'pxt', np.fabs((pxt)[M-t-2]).sum()
        
    return pxt, xt

def timeSeriesGradient(x0, at, px1, KparDiff, regweight, getCovector = False, affine = None):
    (pxt, xt) = timeSeriesCovector(x0, at, px1, KparDiff, regweight, affine=affine)
    dat = np.zeros(at.shape)
    if not (affine == None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    for k in range(at.shape[0]):
        a = np.squeeze(at[k, :, :])
        px = np.squeeze(pxt[k+1, :, :])
        #print 'testgr', (2*a-px).sum()
        dat[k, :, :] = (2*regweight*a-px)
        if not (affine == None):
            dA[k] = np.dot(pxt[k+1].T, xt[k])
            db[k] = pxt[k+1].sum(axis=0)

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

def secondOrderEvolution(x0, a0, rhot, KparDiff, withJacobian=False, withPointSet=None, affine=None):
    T = rhot.shape[0]
    N = x0.shape[0]
    #print M, N
    dim = x0.shape[1]
    dim2 = dim*dim
    timeStep = 1.0/T
    at = np.zeros([T+1, N, dim])
    xt = np.zeros([T+1, N, dim])
    xt[0, :, :] = x0
    at[0, :, :] = a0
    simpleOutput = True
    if not(affine == None):
        aff_ = True
        A = affine[0]
        b = affine[1]
    else:
        aff_=False
    if not (withPointSet==None):
        simpleOutput = False
        K = withPointSet.shape[0]
        zt = np.zeros([T+1,K,dim])
        zt[0, :, :] = withPointSet
        if withJacobian:
            simpleOutput = False
            Jt = np.zeros([T+1, K])
    elif withJacobian:
        simpleOutput = False
        Jt = np.zeros([T+1, M])

    for k in range(T):
        x = np.squeeze(xt[k, :, :])
        a = np.squeeze(at[k, :, :])
        #print 'evolution v:', np.sqrt((v**2).sum(axis=1)).sum()/v.shape[0]
        rho = np.squeeze(rhot[k,:,:])
        zx = KparDiff.applyK(x, a)
        za = -KparDiff.applyDiffKT(x, a[np.newaxis,...], a[np.newaxis,...]) + rho
        if aff_:
            U = np.eye(dim) + timeStep * A[k]
            xt[k+1, :, :] = np.dot(x + timeStep * zx, U.T) + timeStep * b[k]
            Ui = LA.inv(U)
            at[k+1, :, :] = np.dot(a + timeStep * za, Ui)
        else:
            xt[k+1, :, :] = x + timeStep * zx  
            at[k+1, :, :] = a + timeStep * za
        if not (withPointSet == None):
            z = np.squeeze(zt[k, :, :])
            zx = KparDiff.applyK(x, a, firstVar=z)
            if aff_:
                zt[k+1, :, :] =  np.dot(z + timeStep * zx, U.T) + timeStep * b[k]
            else:
                zt[k+1, :, :] = z + timeStep * zx  
            if withJacobian:
                Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(x, a, firstVar=z)
                if aff_:
                    Jt[k+1, :] += timeStep * (np.trace(A[k]))
        elif withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(z, a)
            if aff_:
                Jt[k+1, :] += timeStep * (np.trace(A[k]))
    if simpleOutput:
        return xt, at
    else:
        output = [xt, at]
        if not (withPointSet==None):
            output.append(zt)
        if withJacobian:
            output.append(Jt)
        return output



def secondOrderHamiltonian(x, a, rho, px, pa, KparDiff, affine=None):


    Ht = (px * KparDiff.applyK(x, a)).sum() 
    Ht += (pa*(-KparDiff.applyDiffKT(x, a[np.newaxis,...], a[np.newaxis,...]) + rho)).sum()
    Ht -= (rho**2).sum()/2
    if not(affine == None):
        A = affine[0]
        b = affine[1]
        Ht += (px * (np.dot(x, A.T) + b)).sum() - (pa * np.dot(a, A)).sum()
    return Ht

    
def secondOrderCovector(x0, a0, rhot, px1, pa1, KparDiff, affine = None, times= None):
    T = rhot.shape[0]
    nTarg = len(px1)
    Tsize1 = T/nTarg
    if not(affine == None):
        aff_ = True
        A = affine[0]
        b = affine[1]
    else:
        aff_ = False
    if times == None:
        t1 = (float(T)/nTarg) * (range(nTarg)+1)
    N = x0.shape[0]
    dim = x0.shape[1]
    timeStep = 1.0/T
    [xt, at] = secondOrderEvolution(x0, a0, rhot, KparDiff)
    pxt = np.zeros([T+1, N, dim])
    pxt[T, :, :] = px1[nTarg-1]
    pat = np.zeros([T+1, N, dim])
    pat[T, :, :] = pa1[nTarg-1]
    for t in range(T):
        px = np.squeeze(pxt[T-t, :, :])
        pa = np.squeeze(pat[T-t, :, :])
        x = np.squeeze(xt[T-t-1, :, :])
        a = np.squeeze(at[T-t-1, :, :])
        rho = np.squeeze(rhot[T-t-1, :, :])
        
        if aff_:
            U = np.eye(dim) + timeStep * A[T-t-1]
            px_ = np.dot(px, U)
            Ui = LA.inv(U)
            pa_ = np.dot(pa,Ui.T)
        else:
            px_ = px
            pa_ = pa

        #zpx = KparDiff.applyDiffKT(x, [px], [a])

        #zpa = KparDiff.applyK(x, px)
        #print 'zpa1', zpa.sum()
        #zpy = KparDiff.applyDiffKT(x, [a], [px])

        a1 = np.concatenate((px_[np.newaxis,...], a[np.newaxis,...]))
        a2 = np.concatenate((a[np.newaxis,...], px_[np.newaxis,...]))
        zpx = KparDiff.applyDiffKT(x, a1, a2) - KparDiff.applyDDiffK11and12(x, a, a, pa_)
        zpa = KparDiff.applyK(x, px_) - KparDiff.applyDiffK1and2(x, pa_, a)
#               - KparDiff.applyDiffK(x, pa, a) - KparDiff.applyDiffK2(x, pa, a))

        # if aff_:
        #     zpx += np.dot(px, A[T-t-1])
        #     zpa -= np.dot(pa, A[T-t-1].T)

        ### Checking derivatives
        # if affine == None:
        #     afn = None
        # else:
        #     afn = (A[T-t-1], b[T-t-1])
        # H0 = secondOrderHamiltonian(x,a,rho,px,pa,KparDiff, affine=afn)
        # eps = 0.000001
        # dx = eps*np.random.normal(size=x.shape)
        # dH = secondOrderHamiltonian(x+dx,a,rho,px,pa,KparDiff, affine=afn)
        # print 'Test H; x:', (dH - H0)/eps, (zpx*dx).sum()/eps
        # da = eps*np.random.normal(size=a.shape)
        # dH = secondOrderHamiltonian(x,a+da,rho,px,pa,KparDiff, affine=afn)
        # print 'Test H; a:', (dH - H0)/eps, (zpa*da).sum()/eps


        
        #print 'zpa2', zpx.sum()
        pxt[T-t-1, :, :] = px_ + timeStep * zpx
        pat[T-t-1, :, :] = pa_ + timeStep * zpa
        if (t<T-1) and ((T-1-t)%Tsize1 == 0):
#            print T-t-1, (T-t-1)/Tsize1
            pxt[T-t-1, :, :] += px1[(T-t-1)/Tsize1 - 1]
            pat[T-t-1, :, :] += pa1[(T-t-1)/Tsize1 - 1]

    return pxt, pat, xt, at

# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def secondOrderGradient(x0, a0, rhot, px1, pa1, KparDiff, times = None, getCovector = False, affine=None, controlWeight=1.0):
    (pxt, pat, xt, at) = secondOrderCovector(x0, a0, rhot, px1, pa1, KparDiff, times=times,affine=affine)
    if not (affine == None):
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    Tsize = rhot.shape[0]
    timeStep = 1.0/Tsize
    drhot = np.zeros(rhot.shape)
    if not (affine == None):
        dim = x0.shape[1]
        for k in range(Tsize):
            x = np.squeeze(xt[k, :, :])
            a = np.squeeze(at[k, :, :])
            rho = np.squeeze(rhot[k, :, :])
            px = np.squeeze(pxt[k+1, :, :])
            pa = np.squeeze(pat[k+1, :, :])
            zx = x + timeStep*KparDiff.applyK(x, a)
            za = a + timeStep * (-KparDiff.applyDiffKT(x, a[np.newaxis,...], a[np.newaxis,...]) + rho)
            U = np.eye(dim) + timeStep * affine[0][k]
            Ui = LA.inv(U)
            pa = np.dot(pa, Ui.T)
            za = np.dot(za, Ui)
            dA[k,...] =  ((px[:,:,np.newaxis]*zx[:,np.newaxis,:]).sum(axis=0)
                            - (za[:,:,np.newaxis]*pa[:,np.newaxis,:]).sum(axis=0))
            drhot[k,...] = rho*controlWeight - pa
        db = pxt[1:Tsize+1,...].sum(axis=1)
        # for k in range(rhot.shape[0]):
        #     #np.dot(pxt[k+1].T, xt[k]) - np.dot(at[k].T, pat[k+1])
        #     #dA[k] = -np.dot(pat[k+1].T, at[k]) + np.dot(xt[k].T, pxt[k+1])
        #     db[k] = pxt[k+1].sum(axis=0)

    #drhot = rhot*controlWeight - pat[1:pat.shape[0],...]
    da0 = KparDiff.applyK(x0, a0) - pat[0,...]

    if affine == None:
        if getCovector == False:
            return da0, drhot, xt, at
        else:
            return da0, drhot, xt, at, pxt, pat
    else:
        if getCovector == False:
            return da0, drhot, dA, db, xt, at
        else:
            return da0, drhot, dA, db, xt, at, pxt, pat

        

def secondOrderFiberEvolution(x0, a0, y0, v0, rhot, KparDiff, withJacobian=False, withPointSet=None):
    T = rhot.shape[0]
    N = x0.shape[0]
    M = y0.shape[0]
    #print M, N
    dim = x0.shape[1]
    timeStep = 1.0/T
    at = np.zeros([T+1, M, dim])
    yt = np.zeros([T+1, M, dim])
    vt = np.zeros([T+1, M, dim])
    xt = np.zeros([T+1, N, dim])
    xt[0, :, :] = x0
    at[0, :, :] = a0
    yt[0, :, :] = y0
    vt[0, :, :] = v0
    simpleOutput = True
    if not (withPointSet==None):
        simpleOutput = False
        K = withPointSet.shape[0]
        zt = np.zeros([T+1,K,dim])
        zt[0, :, :] = withPointSet
        if withJacobian:
            simpleOutput = False
            Jt = np.zeros([T+1, K])
    elif withJacobian:
        simpleOutput = False
        Jt = np.zeros([T+1, M])

    for k in range(T):
        x = np.squeeze(xt[k, :, :])
        y = np.squeeze(yt[k, :, :])
        a = np.squeeze(at[k, :, :])
        v = np.squeeze(vt[k, :, :])
        #print 'evolution v:', np.sqrt((v**2).sum(axis=1)).sum()/v.shape[0]
        rho = np.squeeze(rhot[k,:])
        xt[k+1, :, :] = x + timeStep * KparDiff.applyK(y, a, firstVar=x) 
        yt[k+1, :, :] = y + timeStep * KparDiff.applyK(y, a)
        KparDiff.hold()
        at[k+1, :, :] = a + timeStep * (-KparDiff.applyDiffKT(y, a[np.newaxis,...], a[np.newaxis,...]) + rho[:,np.newaxis] * v) 
        vt[k+1, :, :] = v + timeStep * KparDiff.applyDiffK(y, v, a) 
        KparDiff.release()
        if not (withPointSet == None):
            z = np.squeeze(zt[k, :, :])
            zt[k+1, :, :] = z + timeStep * KparDiff.applyK(y, a, firstVar=z)
            if withJacobian:
                Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(y, a, firstVar=z)
        elif withJacobian:
            Jt[k+1, :] = Jt[k, :] + timeStep * KparDiff.applyDivergence(y, a)
    if simpleOutput:
        return xt, at, yt, vt
    else:
        output = [xt, at, yt, vt]
        if not (withPointSet==None):
            output.append(zt)
        if withJacobian:
            output.append(Jt)
        return output
    
def secondOrderFiberHamiltonian(x, a, y, v, rho, px, pa, py, pv, KparDiff):

    Ht = ( (px * KparDiff.applyK(y, a, firstVar=x)).sum() 
           + (py*KparDiff.applyK(y, a)).sum())
    KparDiff.hold()
    Ht += ( (pa*(-KparDiff.applyDiffKT(y, a[np.newaxis,...], a[np.newaxis,...]) + rho[:,np.newaxis] * v)).sum()
            + (pv*KparDiff.applyDiffK(y, v, a)).sum()) 
    KparDiff.release()
    Ht -= (rho**2 * (v**2).sum(axis=1)).sum()/2
    return Ht

    
def secondOrderFiberCovector(x0, a0, y0, v0, rhot, px1, pa1, py1, pv1, KparDiff, times= None):
    T = rhot.shape[0]
    nTarg = len(px1)
    Tsize1 = T/nTarg
    if times == None:
        t1 = (float(T)/nTarg) * (range(nTarg)+1)
    N = x0.shape[0]
    M = y0.shape[0]
    dim = x0.shape[1]
    timeStep = 1.0/T
    [xt, at, yt, vt] = secondOrderFiberEvolution(x0, a0, y0, v0, rhot, KparDiff)
    pxt = np.zeros([T, N, dim])
    pxt[T-1, :, :] = px1[nTarg-1]
    pat = np.zeros([T, M, dim])
    pat[T-1, :, :] = pa1[nTarg-1]
    pyt = np.zeros([T, M, dim])
    pyt[T-1, :, :] = py1[nTarg-1]
    pvt = np.zeros([T, M, dim])
    pvt[T-1, :, :] = pv1[nTarg-1]
    for t in range(T-1):
        px = np.squeeze(pxt[T-t-1, :, :])
        pa = np.squeeze(pat[T-t-1, :, :])
        py = np.squeeze(pyt[T-t-1, :, :])
        pv = np.squeeze(pvt[T-t-1, :, :])
        x = np.squeeze(xt[T-t-1, :, :])
        a = np.squeeze(at[T-t-1, :, :])
        y = np.squeeze(yt[T-t-1, :, :])
        v = np.squeeze(vt[T-t-1, :, :])
        rho = np.squeeze(rhot[T-t-1, :])

        zpx = KparDiff.applyDiffKT(y, px[np.newaxis,...], a[np.newaxis,...], firstVar=x)

        zpa = KparDiff.applyK(x, px, firstVar=y)
        KparDiff.hold()
        #print 'zpa1', zpa.sum()
        zpy = KparDiff.applyDiffKT(x, a[np.newaxis,...], px[np.newaxis,...], firstVar=y)
        KparDiff.release()

        a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...]))
        a2 = np.concatenate((a[np.newaxis,...], py[np.newaxis,...]))
        zpy += KparDiff.applyDiffKT(y, a1, a2)
        KparDiff.hold()
        zpy += KparDiff.applyDDiffK11(y, pv, a, v) + KparDiff.applyDDiffK12(y, a, pv, v)
        zpy -= KparDiff.applyDDiffK11(y, a, a, pa) + KparDiff.applyDDiffK12(y, a, a, pa)

        zpv = KparDiff.applyDiffKT(y, pv[np.newaxis,...], a[np.newaxis,...]) + rho[:,np.newaxis]*pa - (rho[:,np.newaxis]**2)*v
        zpa += (KparDiff.applyK(y, py) + KparDiff.applyDiffK2(y, v, pv)
               - KparDiff.applyDiffK(y, pa, a) - KparDiff.applyDiffK2(y, pa, a))
        KparDiff.release()

        ### Checking derivatives
        # H0 = secondOrderFiberHamiltonian(x,a,y,v,rho,px,pa,py,pv,KparDiff)
        # eps = 0.000001
        # dx = eps*np.random.normal(size=x.shape)
        # dH = secondOrderFiberHamiltonian(x+dx,a,y,v,rho,px,pa,py,pv,KparDiff)
        # print 'Test H; x:', (dH - H0)/eps, (zpx*dx).sum()/eps
        # dy = eps*np.random.normal(size=y.shape)
        # dH = secondOrderFiberHamiltonian(x,a,y+dy,v,rho,px,pa,py,pv,KparDiff)
        # print 'Test H; y:', (dH - H0)/eps, (zpy*dy).sum()/eps
        # da = eps*np.random.normal(size=a.shape)
        # dH = secondOrderFiberHamiltonian(x,a+da,y,v,rho,px,pa,py,pv,KparDiff)
        # print 'Test H; a:', (dH - H0)/eps, (zpa*da).sum()/eps
        # dv = eps*np.random.normal(size=v.shape)
        # dH = secondOrderFiberHamiltonian(x,a,y,v+dv,rho,px,pa,py,pv,KparDiff)
        # print 'Test H; v:', (dH - H0)/eps, (zpv*dv).sum()/eps



        
        #print 'zpa2', zpa.sum()
        pxt[T-t-2, :, :] = px + timeStep * zpx
        pat[T-t-2, :, :] = pa + timeStep * zpa
        pyt[T-t-2, :, :] = py + timeStep * zpy
        pvt[T-t-2, :, :] = pv + timeStep * zpv
        if (t<T-1) and ((T-t-1)%Tsize1 == 0):
#            print T-t-1, (T-t-1)/Tsize1
            pxt[T-t-2, :, :] += px1[(T-t-1)/Tsize1 - 1]
            pat[T-t-2, :, :] += pa1[(T-t-1)/Tsize1 - 1]
            pyt[T-t-2, :, :] += py1[(T-t-1)/Tsize1 - 1]
            pvt[T-t-2, :, :] += pv1[(T-t-1)/Tsize1 - 1]

    return pxt, pat, pyt, pvt, xt, at, yt, vt

# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def secondOrderFiberGradient(x0, a0, y0, v0, rhot, px1, pa1, py1, pv1, KparDiff, times = None, getCovector = False):
    (pxt, pat, pyt, pvt, xt, at, yt, vt) = secondOrderFiberCovector(x0, a0, y0, v0, rhot, px1, pa1, py1, pv1, KparDiff, times=times)
    drhot = np.zeros(rhot.shape)
    # if not (affine == None):
    #     dA = np.zeros(affine[0].shape)
    #     db = np.zeros(affine[1].shape)
    for k in range(rhot.shape[0]):
        rho = np.squeeze(rhot[k, :])
        pa = np.squeeze(pat[k, :, :])
        v = np.squeeze(vt[k, :, :])
        #print 'testgr', pa.sum(), pxt[k,...].sum()
        drhot[k, :] = rho - (pa*v).sum(axis=1)/(v**2).sum(axis=1)
        #drhot[k, :] = rho*(v**2).sum(axis=1) - (pa*v).sum(axis=1)
        # H0 = secondOrderFiberHamiltonian(xt[k,...],at[k,...],yt[k,...],v,rho,pxt[k,...],pa,pyt[k,...],pvt[k,...],KparDiff)
        # eps = 0.000001
        # dr = eps*np.random.normal(size=rho.shape)
        # dH = secondOrderFiberHamiltonian(xt[k,...],at[k,...],yt[k,...],v,rho+dr,pxt[k,...],pa,pyt[k,...],pvt[k,...],KparDiff)
        # print 'Test H; v:', (dH - H0)/eps, -(drhot[k,:]*dr).sum()/eps
        # if not (affine == None):
        #     dA[k] = np.dot(pxt[k].T, xt[k])
        #     db[k] = pxt[k].sum(axis=0)
    if getCovector == False:
        return drhot, xt, at, yt, vt
    else:
        return drhot, xt, at, yt, vt, pxt, pat, pyt, pvt

    # if affine == None:
    #     if getCovector == False:
    #         return dat, xt
    #     else:
    #         return dat, xt, pxt
    # else:
    #     if getCovector == False:
    #         return dat, dA, db, xt
    #     else:
    #         return dat, dA, db, xt, pxt
