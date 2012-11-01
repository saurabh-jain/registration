import numpy as np
import kernelFunctions as kfun

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


# backwards covector evolution along trajectory associated to x0, at
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

# Computes gradient after covariance evolution for deformation cost a^TK(x,x) a
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

