import os
import logging
import numpy as np
import scipy as sp
import grid
import curves
import kernelFunctions as kfun
import pointEvolution as evol
import conjugateGradient as cg
import curveMatching
from affineBasis import *


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDiffOut: background kernel: if not specified, use typeKernel with width sigmaKernelOut
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class CurveMatchingParam(curveMatching.CurveMatchingParam):
    def __init__(self, timeStep = .1, KparDiff = None, KparDist = None, KparDiffOut = None, sigmaKernel = 6.5, sigmaKernelOut=6.5, sigmaDist=2.5, sigmaError=1.0, typeKernel='gauss', errorType='measure'):
        curveMatching.CurveMatchingParam.__init__(self, timeStep = timeStep, KparDiff = KparDiff, KparDist=KparDist, sigmaKernel =  sigmaKernel, sigmaDist = sigmaDist, sigmaError = sigmaError, typeKernel = typeKernel, errorType=errorType)
        self.sigmaKernelOut = sigmaKernelOut
        if KparDiffOut == None:
            self.KparDiffOut = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernelOut)
        else:
            self.KparDiffOut = KparDiffOut


## Main class for curve matching
#        Template: sequence of curve classes (from curves.py); if not specified, opens files in fileTemp
#        Target: sequence of curve classes (from curves.py); if not specified, opens files in fileTarg
#        par: curveMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        regWeightOut: multiplicative constant on background regularization
#        affineWeight: multiplicative constant on affine regularization
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        mu: initial value for quadratic penalty normalization
#        outputDir: where results are saved
#        saveFile: generic name for saved curves
#        typeConstraint: 'stiched', 'sliding'
#        affine: 'affine', 'euclidean' or 'none'
#        maxIter_cg: max iterations in conjugate gradient
#        maxIter_al: max interation for augmented lagrangian

class CurveMatching(curveMatching.CurveMatching):
    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, verb=True, regWeight=1.0, regWeightOut=1.0, affineWeight = 1.0, testGradient=False, mu = 0.1, outputDir='.', saveFile = 'evolution', typeConstraint='stitched', affine='none', rotWeight = None, scaleWeight = None, transWeight = None,  maxIter_cg=1000, maxIter_al=100):
        if Template==None:
            if fileTempl==None:
                print 'Please provide a template curve'
                return
            else:
                self.fv0 = []
                for ftmp in fileTempl:
                    self.fv0.append(curves.Curve(filename=ftmp))
        else:
            self.fv0 = []
            for ftmp in Template:
                self.fv0.append(curves.Curve(curve=ftmp))
        if Target==None:
            if fileTarg==None:
                print 'Please provide a target curve'
                return
            else:
                self.fv1 = []
                for ftmp in fileTarg:
                    self.fv1.append(curves.Curve(filename=ftmp))
        else:
            self.fv1 = []
            for ftmp in Target:
                self.fv1.append(curves.Curve(curve=ftmp))

        self.dim = self.fv0[0].vertices.shape[1]
        self.outputDir = outputDir  
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)

        if self.dim == 2:
            xmin = 1e10
            xmax = -1e10
            ymin = 1e10
            ymax = -1e10
            for kk in range(len(self.fv0)):
                xmin = min(xmin ,self.fv0[kk].vertices[:,0].min(), self.fv1[kk].vertices[:,0].min())
                xmax = max(xmax, self.fv0[kk].vertices[:,0].max(), self.fv1[kk].vertices[:,0].max())
                ymin = min(ymin, self.fv0[kk].vertices[:,1].min(), self.fv1[kk].vertices[:,1].min())
                ymax = max(ymax, self.fv0[kk].vertices[:,1].max(), self.fv1[kk].vertices[:,1].max())
            dx = 0.01*(xmax-xmin)
            dy = 0.01*(ymax-ymin)
            dxy = min(dx,dy)
            [x,y] = np.mgrid[(xmin-10*dxy):(xmax+10*dxy):dxy, (ymin-10*dxy):(ymax+10*dxy):dxy]
            gridDef = grid.Grid(gridPoints=(x,y))

            self.gridDef = []
            self.gridxy = []
            self.inGrid = []
            Kout = np.ones(gridDef.vertices.shape[0])

            for fk in range(len(self.fv0)):
                vdisc = np.zeros(self.fv0[fk].vertices.shape)
                vdisc0 = np.zeros(self.fv0[fk].vertices.shape)
                vdisc[:,0] = xmin + dxy * (np.floor(0.5+(self.fv0[fk].vertices[:,0] - xmin)/dxy))
                vdisc[:,1] = ymin + dxy * (np.floor(0.5+(self.fv0[fk].vertices[:,1] - ymin)/dxy))
                #print vdisc, self.fv0[fk].vertices
                firstFound = - np.ones(vdisc.shape[0], dtype=np.int)
                vdisc0[0, :] = vdisc[0,:]
                firstFound[0] = 0
                k0 = 1 
                for k in range(1,vdisc.shape[0]):
                    d = ((vdisc[k, :] - vdisc0[0:k0, :])**2).sum(axis=1)
                    #print d
                    kk = np.nonzero(d < 1e-10)
                    #print k, k0, kk[0]
                    if len(kk[0])>0:
                        firstFound[k] = kk[0][0]
                    else:
                        vdisc0[k0, :] = vdisc[k, :]
                        firstFound[k] = k0
                        k0 += 1

                vdisc0 = vdisc0[0:k0, :]
                fdisc = np.zeros(self.fv0[fk].faces.shape,dtype=np.int)
                k0 = 0
                for k in range(self.fv0[fk].faces.shape[0]):
                    if firstFound[self.fv0[fk].faces[k, 0]] != firstFound[self.fv0[fk].faces[k,1]]:
                        fdisc[k0, 0] = firstFound[self.fv0[fk].faces[k, 0]]
                        fdisc[k0, 1] = firstFound[self.fv0[fk].faces[k, 1]]
                        k0 += 1
                fv = curves.Curve(FV=(fdisc,vdisc0))

                gr = grid.Grid()
                gr.copy(gridDef)
                D = gr.signedDistPolygon(fv) ;
                print D.min(), D.max(), dxy
                K = D < 1e-10 ; #gr.inPolygon(self.fv0[kk])
                gr.restrict(K)
                self.inGrid.append(K)
                self.gridDef.append(gr)
                self.gridxy.append(self.gridDef[-1].vertices)
                if np.fabs((fv.vertices[fv.faces[:,1]]- fv.vertices[fv.faces[:,0]]).sum(axis=0)).sum() < 0.0001:
                    print 'closed curve' 
                    K1 = D > - 1e-10 ;
                    Kout = np.multiply(Kout, K1)
                else:
                    print 'open curve'
                    #print self.fv0[kk].vertices[self.fv0[kk].faces[:,1]]- self.fv0[kk].vertices[self.fv0[kk].faces[:,0]]
            gr = grid.Grid()
            gr.copy(gridDef)
            gr.restrict(np.int_(Kout))
            self.gridDef.append(gr)
            self.gridxy.append(self.gridDef[-1].vertices)
            self.gridAll = gridDef       
            self.inGrid.append(np.int_(Kout)) 

                #print self.fv0[fk].vertices.shape

                #print self.fv0[0].vertices.shape
                

                        
        self.fvDef = [] 
        self.fvDefB = [] 
        for fv in self.fv0:
            self.fvDef.append(curves.Curve(curve=fv))
            self.fvDefB.append(curves.Curve(curve=fv))
        self.maxIter_cg = maxIter_cg
        self.maxIter_al = maxIter_al
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.regweightOut = regWeightOut
        if param==None:
            self.param = CurveMatchingParam()
        else:
            self.param = param            

        self.affine = affine
        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones([1, self.affineDim])
        #print self.affineDim, self.affB.rotComp, rotWeight, self.affineWeight
        if (len(self.affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[0,self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[0,self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[0,self.affB.transComp] = transWeight

             
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = []
        self.atTry = []
        self.Afft = []
        self.AfftTry = []
        self.xt = []
        self.nut = []
        self.x0 = []
        self.nu0 = []
        self.nu00 = []
        self.ncurve = len(self.fv0)
        self.npt = np.zeros(self.ncurve)
        k=0
        for fv in self.fv0:
            x0 = fv.vertices
            # for q,n in enumerate(x0):
            #     print q, x0[q]
            fk = fv.faces
            xDef0 = x0[fk[:, 0], :]
            xDef1 = x0[fk[:, 1], :]
            nf = xDef1 - xDef0
            nf = nf[:, (1,0)]
            nf[:,0] = -nf[:, 0]
            nu0 = np.zeros(x0.shape)
                #print nf[1:10]
            for kk,j in enumerate(fk[:,0]):
                nu0[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,1]):
                nu0[j, :] += nf[kk,:]

            self.nu00.append(np.copy(nu0))
            nu0 /= np.sqrt((nu0**2).sum(axis=1)).reshape([x0.shape[0], 1])
            self.npt[k] = x0.shape[0]
            self.x0.append(x0)
            self.nu0.append(nu0)
            self.at.append(np.zeros([self.Tsize, x0.shape[0], x0.shape[1]]))
            self.atTry.append(np.zeros([self.Tsize, x0.shape[0], x0.shape[1]]))
            self.Afft.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftTry.append(np.zeros([self.Tsize, self.affineDim]))
            #print x0.shape, nu0.shape
            self.xt.append(np.tile(x0, [self.Tsize+1, 1, 1]))
            self.nut.append(np.tile(np.array(nu0), [self.Tsize+1, 1, 1]))
            k=k+1

        self.npoints = self.npt.sum()
        self.at.append(np.zeros([self.Tsize, self.npoints, self.dim]))
        self.atTry.append(np.zeros([self.Tsize, self.npoints, self.dim]))
        self.x0.append(np.zeros([self.npoints, self.dim]))
        npt = 0
        for fv0 in self.fv0:
            npt1 = npt + fv0.vertices.shape[0]
            self.x0[self.ncurve][npt:npt1, :] = np.copy(fv0.vertices)
            npt = npt1
        self.xt.append(np.tile(self.x0[self.ncurve], [self.Tsize+1, 1, 1]))

        #if self.dim == 2:
            # xmin = 1e10
            # xmax = -1e10
            # ymin = 1e10
            # ymax = -1e10
            # for kk in range(self.ncurve):
            #     xmin = min(xmin ,self.fv0[kk].vertices[:,0].min(), self.fv1[kk].vertices[:,0].min())
            #     xmax = max(xmax, self.fv0[kk].vertices[:,0].max(), self.fv1[kk].vertices[:,0].max())
            #     ymin = min(ymin, self.fv0[kk].vertices[:,1].min(), self.fv1[kk].vertices[:,1].min())
            #     ymax = max(ymax, self.fv0[kk].vertices[:,1].max(), self.fv1[kk].vertices[:,1].max())
            # dx = 0.01*(xmax-xmin)
            # dy = 0.01*(ymax-ymin)
            # dxy = min(dx,dy)

        self.typeConstraint = typeConstraint

        if typeConstraint == 'stitched':
            self.cval = np.zeros([self.Tsize+1, self.npoints, self.dim])
            self.lmb = np.zeros([self.Tsize+1, self.npoints, self.dim])
            self.constraintTerm = self.constraintTermStitched
            self.constraintTermGrad = self.constraintTermGradStitched
            self.useKernelDotProduct = True
            self.dotProduct = self.kernelDotProduct
        elif typeConstraint == 'sliding':
            self.cval = np.zeros([self.Tsize+1, self.npoints])
            self.lmb = np.zeros([self.Tsize+1, self.npoints])
            self.constraintTerm = self.constraintTermSliding
            self.constraintTermGrad = self.constraintTermGradSliding
            self.useKernelDotProduct = False
            self.dotProduct = self.standardDotProduct            
        else:
            print 'Unrecognized constraint type'
            return
        
        self.mu = np.sqrt(mu)
        self.obj = None
        self.objTry = None
        self.saveFile = saveFile
        self.gradCoeff = self.fv0[0].vertices.shape[0]
        for k in range(self.ncurve):
            self.fv0[k].saveVTK(self.outputDir+'/Template'+str(k)+'.vtk', normals=self.nu0[k])
            self.fv1[k].saveVTK(self.outputDir+'/Target'+str(k)+'.vtk')


    def constraintTermStitched(self, xt, nut, at, Afft):
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize+1):
            zB = np.squeeze(xt[self.ncurve][t, :, :])
            npt = 0
            for k in range(self.ncurve):
                z = np.squeeze(xt[k][t, :, :]) 
                npt1 = npt + self.npt[k]
                cval[t,npt:npt1, :] = z - zB[npt:npt1, :]
                npt = npt1

            obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() + self.cstrFun(cval[t, :]/self.mu).sum())
        return obj,cval


    def cstrFun(self, x):
        return (np.multiply(x,x)/2.)
    #u = np.fabs(x)
    #  return u + np.log((1+np.exp(-2*u))/2)

    def derCstrFun(self, x):
        return x
    #u = np.exp(-2*np.fabs(x))
    #  return np.multiply(np.sign(x), np.divide(1-u, 1+u))

    def constraintTermGradStitched(self, xt, nut, at, Afft):
        lmb = np.zeros(self.cval.shape)
        dxcval = []
        dacval = []
        dAffcval = []
        for u in xt:
            dxcval.append(np.zeros(u.shape))
        for u in at:
            dacval.append(np.zeros(u.shape))
        for u in Afft:
            dAffcval.append(np.zeros(u.shape))
        for t in range(self.Tsize+1):
            zB = np.squeeze(xt[self.ncurve][t, :, :])
            npt = 0
            for k in range(self.ncurve):
                z = np.squeeze(xt[k][t, :, :]) 
                npt1 = npt + self.npt[k]
                lmb[t, npt:npt1] = self.lmb[t, npt:npt1] - self.derCstrFun((z - zB[npt:npt1, :])/self.mu)/self.mu
                dxcval[k][t] = lmb[t, npt:npt1]
                dxcval[-1][t, npt:npt1] = -lmb[t, npt:npt1]
                npt = npt1

        return lmb, dxcval, dacval, dAffcval

    def constraintTermSliding(self, xt, nut, at, Afft):
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        dim2 = self.dim**2
        for t in range(self.Tsize):
            zB = np.squeeze(xt[self.ncurve][t, :, :])
            aB = np.squeeze(at[self.ncurve][t, :, :])
            npt = 0
            r2 = self.param.KparDiffOut.applyK(zB, aB)
            for k in range(self.ncurve):
                npt1 = npt + self.npt[k]
                a = at[k][t]
                x = xt[k][t]
                if self.affineDim > 0:
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A = AB[0:dim2].reshape([self.dim, self.dim])
                    b = AB[dim2:dim2+self.dim]
                else:
                    A = np.zeros([self.dim, self.dim])
                    b = np.zeros(self.dim)
                z = zB[npt:npt1, :]
                nu = np.zeros(x.shape)
                fk = self.fv0[k].faces
                xDef0 = z[fk[:, 0], :]
                xDef1 = z[fk[:, 1], :]
                nf = xDef1 - xDef0
                nf = nf[:, (1,0)]
                nf[:, 0] = -nf[:, 0]
                #nf =  np.cross(xDef1-xDef0, xDef2-xDef0)
                for kk,j in enumerate(fk[:,0]):
                    nu[j, :] += nf[kk,:]
                for kk,j in enumerate(fk[:,1]):
                    nu[j, :] += nf[kk,:]
                nu /= np.sqrt((nu**2).sum(axis=1)).reshape([nu.shape[0], 1])


                r = self.param.KparDiff.applyK(x, a, firstVar=z) + np.dot(z, A.T) + b
                cval[t,npt:npt1] = np.squeeze(np.multiply(nu, r - r2[npt:npt1, :]).sum(axis=1))
                npt = npt1

            obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() + self.cstrFun(cval[t, :]/self.mu).sum())
            #print 'slidingV2', obj
        return obj,cval

    def constraintTermGradSliding(self, xt, nut, at, Afft):
        lmb = np.zeros(self.cval.shape)
        dxcval = []
        dAffcval = []
        dacval = []
        for u in xt:
            dxcval.append(np.zeros(u.shape))
        for u in at:
            dacval.append(np.zeros(u.shape))
        for u in Afft:
            dAffcval.append(np.zeros(u.shape))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            zB = xt[self.ncurve][t]
            aB = at[self.ncurve][t]
            r2 = self.param.KparDiffOut.applyK(zB, aB)
            npt = 0
            for k in range(self.ncurve):
                npt1 = npt + self.npt[k]
                a = at[k][t]
                x = xt[k][t]
                if self.affineDim > 0:
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A = AB[0:dim2].reshape([self.dim, self.dim])
                    b = AB[dim2:dim2+self.dim]
                else:
                    A = np.zeros([self.dim, self.dim])
                    b = np.zeros(self.dim)
                z = zB[npt:npt1, :]
                fk = self.fv0[k].faces
                nu = np.zeros(x.shape)
                xDef0 = z[fk[:, 0], :]
                xDef1 = z[fk[:, 1], :]
                nf = xDef1 - xDef0
                nf = nf[:, (1,0)] 
                nf[:,0] = -nf[:,0]
                #nf =  np.cross(xDef1-xDef0, xDef2-xDef0)
                for kk,j in enumerate(fk[:,0]):
                    nu[j, :] += nf[kk,:]
                for kk,j in enumerate(fk[:,1]):
                    nu[j, :] += nf[kk,:]
                normNu = np.sqrt((nu**2).sum(axis=1))
                nu /= normNu.reshape([nu.shape[0], 1])

                dv = self.param.KparDiff.applyK(x, a, firstVar=z) + np.dot(z, A.T) + b - r2[npt:npt1, :]
                lmb[t, npt:npt1] = self.lmb[t, npt:npt1] - self.derCstrFun(np.multiply(nu, dv).sum(axis=1)/self.mu)/self.mu
                #lnu = np.multiply(nu, np.mat(lmb[t, npt:npt1]).T)
                lnu = np.multiply(nu, lmb[t, npt:npt1].reshape([self.npt[k], 1]))
                #print lnu.shape
                dxcval[k][t] = self.param.KparDiff.applyDiffKT(z, [a], [lnu], firstVar=x)
                dxcval[self.ncurve][t][npt:npt1, :] += (self.param.KparDiff.applyDiffKT(x, [lnu], [a], firstVar=z)
                                         - self.param.KparDiffOut.applyDiffKT(zB, [lnu], [aB], firstVar=z))
                dxcval[self.ncurve][t] -= self.param.KparDiffOut.applyDiffKT(z, [aB], [lnu], firstVar=zB)
                dxcval[self.ncurve][t][npt:npt1, :] += np.dot(lnu, A)
                dacval[k][t] = self.param.KparDiff.applyK(z, lnu, firstVar=x)
                if self.affineDim > 0:
                    dAffcval[k][t, :] = (np.dot(self.affineBasis.T, np.vstack([np.dot(lnu.T, z).reshape([dim2,1]), lnu.sum(axis=0).reshape([self.dim,1])]))).flatten()
                dacval[self.ncurve][t] -= self.param.KparDiffOut.applyK(z, lnu, firstVar=zB)
                lv = np.multiply(dv, lmb[t, npt:npt1].reshape([self.npt[k],1]))
                lv /= normNu.reshape([nu.shape[0], 1])
                lv -= np.multiply(nu, np.multiply(nu, lv).sum(axis=1).reshape([nu.shape[0], 1]))
                lv = lv[:, (1,0)]
                lv[:,1] = -lv[:,1]
                lvf = lv[fk[:,0]] + lv[fk[:,1]]
                dnu = np.zeros(x.shape)
                #foo = np.cross(xDef2-xDef1, lvf)
                for kk,j in enumerate(fk[:,0]):
                    dnu[j, :] += lvf[kk,:]
                    #foo = np.cross(xDef0-xDef2, lvf)
                for kk,j in enumerate(fk[:,1]):
                    dnu[j, :] -= lvf[kk,:]
                dxcval[self.ncurve][t][npt:npt1, :] -= dnu 
                npt = npt1

                #obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() + np.multiply(cval[t, :], cval[t, :]).sum()/(2*self.mu))
        return lmb, dxcval, dacval, dAffcval


    def testConstraintTerm(self, xt, nut, at, Afft):
        xtTry = []
        atTry = []
        AfftTry = []
        eps = 0.00000001
        for k in range(self.ncurve):
            xtTry.append(xt[k] + eps*np.random.randn(self.Tsize+1, self.npt[k], self.dim))
        xtTry.append(xt[self.ncurve] + eps*np.random.randn(self.Tsize+1, self.npoints, self.dim))
        for k in range(self.ncurve):
            atTry.append(at[k] + eps*np.random.randn(self.Tsize, self.npt[k], self.dim))
        atTry.append(at[self.ncurve] + eps*np.random.randn(self.Tsize, self.npoints, self.dim))

        if self.affineDim > 0:
            for k in range(self.ncurve):
                AfftTry.append(Afft[k] + eps*np.random.randn(self.Tsize, self.affineDim))
            

        u0 = self.constraintTerm(xt, nut, at, Afft)
        ux = self.constraintTerm(xtTry, nut, at, Afft)
        ua = self.constraintTerm(xt, nut, atTry, Afft)
        [l, dx, da, dA] = self.constraintTermGrad(xt, nut, at, Afft)
        vx = 0
        for k in range(self.ncurve+1):
            vx += np.multiply(dx[k], xtTry[k]-xt[k]).sum()/eps
        va = 0
        for k in range(self.ncurve+1):
            va += np.multiply(da[k], atTry[k]-at[k]).sum()/eps
        print 'Testing constraints:'
        print 'var x:', self.Tsize*(ux[0]-u0[0])/(eps), -vx 
        print 'var a:', self.Tsize*(ua[0]-u0[0])/(eps), -va 
        if self.affineDim > 0:
            uA = self.constraintTerm(xt, nut, at, AfftTry)
            vA = 0
            for k in range(self.ncurve):
                vA += np.multiply(dA[k], AfftTry[k]-Afft[k]).sum()/eps
            print 'var affine:', self.Tsize*(uA[0]-u0[0])/(eps), -vA 

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian = False):
        param = self.param
        timeStep = 1.0/self.Tsize
        xt = []
        nut = []
        dim2 = self.dim**2
        if withJacobian:
            Jt = []
            for k in range(self.ncurve):
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                if self.affineDim > 0:
                    for t in range(self.Tsize):
                        AB = np.dot(self.affineBasis, Afft[k][t]) 
                        A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                        A[1][t] = AB[dim2:dim2+self.dim]
                xJ = evol.landmarkDirectEvolutionEuler(self.x0[k], at[k], param.KparDiff, affine = A, withNormals = self.nu0[k], withJacobian = True)
                xt.append(xJ[0])
                nut.append(xJ[1])
                Jt.append(xJ[2])
            xJ = evol.landmarkDirectEvolutionEuler(self.x0[self.ncurve], at[self.ncurve], param.KparDiffOut, withJacobian=True)
            xt.append(xJ[0])
            Jt.append(xJ[1])
        else:
            for k in range(self.ncurve):
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                if self.affineDim > 0:
                    for t in range(self.Tsize):
                        AB = np.dot(self.affineBasis, Afft[k][t]) 
                        A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                        A[1][t] = AB[dim2:dim2+self.dim]
                xJ = evol.landmarkDirectEvolutionEuler(self.x0[k], at[k], param.KparDiff, withNormals = self.nu0[k], affine = A)
                xt.append(xJ[0])
                nut.append(xJ[1])
            xt.append(evol.landmarkDirectEvolutionEuler(self.x0[self.ncurve], at[self.ncurve], param.KparDiffOut))
        #print xt[-1, :, :]
        #print obj
        obj=0
        for t in range(self.Tsize):
            zB = np.squeeze(xt[self.ncurve][t, :, :])
            for k in range(self.ncurve):
                z = np.squeeze(xt[k][t, :, :]) 
                a = np.squeeze(at[k][t, :, :])
                ra = param.KparDiff.applyK(z,a)
                obj += self.regweight*timeStep*np.multiply(a, ra).sum()
                if self.affineDim > 0:
                    obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[k][t].shape), Afft[k][t]**2).sum()
                #print t,k,obj

            a = np.squeeze(at[self.ncurve][t, :, :])
            ra = param.KparDiffOut.applyK(zB,a)
            obj += self.regweightOut*timeStep*np.multiply(a, ra).sum()

            #print 'obj before constraints:', obj
        cstr = self.constraintTerm(xt, nut, at, Afft)
        obj += cstr[0]

        if withJacobian:
            return obj, xt, Jt, cstr[1]
        elif withTrajectory:
            return obj, xt, cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0
            for fv1 in self.fv1:
                self.obj0 += self.param.fun_obj0(fv1, self.param.KparDist) / (self.param.sigmaError**2)

            (self.obj, self.xt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            selfobj = 2*self.obj0

            npt = 0 
            for k in range(self.ncurve):
                npt1 = npt + self.npt[k]
                self.fvDefB[k].updateVertices(np.squeeze(self.xt[-1][self.Tsize, npt:npt1, :]))
                selfobj += self.param.fun_obj(self.fvDefB[k], self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                self.fvDef[k].updateVertices(np.squeeze(self.xt[k][self.Tsize, :, :]))
                selfobj += self.param.fun_obj(self.fvDef[k], self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                npt = npt1

                #print 'Deformation based:', self.obj, 'data term:', selfobj, self.regweightOut
            self.obj += selfobj

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        atTry = []
        for k in range(self.ncurve+1):
            atTry.append(self.at[k] - eps * dir[k].diff)
        AfftTry = []
        if self.affineDim > 0:
            for k in range(self.ncurve):
                AfftTry.append(self.Afft[k] - eps * dir[k].aff)
        else:
            for k in range(self.ncurve):
                AfftTry.append(self.Afft[k])
        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry = 0

        ff = []
        npt = 0
        for k in range(self.ncurve):
            npt1 = npt + self.npt[k]
            ff = curves.Curve(curve=self.fvDef[k])
            ff.updateVertices(np.squeeze(foo[1][self.ncurve][self.Tsize, npt:npt1, :]))
            objTry += self.param.fun_obj(ff, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
            ff.updateVertices(np.squeeze(foo[1][k][self.Tsize, :, :]))
            objTry +=  self.param.fun_obj(ff, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
            npt = npt1
            #print 'Deformation based:', foo[0], 'data term:', objTry+self.obj0
        objTry += foo[0]+ 2*self.obj0

        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500


        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            self.cval = foo[2]


        return objTry


    def covectorEvolution(self, at, Afft, px1, pnu1):
        M = self.Tsize
        timeStep = 1.0/M
        xt = []
        nut = []
        pxt = []
        pnut = []
        A = []
        dim2 = self.dim**2
        for k in range(self.ncurve):
            A.append([np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])])
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A[k][0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[k][1][t] = AB[dim2:dim2+self.dim]
            xJ = evol.landmarkDirectEvolutionEuler(self.x0[k], at[k], self.param.KparDiff, withNormals = self.nu0[k], affine = A[k])
            xt.append(xJ[0])
            nut.append(xJ[1])
            pxt.append(np.zeros([M, self.npt[k], self.dim]))
            pnut.append(np.zeros([M, self.npt[k], self.dim]))
            #print pnu1[k].shape
            pxt[k][M-1, :, :] = px1[k]
            pnut[k][M-1] = pnu1[k]
        
        xt.append(evol.landmarkDirectEvolutionEuler(self.x0[self.ncurve], at[self.ncurve], self.param.KparDiffOut))
        pxt.append(np.zeros([M, self.npoints, self.dim]))
        pxt[self.ncurve][M-1, :, :] = px1[self.ncurve]
        #lmb = np.zeros([self.npoints, self.dim])
        foo = self.constraintTermGrad(xt, nut, at, Afft)
        lmb = foo[0]
        dxcval = foo[1]
        dacval = foo[2]
        dAffcval = foo[3]
        
        for k in range(self.ncurve):
            pxt[k][M-1, :, :] += timeStep * dxcval[k][M]
        pxt[self.ncurve][M-1, :, :] += timeStep * dxcval[self.ncurve][M]
        
        for t in range(M-1):
            npt = 0
            for k in range(self.ncurve):
                npt1 = npt + self.npt[k]
                px = np.squeeze(pxt[k][M-t-1, :, :])
                z = np.squeeze(xt[k][M-t-1, :, :])
                a = np.squeeze(at[k][M-t-1, :, :])
                zpx = np.copy(dxcval[k][M-t-1])
                a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*self.regweight*a[np.newaxis,...]))
                a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
                zpx += self.param.KparDiff.applyDiffKT(z, a1, a2)
                if self.affineDim > 0:
                    zpx += np.dot(px, A[k][0][M-t-1])
                    
                pxt[k][M-t-2, :, :] = np.squeeze(pxt[k][M-t-1, :, :]) + timeStep * zpx
                npt = npt1
            px = np.squeeze(pxt[self.ncurve][M-t-1, :, :])
            z = np.squeeze(xt[self.ncurve][M-t-1, :, :])
            a = np.squeeze(at[self.ncurve][M-t-1, :, :])
            zpx = np.copy(dxcval[self.ncurve][M-t-1])
            a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*self.regweightOut*a[np.newaxis,...]))
            a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
            zpx += self.param.KparDiffOut.applyDiffKT(z, a1, a2)
            pxt[self.ncurve][M-t-2, :, :] = np.squeeze(pxt[self.ncurve][M-t-1, :, :]) + timeStep * zpx
            

        return pxt, pnut, xt, nut, dacval, dAffcval


    def HamiltonianGradient(self, at, Afft, px1, pnu1, getCovector = False):
        (pxt, pnut, xt, nut, dacval, dAffcval) = self.covectorEvolution(at, Afft, px1, pnu1)

        dat = []
        if self.useKernelDotProduct:
            for k in range(self.ncurve+1):
                dat.append(2*self.regweight*at[k] - pxt[k] - dacval[k])
        else:
            for k in range(self.ncurve+1):
                dat.append(-dacval[k])
            for t in range(self.Tsize):
                npt = 0
                for k in range(self.ncurve):
                    npt1 = npt + self.npt[k]
                    #print k, at[k][t].shape, pxt[k][t].shape, npt, npt1, dat[k][t].shape
                    dat[k][t] += self.param.KparDiff.applyK(xt[k][t], 2*self.regweight*at[k][t] - pxt[k][t])
                    npt=npt1
                dat[self.ncurve][t] += self.param.KparDiffOut.applyK(xt[self.ncurve][t], 2*self.regweightOut*at[self.ncurve][t] - pxt[self.ncurve][t])
        dAfft = []
        if self.affineDim > 0:
            for k in range(self.ncurve):
                dAfft.append(2*np.multiply(self.affineWeight, Afft[k]) - dAffcval[k])
                for t in range(self.Tsize):
                    dA = np.dot(pxt[k][t].T, xt[k][t]).reshape([self.dim**2, 1])
                    db = pxt[k][t].sum(axis=0).reshape([self.dim,1]) 
                    dAff = np.dot(self.affineBasis.T, np.vstack([dA, db]))
                    dAfft[k][t] -=  dAff.reshape(dAfft[k][t].shape)
                dAfft[k] = np.divide(dAfft[k], self.affineWeight)
 
        if getCovector == False:
            return dat, dAfft, xt, nut
        else:
            return dat, dAfft, xt, nut, pxt, pnut

    def endPointGradient(self):
        px1 = []
        pxB = np.zeros([self.npoints, self.dim])
        npt = 0 
        for k in range(self.ncurve):
            npt1 = npt + self.npt[k]
            px = -self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist) / self.param.sigmaError**2
            pxB[npt:npt1, :] = -self.param.fun_objGrad(self.fvDefB[k], self.fv1[k], self.param.KparDist) / self.param.sigmaError**2
            px1.append(px)            
            npt = npt1

        px1.append(pxB)
        return px1

    def addProd(self, dir1, dir2, beta):
        dir = []
        for k in range(self.ncurve):
            dir.append(curveMatching.Direction())
            dir[k].diff = dir1[k].diff + beta * dir2[k].diff
            if self.affineDim > 0:
                dir[k].aff = dir1[k].aff + beta * dir2[k].aff
        dir.append(curveMatching.Direction())
        dir[-1].diff = dir1[-1].diff + beta * dir2[-1].diff
        #     for k in range(self.nsurf):
        #         dir[1].append(dir1[1][k] + beta * dir2[1][k])
        return dir

    def copyDir(self, dir0):
        dir = []
        for d in dir0:
            dir.append(curveMatching.Direction())
            dir[-1].diff = np.copy(d.diff)
            dir[-1].aff = np.copy(d.aff)
        return dir

        
    def kernelDotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for k in range(self.ncurve):
            for t in range(self.Tsize):
                z = np.squeeze(self.xt[k][t, :, :])
                gg = np.squeeze(g1[k].diff[t, :, :])
                u = self.param.KparDiff.applyK(z, gg)
                if self.affineDim > 0:
                    uu = np.multiply(g1[k].aff[t], self.affineWeight)
                ll = 0
                for gr in g2:
                    ggOld = np.squeeze(gr[k].diff[t, :, :])
                    res[ll]  +=  np.multiply(ggOld,u).sum()
                    if self.affineDim > 0:
                        res[ll] += np.multiply(uu, gr[k].aff[t]).sum()
                    ll = ll + 1
      
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[self.ncurve][t, :, :])
            gg = np.squeeze(g1[self.ncurve].diff[t, :, :])
            u = self.param.KparDiffOut.applyK(z, gg)
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr[self.ncurve].diff[t, :, :]) 
                res[ll]  +=  np.multiply(ggOld,u).sum()
                ll = ll + 1

        return res

    def standardDotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for ll,gr in enumerate(g2):
            res[ll]=0
            for k in range(self.ncurve):
                res[ll] += np.multiply(g1[k].diff, gr[k].diff).sum()
                if self.affineDim > 0:
                    uu = np.multiply(g1[k].aff, self.affineWeight)
                    res[ll] += np.multiply(uu, gr[k].aff).sum()
                    #+np.multiply(g1[1][k][:, dim2:dim2+self.dim], gr[1][k][:, dim2:dim2+self.dim]).sum())
            res[ll] += np.multiply(g1[self.ncurve].diff, gr[self.ncurve].diff).sum() / self.coeffZ
        return res



    def getGradient(self, coeff=1.0):
        px1 = self.endPointGradient()
        #px1.append(np.zeros([self.npoints, self.dim]))
        pnu1 = []
        for k in range(self.ncurve):
            pnu1.append(np.zeros(self.nu0[k].shape))
            #for p in px1:
            #print p.sum()
        foo = self.HamiltonianGradient(self.at, self.Afft, px1, pnu1)
        grd = []
        for kk in range(self.ncurve):
            grd.append(curveMatching.Direction())
            grd[kk].diff = foo[0][kk] / (coeff*self.Tsize)
            if self.affineDim > 0:
                grd[kk].aff = foo[1][kk] / (coeff*self.Tsize)
        grd.append(curveMatching.Direction())
        grd[self.ncurve].diff = foo[0][self.ncurve] * (self.coeffZ/(coeff*self.Tsize))
        return grd

    def randomDir(self):
        dirfoo = []
        for k in range(self.ncurve):
            dirfoo.append(curveMatching.Direction())
            dirfoo[k].diff = np.random.randn(self.Tsize, self.npt[k], self.dim)
            if self.affineDim > 0:
                dirfoo[k].aff = np.random.randn(self.Tsize, self.affineDim)
        dirfoo.append(curveMatching.Direction())
        dirfoo[-1].diff = np.random.randn(self.Tsize, self.npoints, self.dim)
        return dirfoo

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = []
        for a in self.atTry:
            self.at.append(np.copy(a))
        self.Afft = []
        for a in self.AfftTry:
            self.Afft.append(np.copy(a))

    def endOfIteration(self):
        (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withJacobian=True)
        logging.info('mean constraint %.3f %.3f'%(np.sqrt((self.cval**2).sum()/self.cval.size), np.fabs(self.cval).sum() / self.cval.size))
        #self.testConstraintTerm(self.xt, self.nut, self.at, self.Afft)
        self.iter += 1
        if self.iter %10 == 0:
            self.printResults(Jt)
#        else:
        nn = 0 
        for k in range(self.ncurve):
            n1 = self.xt[k].shape[1] ;
            self.fvDefB[k].updateVertices(np.squeeze(self.xt[-1][-1, nn:nn+n1, :]))
            self.fvDef[k].updateVertices(np.squeeze(self.xt[k][-1, :, :]))
            nn += n1
                
    def printResults(self,Jt):
        nn = 0 ;
        yt0 = [] ;
        for k in range(self.ncurve):
            if self.dim==2:
                A = self.affB.getTransforms(self.Afft[k])
                (xt,yt) = evol.landmarkDirectEvolutionEuler(self.x0[k], self.at[k], self.param.KparDiff, affine=A, withPointSet=self.gridxy[k])
                yt0.append(yt)
                #print xt.shape, yt.shape
            n1 = self.xt[k].shape[1] ;
            for kk in range(self.Tsize+1):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[-1][kk, nn:nn+n1, :]))
                self.fvDef[k].saveVTK(self.outputDir +'/'+ self.saveFile+str(k)+'Out'+str(kk)+'.vtk', scalars = Jt[-1][kk, nn:nn+n1], scal_name='Jacobian')
                self.fvDef[k].updateVertices(np.squeeze(self.xt[k][kk, :, :]))
                self.fvDef[k].saveVTK(self.outputDir +'/'+self.saveFile+str(k)+'In'+str(kk)+'.vtk', scalars = Jt[k][kk, :], scal_name='Jacobian')
                if self.dim == 2:
                    self.gridDef[k].vertices = np.copy(yt[kk, :, :])
                    self.gridDef[k].saveVTK(self.outputDir +'/grid'+str(k)+'In'+str(kk)+'.vtk')
            nn += n1
            #print xt.shape, yt.shape
            I1 = np.nonzero(self.inGrid[-1])
        if self.dim==2:
            (xt,yt) = evol.landmarkDirectEvolutionEuler(self.x0[self.ncurve], self.at[self.ncurve], self.param.KparDiffOut, withPointSet=self.gridxy[-1])
            yt0.append(yt)
            for kk in range(self.Tsize+1):
                self.gridDef[-1].vertices = np.copy(yt[kk, :, :])
                self.gridDef[-1].saveVTK(self.outputDir +'/gridOut'+str(kk)+'.vtk')
                for k in range(self.ncurve+1):
                    I1 = np.nonzero(self.inGrid[k])
                    self.gridAll.vertices[I1] = np.copy(yt0[k][kk, ...])
                self.gridAll.saveVTK(self.outputDir +'/gridAll'+str(kk)+'.vtk')

    def endOptim(self):
        if self.iter %10 > 0:
            (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withJacobian=True)
            self.printResults(Jt)

    def optimizeMatching(self):
	self.coeffZ = 1.0
	grd = self.getGradient(self.gradCoeff)
	[grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 1000
        self.muEps = 0.01
        it = 0
        self.muEpsCount = 1;
        while (self.muEpsCount < 20) & (it<self.maxIter_al)  :
            print 'Starting Minimization: gradEps = ', self.gradEps, ' muEps = ', self.muEps, ' mu = ', self.mu
            self.iter = 0 
            #self.coeffZ = max(1.0, self.mu)
            cg.cg(self, verb = self.verb, maxIter = self.maxIter_cg, TestGradient = self.testGradient, epsInit=0.1)
            if self.converged:
                if (((self.cval**2).sum()/self.cval.size) > self.muEps**2):
                    self.mu *= 0.1
                    self.gradEps *= 10
                else:
                    self.gradEps *= .5
                    for t in range(self.Tsize+1):
                        self.lmb[t, ...] -= 0.5*self.derCstrFun(self.cval[t, ...]/self.mu)/self.mu
                    print 'mean lambdas', np.fabs(self.lmb).sum() / self.lmb.size
                    self.muEps = np.sqrt((self.cval**2).sum()/(1.5*self.cval.size))
                    self.muEpsCount += 1
                    #self.muEps /2
            # else:
            #     self.mu *= 0.9
            self.obj = None
            it = it+1
            
            #return self.fvDef

