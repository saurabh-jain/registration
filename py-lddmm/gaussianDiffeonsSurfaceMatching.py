import os
import numpy as np
import numpy.linalg as LA
import scipy as sp
import surfaces
import kernelFunctions as kfun
import gaussianDiffeons as gd
import pointEvolution as evol
import conjugateGradient as cg
import surfaceMatching
from affineBasis import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normalization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam(surfaceMatching.SurfaceMatchingParam):
    def __init__(self, timeStep = .1, sigmaKernel = 6.5, sigmaDist=2.5, sigmaError=1.0, errorType='measure'):
        surfaceMatching.SurfaceMatchingParam.__init__(self, timeStep = timeStep, sigmaKernel =  sigmaKernel, sigmaDist = sigmaDist, sigmaError = sigmaError, typeKernel = 'gauss', errorType=errorType)
        self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        self.KparDist = kfun.Kernel(name = 'gauss', sigma = self.sigmaDist)

class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []


## Main class for surface matching
#        Template: surface class (from surface.py); if not specified, opens fileTemp
#        Target: surface class (from surface.py); if not specified, opens fileTarg
#        param: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        affineWeight: multiplicative constant on affine regularization
#        rotWeight: multiplicative constant on affine regularization (supercedes affineWeight)
#        scaleWeight: multiplicative constant on scale regularization (supercedes affineWeight)
#        transWeight: multiplicative constant on translation regularization (supercedes affineWeight)
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'similitude', 'euclidean', 'translation' or 'none'
#        maxIter: max iterations in conjugate gradient
class SurfaceMatching(surfaceMatching.SurfaceMatching):
    def __init__(self, Template=None, Target=None, Diffeons=None, EpsilonNet=None, DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, zeroVar=False, fileTempl=None,
                 fileTarg=None, param=None, maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                print 'Please provide a template surface'
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
        if Target==None:
            if fileTarg==None:
                print 'Please provide a target surface'
                return
            else:
                self.fv1 = surfaces.Surface(filename=fileTarg)
        else:
            self.fv1 = surfaces.Surface(surf=Target)

        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)
        self.fvDef = surfaces.Surface(surf=self.fv0)
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.affine = affine
        affB = AffineBasis(self.dim, affine)
        self.affineDim = affB.affineDim
        self.affineBasis = affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[affB.rotComp] = rotWeight
        if (len(affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[affB.simComp] = scaleWeight
        if (len(affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[affB.transComp] = transWeight

        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param
        self.x0 = self.fv0.vertices
        if Diffeons==None:
            if EpsilonNet==None:
                if DiffeonEpsForNet==None:
                    if DiffeonSegmentationRatio==None:
                        self.c0 = np.copy(self.x0) ;
                        self.S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                        self.idx = None
                    else:
                        (self.c0, self.S0, self.idx) = gd.generateDiffeonsFromSegmentation(self.fv0, DiffeonSegmentationRatio)
                        #self.S0 *= self.param.sigmaKernel**2;
                else:
                    (self.c0, self.S0, self.idx) = gd.generateDiffeonsFromNet(self.fv0, DiffeonEpsForNet)
            else:
                (self.c0, self.S0, self.idx) = gd.generateDiffeons(self.fv0, EpsilonNet[0], EpsilonNet[1])
        else:
            (self.c0, self.S0, self.idx) = Diffeons
        if zeroVar:
            self.S0 = np.zeros(self.S0.shape)

        self.ndf = self.c0.shape[0]
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.c0.shape[0], self.x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.c0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.ct = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.St = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')

    # def dataTerm(self, _fvDef):
    #     obj = self.param.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
    #     return obj

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian=False, initial = None):
        if initial == None:
            x0 = self.x0
            c0 = self.c0
            S0 = self.S0
        else:
            (x0, c0, S0) = initial
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        if withJacobian:
            (xt, ct, St, Jt)  = evol.gaussianDiffeonsEvolutionEuler(x0, c0, S0, at, param.sigmaKernel, affine=A, withJacobian=True)
        else:
            (xt, ct, St)  = evol.gaussianDiffeonsEvolutionEuler(x0, c0, S0, at, param.sigmaKernel, affine=A)
        #print xt[-1, :, :]
        #print obj
        obj=0
        #print St.shape
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            c = np.squeeze(ct[t, :, :])
            S = np.squeeze(St[t, :, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            rcc = gd.computeProducts(c, S, param.sigmaKernel)
            obj = obj + self.regweight*timeStep*np.multiply(a, np.dot(rcc,a)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, ct, St, Jt
        elif withTrajectory:
            return obj, xt,  ct, St
        else:
            return obj

    
    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt, self.ct, self.St) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)
            #print self.obj0, self.obj

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        obj, xt, ct, St = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry += obj

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(xt[-1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry

            #print 'objTry=',objTry, dir.diff.sum()
        return objTry



    def endPointGradient(self):
        px = self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist)
        return px / self.param.sigmaError**2


    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        pc1 = np.zeros(self.c0.shape)
        pS1 = np.zeros(self.S0.shape)
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        foo = evol.gaussianDiffeonsGradient(self.x0, self.c0, self.S0,
                                            self.at, px1, pc1, pS1, self.param.sigmaKernel, self.regweight, affine=A)
        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
            grd.aff /= (coeff*self.Tsize)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)
        dir.aff = np.copy(dir0.aff)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.ndf, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            c = np.squeeze(self.ct[t, :, :])
            S = np.squeeze(self.St[t, :, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            rcc = gd.computeProducts(c, S, self.param.sigmaKernel)
            (L, W) = LA.eigh(rcc)
            rcc += (L.max()/1000)*np.eye(rcc.shape[0])
            u = np.dot(rcc, gg)
            uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    res[ll] += np.multiply(uu, gr.aff[t]).sum()
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        #print self.at

    def endOfIteration(self):
        (obj1, self.xt, self.ct, self.St, Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True, withJacobian=True)
        for kk in range(self.Tsize+1):
            self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
            if self.idx == None:
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
            else:
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
            gd.saveDiffeons(self.outputDir +'/'+ self.saveFile+'Diffeons'+str(kk)+'.vtk', self.ct[kk,:,:], self.St[kk,:,:,:])

    def restart(self, EpsilonNet=None, DiffeonEpsForNet=None, DiffeonSegmentationRatio=None):
        if EpsilonNet==None:
            if DiffeonEpsForNet==None:
                if DiffeonSegmentationRatio==None:
                    c0 = np.copy(self.x0) ;
                    S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                    #net = range(c0.shape[0])
                    idx = range(c0.shape[0])
                else:
                    (c0, S0, idx) = gd.generateDiffeonsFromSegmentation(self.fv0, DiffeonSegmentationRatio)
                    #self.S0 *= self.param.sigmaKernel**2;
            else:
                (c0, S0, idx) = gd.generateDiffeonsFromNet(self.fv0, DiffeonEpsForNet)
        else:
            net = EspilonNet[2] 
            (c0, S0, idx) = gd.generateDiffeons(self.fv0, EpsilonNet[0], EpsilonNet[1])

        at = np.zeros([self.Tsize, c0.shape[0], self.x0.shape[1]])
        fvDef = surfaces.Surface(surf=self.fvDef)
        for t in range(self.Tsize):
            fvDef.updateVertices(np.squeeze(self.xt[t, :, :]))
            (AV, AF) = fvDef.computeVertexArea()
            weights = np.zeros([c0.shape[0], self.c0.shape[0]])
            diffArea = np.zeros(self.c0.shape[0])
            diffArea2 = np.zeros(c0.shape[0])
            for k in range(self.npt):
                diffArea[self.idx[k]] += AV[k] 
                diffArea2[idx[k]] += AV[k]
                weights[idx[k], self.idx[k]] += AV[k]
            weights /= diffArea.reshape([1, self.c0.shape[0]])
            at[t] = np.dot(weights, self.at[t, :, :])
        self.c0 = c0
        self.idx = idx
        self.S0 = S0
        self.at = at
        self.ndf = self.c0.shape[0]
        self.ct = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.St = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.optimizeMatching()


    def optimizeMatching(self):
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        print 'Gradient lower bound: ', self.gradEps
        cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

