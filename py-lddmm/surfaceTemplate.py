import os
import numpy as np
import scipy as sp
import copy
import surfaces
import kernelFunctions as kfun
import pointEvolution as evol
import surfaceMatching as smatch
import multiprocessing as mp
import conjugateGradient as cg
from affineBasis import *

class Direction:
    def __init__(self):
        self.prior = []
        self.all = []

## Main class for surface template estimation
#        HyperTmpl: surface class (from surface.py); if not specified, opens fileHTempl
#        Targets: list of surface class (from surface.py); if not specified, open them from fileTarg
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
class SurfaceTemplate(smatch.SurfaceMatching):

    def __init__(self, HyperTmpl=None, Targets=None, fileHTempl=None, fileTarg=None, param=None, maxIter=1000, lambdaPrior = 1.0, regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if HyperTmpl==None:
            if fileHTempl==None:
                print 'Please provide A hyper-template surface'
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileHTempl)
        else:
            self.fv0 = surfaces.Surface(surf=HyperTmpl)
        if Targets==None:
            if fileTarg==None:
                print 'Please provide Target surfaces'
                return
            else:
                for ff in fileTarg:
                    self.fv1.append(surfaces.Surface(filename=ff))
        else:
            self.fv1 = []
            for ff in Targets:
                self.fv1.append(surfaces.Surface(surf=ff))

        self.Ntarg = len(self.fv1)
        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        self.saveFile = saveFile
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)

        self.fv0.saveVTK(self.outputDir +'/'+ 'HyperTemplate.vtk')
        for kk in range(self.Ntarg):
            self.fv1[kk].saveVTK(self.outputDir +'/'+ 'Target'+str(kk)+'.vtk')

        self.fvTmpl = surfaces.Surface(surf=self.fv0)
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = 1.
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

        self.lambdaPrior = lambdaPrior
        if param==None:
            self.param = smatch.SurfaceMatchingParam()
        else:
            self.param = param

        self.Tsize = int(round(1.0/self.param.timeStep))

        # Htempl to Templ
        self.at = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])

        #Template to all
        self.atAll = []
        self.atAllTry = []
        self.xtAll = []
        self.fvDef = []
        self.AfftAll = []
        self.AfftAllTry = []
        for ff in self.fv1:
            self.atAll.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            #smatch.SurfaceMatching(Template=self.fvTmpl, Target=ff,par=self.param))
            self.atAllTry.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            self.AfftAll.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftAllTry.append(np.zeros([self.Tsize, self.affineDim]))
            self.xtAll.append(np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1]))
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.tmplCoeff = 1.0#float(self.Ntarg)
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]

    def init(self, ff):
        self.fv0 = ff.fvTmpl
        self.fv1 = ff.fv1

        self.fvTmpl = surfaces.Surface(surf=self.fv0)
        self.maxIter = ff.maxIter
        self.verb = ff.verb
        self.testGradient = ff.testGradient
        self.regweight = 1.
        self.lambdaPrior = ff.lambdaPrior
        self.param = ff.param

        self.Tsize = int(round(1.0/self.param.timeStep))

        self.at = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.atAll = []
        self.atAllTry = []
        self.AfftAll = []
        self.AfftAllTry = []
        self.xtAll = []
        for f0 in self.fv1:
            self.atAll.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            #smatch.SurfaceMatching(Template=self.fvTmpl, Target=ff,par=self.param))
            self.atAllTry.append(np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]]))
            self.AfftAll.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftAllTry.append(np.zeros([self.Tsize, self.affineDim]))
            self.xtAll.append(np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1]))
            self.fvDef.append(surfaces.Surface(surf=fv0))

        self.Ntarg = len(self.fv1)
        self.tmplCoeff = 1.0 #float(self.Ntarg)
        self.obj = None #ff.obj
        self.obj0 = ff.obj0
        self.objTry = ff.objTry


        for kk in range(self.Ntarg):
            self.atAll[kk] = np.copy(ff.atAll[kk])
            self.AfftAll[kk] = np.copy(ff.AfftAll[kk])
            self.xtAll[kk] = np.copy(ff.xt[kk])


    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0
            for fv in self.fv1:
                self.obj0 += self.param.fun_obj0(fv, self.param.KparDist) / (self.param.sigmaError**2)

            self.obj = self.obj0

            # Regularization part
            foo = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.xt = np.copy(foo[1])
            self.fvTmpl.updateVertices(np.squeeze(self.xt[-1]))
            self.obj += foo[0]*self.lambdaPrior
            ff = surfaces.Surface(surf=self.fv0)
            for (kk, a) in enumerate(self.atAll):
                foo = self.objectiveFunDef(a, self.AfftAll[kk], withTrajectory=True, x0 = self.fvTmpl.vertices)
                ff.updateVertices(np.squeeze(foo[1][-1]))
                self.obj += foo[0] + self.param.fun_obj(ff, self.fv1[kk], self.param.KparDist) / (self.param.sigmaError**2)
                self.xtAll[kk] = np.copy(foo[1])

        return self.obj

    def getVariable(self):
        return self.fvTmpl


    def copyDir(self, dir):
        dfoo = Direction()
        dfoo.prior = np.copy(dir.prior)
        for d in dir.all:
            dfoo.all.append(smatch.Direction())
            dfoo.all[-1].diff = np.copy(d.diff)
            dfoo.all[-1].aff = np.copy(d.aff)
        return(dfoo)

    def randomDir(self):
        dfoo = Direction()
        dfoo.prior = np.random.randn(self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1])
        dfoo.all = []
        for k in range(self.Ntarg):
            dfoo.all.append(smatch.Direction())
            dfoo.all[k].diff = np.random.randn(self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1])
            dfoo.all[k].aff = np.random.randn(self.Tsize, self.affineDim)
        return(dfoo)

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.prior
        foo = self.objectiveFunDef(atTry, self.Afft, withTrajectory=True)
        objTry += foo[0]*self.lambdaPrior
        x0 = np.squeeze(foo[1][-1])
        #print -1, objTry - self.obj0
        atAllTry = []
        AfftAllTry = []
        ff = surfaces.Surface(surf=self.fv0)
        for (kk, d) in enumerate(dir.all):
            atAllTry.append(self.atAll[kk] - eps * d.diff)
            AfftAllTry.append(self.AfftAll[kk] - eps * d.aff)
            foo = self.objectiveFunDef(atAllTry[kk], AfftAllTry[kk], withTrajectory=True, x0 = x0)
            ff.updateVertices(np.squeeze(foo[1][-1]))
            objTry += foo[0] + self.param.fun_obj(ff, self.fv1[kk], self.param.KparDist) / (self.param.sigmaError**2)

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.atAllTry = atAllTry
            self.AfftAllTry = AfftAllTry
            self.objTry = objTry

        return objTry


    def gradientComponent(self, q, kk):
        px1 = - self.param.fun_objGrad(self.fvDef[kk], self.fv1[kk], self.param.KparDist) / self.param.sigmaError**2
        #print 'in fun', kk
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.AfftAll[kk][t])
                #print self.dim, dim2, AB.shape, self.affineBasis.shape
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        foo = evol.landmarkHamiltonianGradient(self.fvTmpl.vertices, self.atAll[kk], px1, self.param.KparDiff, self.regweight, getCovector = True, affine=A)
        #print foo[0].shape, foo[1].shape
        grd = foo[0:3]
        pxTmpl = foo[-1][0, :, :]
        #print 'before put', kk
        q.put([kk, grd, pxTmpl])
        #print 'end fun', kk

    def getGradient(self, coeff=1.0):
        #grd0 = np.zeros(self.atAll.shape)
        #pxIncr = np.zeros([self.Ntarg, self.atAll.shape[1], self.atAll.shape[2]])
        pxTmpl = np.zeros(self.at.shape[1:3])
        q = mp.Queue()
        #procGrd = []
        # for kk in range(Ntarg):
        #     procGrd.append(mp.Process(target = gradientComponent, args=(q,kk,fvTmpl, fv1[kk], xt[kk], at[kk], KparDist, KparDiff, regWeight, sigmaError,)))
        # for kk in range(Ntarg):
        #     procGrd[kk].start()
        # for kk in range(Ntarg):
        #     procGrd[kk].join()
        # print 'all joined'
        grd = Direction()
        for kk in range(self.Ntarg):
            self.gradientComponent(q, kk)
            grd.all.append(smatch.Direction())

        dim2 = self.dim**2
        for kk in range(self.Ntarg):
            foo = q.get()
            dat = foo[1][0]/(coeff*self.Tsize)
            dAfft = np.zeros(self.AfftAll[kk].shape)
            if self.affineDim > 0:
                dA = foo[1][1]
                db = foo[1][2]
                dAfft = 2 * self.AfftAll[kk]
                for t in range(self.Tsize):
                    dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
                    dAfft[t] -=  np.divide(dAff.reshape(dAfft[t].shape), self.affineWeight.reshape(dAfft[t].shape))
                dAfft /= (coeff*self.Tsize)
                #print dat.shape
            grd.all[foo[0]].diff = dat
            grd.all[foo[0]].aff = dAfft
            #print foo[0], grd[foo[0]][1].shape
            pxTmpl += foo[2]

        #print pxTmpl.shape
        foo2 = evol.landmarkHamiltonianGradient(self.fv0.vertices, self.at, pxTmpl/self.lambdaPrior, self.param.KparDiff, self.regweight)
        # xtPrior = np.copy(foo2[1])
        grd.prior = foo2[0] / (self.tmplCoeff*coeff*self.Tsize)
        return grd

    def dotProduct(self, g1, g2):
        grd2 = 0
        grd12 = 0
        res = np.zeros(len(g2))
        for (kk, gg1) in enumerate(g1.all):
            for t in range(self.Tsize):
                #print gg1[0].shape, gg1[1].shape
                z = np.squeeze(self.xtAll[kk][t, :, :])
                gg = np.squeeze(gg1.diff[t, :, :])
                u = self.param.KparDiff.applyK(z, gg)
                uu = np.multiply(gg1.aff[t], self.affineWeight.reshape(gg1.aff[t].shape))
                #u = rzz*gg
                ll = 0
                for gr in g2:
                    ggOld = np.squeeze(gr.all[kk].diff[t, :, :])
                    res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                    if self.affineDim > 0:
                        res[ll] += np.multiply(uu, gr.all[kk].aff[t]).sum()
                    ll = ll + 1

        for t in range(g1.prior.shape[0]):
            z = np.squeeze(self.xt[t, :, :])
            #rzz = kfun.kernelMatrix(self.param.KparDiff, z)
            gg = np.squeeze(g1.prior[t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            #u = rzz*gg
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.prior[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()*(self.tmplCoeff*self.lambdaPrior)
                ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        for kk,a in enumerate(self.atAllTry):
            self.atAll[kk] = np.copy(a)
            self.AfftAll[kk] = np.copy(self.AfftAllTry[kk])

    def addProd(self, dir1, dir2, coeff):
        res = Direction()
        res.prior = dir1.prior + coeff * dir2.prior
        for kk, dd in enumerate(dir1.all):
            res.all.append(smatch.Direction())
            res.all[kk].diff = dd.diff + coeff*dir2.all[kk].diff
            res.all[kk].aff = dd.aff + coeff*dir2.all[kk].aff
        return res


    def endOfIteration(self):
        (obj1, self.xt, Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True, withJacobian=True)
        self.fvTmpl.updateVertices(np.squeeze(self.xt[-1, :, :]))
        self.fvTmpl.saveVTK(self.outputDir +'/'+ 'Template.vtk', scalars = Jt[-1], scal_name='Jacobian')
        for kk in range(self.Ntarg):
            (obj1, self.xtAll[kk], Jt) = self.objectiveFunDef(self.atAll[kk], self.AfftAll[kk],
                                                              withTrajectory=True, x0 = np.squeeze(self.xt[-1]), withJacobian=True)
            self.fvDef[kk].updateVertices(self.xtAll[kk][-1])
            self.fvDef[kk].saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[-1, :], scal_name='Jacobian')

    def computeTemplate(self):
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.1, np.sqrt(grd2) / 10000)
        cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=True)
        return self

