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


## Main class for surface template estimation
#        Template: surface class (from surface.py); if not specified, opens fileTemplate
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
class SurfaceTimeSeries(smatch.SurfaceMatching):
    def __init__(self, Template=None, Targets=None, fileTemplate=None, fileTarg=None, param=None, maxIter=1000, verb=True,
                 regWeight = 1.0, affineWeight = 1.0, rotWeight = None, scaleWeight = None, transWeight = None,
                 testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if Template==None:
            if fileTemplate==None:
                print 'A template surface is needed'
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTemplate)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
        if Targets==None:
            if fileTarg==None:
                print 'Target surfaces are needed'
                return
            else:
                self.fv1=[]
                for ff in fileTarg:
                    self.fv1.append(surfaces.Surface(filename=ff))
        else:
            self.fv1=[]
            for ff in Targets:
                self.fv1.append(surfaces.Surface(surf=ff))


            #self.fvTmpl = byufun.ByuSurf(self.fv0)
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
            self.param = smatch.SurfaceMatchingParam()
        else:
            self.param = param

        self.Tsize = int(round(1.0/self.param.timeStep))
        # for ff in self.fv1:
        #     y0 = ff.vertices
        #     xDef1 = y0[ff.faces[:, 0], :]
        #     xDef2 = y0[ff.faces[:, 1], :]
        #     xDef3 = y0[ff.faces[:, 2], :]
        #     ff.centers = (xDef1 + xDef2 + xDef3) / 3
        #     ff.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)
        #     #self.at.append(np.zeros([Tsize, x0.shape[0], x0.shape[1]]))

        self.at = []
        self.xt = []
        self.fvDef = []
        self.atTry = []
        self.Afft = []
        self.AfftTry = []

        for ff in self.fv1:
            self.at.append(np.zeros([self.Tsize, self.npt, self.dim]))
            self.atTry.append(np.zeros([self.Tsize, self.npt, self.dim]))
            self.Afft.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftTry.append(np.zeros([self.Tsize, self.affineDim]))
            self.xt.append(np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1]))
            self.fvDef.append(surfaces.Surface(surf=self.fv0))

        self.Ntarg = len(self.fv1)
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]


    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0
            for fv in self.fv1:
                self.obj0 += self.param.fun_obj0(fv, self.param.KparDist) / (self.param.sigmaError**2)

            self.obj = self.obj0
            kk=0 ;
            x0 = self.fv0.vertices
            for (kk, a) in enumerate(self.at):
                foo = self.objectiveFunDef(a, self.Afft[kk], withTrajectory=True, x0 = x0)
                x0 = np.squeeze(foo[1][-1])
                self.fvDef[kk].updateVertices(x0)
                self.obj += foo[0] + self.param.fun_obj(self.fvDef[kk], self.fv1[kk], self.param.KparDist) / (self.param.sigmaError**2)
                self.xt[kk] = np.copy(foo[1])

        return self.obj

    def getVariable(self):
        return self

    def copyDir(self, dir):
        dir2 = []
        for d in dir:
            dir2.append(smatch.Direction())
            dir2[-1].diff = np.copy(d.diff)
            dir2[-1].aff = np.copy(d.aff)
        return(dir2)

    def randomDir(self):
        dir2 = []
        for k in range(self.Ntarg):
            dir2.append(smatch.Direction())
            dir2[k].diff = np.random.randn(self.Tsize, self.npt, self.dim)
            dir2[k].aff = np.random.randn(self.Tsize, self.affineDim)
        return(dir2)

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        kk=0 ;
        atTry = []
        AfftTry = []
        x0 = self.fv0.vertices
        ff = surfaces.Surface(surf=self.fv0)
        for (kk,d) in enumerate(dir):
            #m.fv0.updateVertices(x0)
            atTry.append(self.at[kk] - eps * d.diff)
            AfftTry.append(self.Afft[kk] - eps * d.aff)
            foo = self.objectiveFunDef(atTry[kk], AfftTry[kk], withTrajectory = True, x0=x0)
            x0 = np.squeeze(foo[1][-1])
            ff.updateVertices(x0)
            objTry += foo[0] + self.param.fun_obj(ff, self.fv1[kk], self.param.KparDist) / (self.param.sigmaError**2)
            #print kk, objTry - self.obj0

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry

        return objTry


    def getGradient(self, coeff=1.0):
        px0 = np.zeros(self.fv0.vertices.shape)
        grd = []
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        for kk in range(self.Ntarg):
            k0 = self.Ntarg - 1 - kk
            px1 = px0 - self.param.fun_objGrad(self.fvDef[k0], self.fv1[k0], self.param.KparDist) / self.param.sigmaError**2
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.AfftAll[k0][t])
                    #print self.dim, dim2, AB.shape, self.affineBasis.shape
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
                    #print 'in fun', k0
            foo = evol.landmarkHamiltonianGradient(self.xt[k0][0], self.at[k0], px1, self.param.KparDiff, self.regweight,
                                                   getCovector = True, affine=A)
            Dir = smatch.Direction()
            Dir.diff = foo[0]/(coeff*self.Tsize)
            dAfft = np.zeros(self.Afft[k0].shape)
            if self.affineDim > 0:
                dA = foo[1]
                db = foo[2]
                dAfft = 2 * self.Afft[k0]
                for t in range(self.Tsize):
                    dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
                    dAfft[t] -=  np.divide(dAff.reshape(dAfft[t].shape), self.affineWeight.reshape(dAfft[t].shape))
                dAfft /= (coeff*self.Tsize)
            Dir.aff = dAfft
            grd.insert(0, Dir)
            px0 = np.copy(np.squeeze(foo[4][0]))

        return grd

    def dotProduct(self, g1, g2):
        grd2 = 0
        grd12 = 0
        res = np.zeros(len(g2))
        for (kk, gg1) in enumerate(g1):
            for t in range(self.Tsize):
                z = np.squeeze(self.xt[kk][t, :, :])
                #rzz = kfun.kernelMatrix(self.param.KparDiff, z)
                gg = np.squeeze(gg1.diff[t, :, :])
                #u = rzz*gg
                u = self.param.KparDiff.applyK(z, gg)
                uu = np.multiply(gg1.aff[t], self.affineWeight.reshape(gg1.aff[t].shape))
                ll = 0
                for gr in g2:
                    ggOld = np.squeeze(gr[kk].diff[t, :, :])
                    res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                    if self.affineDim > 0:
                        res[ll] += np.multiply(uu, gr[kk].aff[t]).sum()
                    ll = ll + 1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        for kk,a in enumerate(self.atTry):
            self.at[kk] = np.copy(a)
            self.Afft[kk] = np.copy(self.AfftTry[kk])

    def addProd(self, dir1, dir2, coeff):
        res = []
        for kk, dd in enumerate(dir1):
            res.append(smatch.Direction())
            res[kk].diff = dd.diff + coeff*dir2[kk].diff
            res[kk].aff = dd.aff + coeff*dir2[kk].aff

        return res


    def endOfIteration(self):
        x0 = self.fv0.vertices
        t0 = 0
        for kk in range(self.Ntarg):
            (obj1, self.xt[kk], Jt) = self.objectiveFunDef(self.at[kk], self.Afft[kk],
                                                           withTrajectory=True, x0 = x0, withJacobian=True)
            x0 = self.xt[kk][-1]
            for t in range(self.Tsize):
                self.fvDef[kk].updateVertices(self.xt[kk][t])
                self.fvDef[kk].saveVTK(self.outputDir +'/'+ self.saveFile+str(t0)+'.vtk', scalars = Jt[t, :], scal_name='Jacobian')
                t0 += 1
            self.fvDef[kk].updateVertices(x0)

        self.fvDef[-1].saveVTK(self.outputDir +'/'+ self.saveFile+str(t0)+'.vtk', scalars = Jt[-1, :], scal_name='Jacobian')


    def computeMatching(self):
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.1, np.sqrt(grd2) / 10000)

        cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=True)
        return self

