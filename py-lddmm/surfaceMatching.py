import os
import numpy as np
import numpy.linalg as la
import scipy as sp
import logging
import surfaces
import kernelFunctions as kfun
import pointEvolution as evol
import conjugateGradient as cg
from affineBasis import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam:
    def __init__(self, timeStep = .1, KparDiff = None, KparDist =
                 None, sigmaKernel = 6.5, sigmaDist=2.5,
                 sigmaError=1.0, errorType = 'measure',  typeKernel='gauss'):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.sigmaDist = sigmaDist
        self.sigmaError = sigmaError
        self.typeKernel = typeKernel
        self.errorType = errorType
        if errorType == 'current':
            self.fun_obj0 = surfaces.currentNorm0
            self.fun_obj = surfaces.currentNormDef
            self.fun_objGrad = surfaces.currentNormGradient
        elif errorType=='measure':
            self.fun_obj0 = surfaces.measureNorm0
            self.fun_obj = surfaces.measureNormDef
            self.fun_objGrad = surfaces.measureNormGradient
        elif errorType=='varifold':
            self.fun_obj0 = surfaces.varifoldNorm0
            self.fun_obj = surfaces.varifoldNormDef
            self.fun_objGrad = surfaces.varifoldNormGradient
        else:
            logging.error('Unknown error Type: ' + self.errorType)
        if KparDiff == None:
            self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        else:
            self.KparDiff = KparDiff
        if KparDist == None:
            self.KparDist = kfun.Kernel(name = 'gauss', sigma = self.sigmaDist)
        else:
            self.KparDist = KparDist

class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []
        self.initx = []


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
class SurfaceMatching(object):

    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, verb=True,
                 subsampleTargetSize=-1,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                logging.error('Please provide a template surface')
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
        if Target==None:
            if fileTarg==None:
                logging.error('Please provide a target surface')
                return
            else:
                self.fv1 = surfaces.Surface(filename=fileTarg)
        else:
            self.fv1 = surfaces.Surface(surf=Target)

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()

        self.saveRate = 10
        self.iter = 0
        self.setOutputDir(outputDir)
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.symmetric = symmetric
        self.testGradient = testGradient
        self.regweight = regWeight
        self.affine = affine
        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(self.affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[self.affB.transComp] = transWeight

        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        v0 = self.fv0.surfVolume()
        v1 = self.fv1.surfVolume()
        if (v0*v1 < 0):
            self.fv1.flipFaces()
        #self.fv0Fine = surfaces.Surface(surf=self.fv0)
        self.fvInit = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fvInit.Simplify(subsampleTargetSize)
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        self.x0 = np.copy(self.fvInit.vertices)
        self.x0try = np.copy(self.fvInit.vertices)
        self.fvDef = surfaces.Surface(surf=self.fvInit)
        self.npt = self.fvInit.vertices.shape[0]

        z= self.fvInit.surfVolume()
        if (z < 0):
            self.fv0ori = -1
        else:
            self.fv0ori = 1

        z= self.fv1.surfVolume()
        if (z < 0):
            self.fv1ori = -1
        else:
            self.fv1ori = 1

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.x0.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.coeffAff1 = 1.
        self.coeffAff2 = 100.
        self.coeffAff = self.coeffAff1
        self.coeffInitx = .1
        self.affBurnIn = 20

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef, _fvInit = None):
        obj = self.param.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        if _fvInit != None:
            obj += self.param.fun_obj(_fvInit, self.fv0, self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian=False, x0 = None):
        if x0 == None:
            x0 = self.x0
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
            (xt,Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, param.KparDiff, affine=A, withJacobian=True)
        else:
            xt  = evol.landmarkDirectEvolutionEuler(x0, at, param.KparDiff, affine=A)
        #print xt[-1, :, :]
        #print obj
        obj=0
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            ra = param.KparDiff.applyK(z, a)
            self.v[t, :] = ra
            obj = obj + self.regweight*timeStep*np.multiply(a, (ra)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, Jt
        elif withTrajectory:
            return obj, xt
        else:
            return obj

    # def  _objectiveFun(self, at, Afft, withTrajectory = False):
    #     (obj, xt) = self.objectiveFunDef(at, Afft, withTrajectory=True)
    #     self.fvDef.updateVertices(np.squeeze(xt[-1, :, :]))
    #     obj0 = self.dataTerm(self.fvDef)

    #     if withTrajectory:
    #         return obj+obj0, xt
    #     else:
    #         return obj+obj0

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            if self.symmetric:
                self.obj0 += self.param.fun_obj0(self.fv0, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            foo = surfaces.Surface(surf=self.fvDef)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            if self.symmetric:
                self.fvInit.updateVertices(np.squeeze(self.x0))
            foo.computeCentersAreas()
            if self.symmetric:
                self.obj += self.obj0 + self.dataTerm(self.fvDef, self.fvInit)
            else:
                self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft, self.x0]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        if self.symmetric:
            x0Try = self.x0 - eps * dir.initx
        else:
            x0Try = self.x0

        foo = self.objectiveFunDef(atTry, AfftTry, x0 = x0Try, withTrajectory=True)
        objTry += foo[0]

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        if self.symmetric:
            ffI = surfaces.Surface(surf=self.fvInit)
            ffI.updateVertices(x0Try)
            objTry += self.dataTerm(ff, ffI)
        else:
            objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry
            self.x0Try = x0Try
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



    def endPointGradient(self):
        px = self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist)
        return px / self.param.sigmaError**2

    def initPointGradient(self):
        px = self.param.fun_objGrad(self.fvInit, self.fv0, self.param.KparDist)
        return px / self.param.sigmaError**2

    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        foo = evol.landmarkHamiltonianGradient(self.x0, self.at, px1, self.param.KparDiff, self.regweight, affine=A, getCovector=True)
        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff /= (self.coeffAff*coeff*self.Tsize)
            #            dAfft[:,0:self.dim**2]/=100
        if self.symmetric:
            grd.initx = (self.initPointGradient() - foo[4][0,...])/(self.coeffInitx * coeff)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        dir.initx = dir1.initx + beta * dir2.initx
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)
        dir.aff = np.copy(dir0.aff)
        dir.initx = np.copy(dir0.initx)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        dirfoo.initx = np.random.randn(self.npt, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            #uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            uu = g1.aff[t]
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    #print np.multiply(np.multiply(g1[1][t], gr[1][t]), self.affineWeight).shape
                    #res[ll] += np.multiply(uu, gr.aff[t]).sum() * self.coeffAff
                    res[ll] += np.multiply(uu, gr.aff[t]).sum() * self.coeffAff
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1

        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        self.x0 = np.copy(self.x0Try)
        #print self.at

    def endOfIteration(self):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0) :
            logging.info('Saving surfaces...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            if self.affineDim <=0:
                xtEPDiff, atEPdiff = evol.landmarkEPDiff(self.at.shape[0], self.x0,
                                                         np.squeeze(self.at[0, :, :]), self.param.KparDiff)
                self.fvDef.updateVertices(np.squeeze(xtEPDiff[-1, :, :]))
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+'EPDiff.vtk')
                logging.info('EPDiff difference %f' %(np.fabs(self.xt[-1,:,:] - xtEPDiff[-1,:,:]).sum()) )

            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            (xt, Jt)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                              withJacobian=True)
            if self.affine=='euclidean' or self.affine=='translation':
                f = surfaces.Surface(surf=self.fvInit)
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.x0.shape[0])
                dt = 1.0 /self.Tsize ;
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yyt = np.dot(self.xt[t,...] - X[1][t, ...], U.T)
                    zt = np.dot(xt[t,...] - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        at = np.dot(self.at[t,...], U.T)
                        vt = self.param.KparDiff.applyK(yyt, at, firstVar=zt)
                    f.updateVertices(yyt)
                    vf = surfaces.vtkFields() ;
                    vf.scalars.append('Jacobian') ;
                    vf.scalars.append(np.exp(Jt[t, :])-1)
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity') ;
                    vf.vectors.append(vt)
                    nu = self.fv0ori*f.computeVertexNormals()
                    displ += dt * (vt*nu).sum(axis=1)
                    f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk', vf)
                f = surfaces.Surface(surf=self.fv1)
                yyt = np.dot(f.vertices - X[1][-1, ...], U.T)
                f.updateVertices(yyt)
                f.saveVTK(self.outputDir +'/TargetCorrected.vtk')
            fvDef = surfaces.Surface(surf=self.fvInit)
            AV0 = fvDef.computeVertexArea()
            nu = self.fv0ori*self.fvInit.computeVertexNormals()
            v = self.v[0,...]
            displ = np.zeros(self.npt)
            dt = 1.0 /self.Tsize ;
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(xt[kk, :, :]))
                AV = fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])-1
                vf = surfaces.vtkFields() ;
                vf.scalars.append('Jacobian') ;
                vf.scalars.append(np.exp(Jt[kk, :])-1)
                vf.scalars.append('Jacobian_T') ;
                vf.scalars.append(AV[:,0])
                vf.scalars.append('Jacobian_N') ;
                vf.scalars.append(np.exp(Jt[kk, :])/(AV[:,0]+1)-1)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                displ += dt * (v*nu).sum(axis=1)
                if kk < self.Tsize:
                    nu = self.fv0ori*self.fvDef.computeVertexNormals()
                    v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity') ;
                vf.vectors.append(self.v[kkm,:])
                fvDef.saveVTK2(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', vf)
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
        else:
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        logging.info('Gradient lower bound: %f' %(self.gradEps))
        self.coeffAff = self.coeffAff1
        cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt

