import os
import numpy as np
import scipy as sp
import surfaces
from pointSets import read3DVector
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
        else:
            print 'Unknown error Type: ', self.errorType
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
class SurfaceMatching:

    def __init__(self, Template=None, Target=None, Fiber=None, fileTempl=None, fileTarg=None, fileFiber=None, param=None, maxIter=1000, regWeight = 1.0, verb=True,
                 subsampleTargetSize=-1, testGradient=True, saveFile = 'evolution', outputDir = '.'):
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

        if Fiber==None:
            if fileFiber==None:
                print 'Please provide fiber structure'
                return
            else:
                (self.y0, self.v0) = read3DVectorField(fileFiber)
        else:
            self.y0 = np.copy(Fiber[0])
            self.v0 = np.copy(Fiber[1])

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()

        self.saveRate = 10
        self.iter = 0
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight

        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        self.fv0Fine = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fv0.Simplify(subsampleTargetSize)
            v0 = self.fv0.surfVolume()
            v1 = self.fv1.surfVolume()
            if (v0*v1 < 0):
                self.fv1.flipFaces()
            print 'simplified template', self.fv0.vertices.shape[0]
        self.x0 = self.fv0.vertices
        self.fvDef = surfaces.Surface(surf=self.fv0)
        self.npt = self.y0.shape[0]
        self.a0 = np.zeros([self.y0.shape[0], self.x0.shape[1]])

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.rhot = np.zeros([self.Tsize, self.y0.shape[0]])
        self.rhotTry = np.zeros([self.Tsize, self.y0.shape[0]])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.at = np.tile(self.a0, [self.Tsize+1, 1, 1])
        self.yt = np.tile(self.y0, [self.Tsize+1, 1, 1])
        self.vt = np.tile(self.v0, [self.Tsize+1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef):
        obj = self.param.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, rhot, withTrajectory = False, withJacobian=False, Init = None):
        if Init == None:
            x0 = self.x0
            y0 = self.y0
            v0 = self.v0
            a0 = self.a0
        else:
            x0 = Init[0]
            y0 = Init[2]
            v0 = Init[3]
            a0 = Init[1]
            
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        #print a0.shape
        if withJacobian:
            (xt, at, yt, vt, Jt)  = evol.secondOrderFiberEvolution(x0, a0, y0, v0, rhot, param.KparDiff, withJacobian=True)
        else:
            (xt, at, yt, vt)  = evol.secondOrderFiberEvolution(x0, a0, y0, v0, rhot, param.KparDiff)
        #print xt[-1, :, :]
        #print obj
        obj=0
        for t in range(self.Tsize):
            v = np.squeeze(vt[t, :, :])
            rho = np.squeeze(rhot[t, :])
            
            obj = obj + ((rho[:,np.newaxis]**2) * (v**2).sum(axis=1)).sum()/2
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, at, yt, vt, Jt
        elif withTrajectory:
            return obj, xt, at, yt, vt
        else:
            return obj

    def  _objectiveFun(self, rhot, withTrajectory = False):
        (obj, xt, at, yt, vt) = self.objectiveFunDef(rhot, withTrajectory=True)
        self.fvDef.updateVertices(np.squeeze(xt[-1, :, :]))
        obj0 = self.dataTerm(self.fvDef)

        if withTrajectory:
            return obj+obj0, xt, at, yt, vt
        else:
            return obj+obj0

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt, self.at, self.yt, self.vt) = self.objectiveFunDef(self.rhot, withTrajectory=True)
            foo = surfaces.Surface(surf=self.fvDef)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)

        return self.obj

    def getVariable(self):
        return self.rhot

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        rhotTry = self.rhot - eps * dir.diff
        foo = self.objectiveFunDef(rhotTry, withTrajectory=True)
        objTry += foo[0]

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.rhotTry = rhotTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



    def endPointGradient(self):
        px = self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist)
        return px / self.param.sigmaError**2


    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        pa1 = np.zeros(self.a0.shape)
        py1 = np.zeros(self.v0.shape)
        pv1 = np.zeros(self.v0.shape)
        foo = evol.secondOrderFiberGradient(self.x0, self.a0, self.y0, self.v0, self.rhot, px1, pa1, py1, pv1, self.param.KparDiff)
        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)

        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        gg = g1.diff
        ll = 0
        for gr in g2:
            ggOld = gr.diff
            res[ll]  = (ggOld*gg).sum()
            ll = ll+1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.rhot = np.copy(self.rhotTry)
        #print self.at

    def endOfIteration(self):
        self.iter += 1
        if (self.iter % self.saveRate == 0) :
            (obj1, self.xt, self.at, self.yt, self.vt) = self.objectiveFunDef(self.rhot, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            (xt, at, yt, vt, zt, Jt)  = evol.secondOrderFiberEvolution(self.x0, self.a0, self.y0, self.v0,  self.rhot, self.param.KparDiff,
                                                           withPointSet = self.fv0Fine.vertices, withJacobian=True)
            fvDef = surfaces.Surface(surf=self.fv0Fine)
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(zt[kk, :, :]))
                fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
        else:
            (obj1, self.xt, self.at, self.yt, self.vt) = self.objectiveFunDef(self.rhot, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        print 'Gradient lower bound:', self.gradEps
        cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt

