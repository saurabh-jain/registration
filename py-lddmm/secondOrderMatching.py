import os
import numpy as np
import scipy as sp
import surfaces
from pointSets import *
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

class Direction:
    def __init__(self):
        self.a0 = []
        self.rhot = []


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
    def __init__(self, Template=None, Targets=None, fileTempl=None, fileTarg=None, param=None, initialMomentum=None,
                 maxIter=1000, regWeight = 1.0, verb=True, typeRegression='spline2',
                 subsampleTargetSize=-1, testGradient=True, saveFile = 'evolution', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                print 'Please provide a template surface'
                return
            else:
                self.fv0 = surfaces.Surface(filename=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
        if Targets==None:
            if fileTarg==None:
                print 'Please provide a list of target surfaces'
                return
            else:
                self.fv1 = [] ;
                for f in fileTarg:
                    self.fv1.append(surfaces.Surface(filename=f))
        else:
            self.fv1 = [] ;
            for s in Targets:
                self.fv1.append(surfaces.Surface(surf=s))


        self.nTarg = len(self.fv1)
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
        self.typeRegression = typeRegression 

        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        self.fv0Fine = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fv0.Simplify(subsampleTargetSize)
            v0 = self.fv0.surfVolume()
            for s in self.fv1:
                v1 = s.surfVolume()
                if (v0*v1 < 0):
                    s.flipFaces()
            print 'simplified template', self.fv0.vertices.shape[0]
        self.x0 = self.fv0.vertices
        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.npt = self.x0.shape[0]
        self.Tsize1 = int(round(1.0/self.param.timeStep))
        self.Tsize = self.nTarg*self.Tsize1
        self.rhot = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if initialMomentum==None:
            self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.a0 = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.at = np.tile(self.a0, [self.Tsize+1, 1, 1])
        else:
            self.a0 = initialMomentum
            (self.xt, self.at)  = evol.secondOrderEvolution(self.x0, self.a0, self.rhot, self.param.KparDiff)

        self.rhotTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.a0Try = np.zeros([self.x0.shape[0], self.x0.shape[1]])

        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        for k,s in enumerate(self.fv1):
            s.saveVTK(self.outputDir+'/Target'+str(k)+'.vtk')

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef):
        obj = 0
        for k,s in enumerate(_fvDef):
            obj += self.param.fun_obj(s, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, a0, rhot, withTrajectory = False, withJacobian=False, Init = None):
        if Init == None:
            x0 = self.x0
        else:
            x0 = Init[0]
            
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        #print a0.shape
        if withJacobian:
            (xt, at, Jt)  = evol.secondOrderEvolution(x0, a0, rhot, param.KparDiff, withJacobian=True)
        else:
            (xt, at)  = evol.secondOrderEvolution(x0, a0, rhot, param.KparDiff)
        #print xt[-1, :, :]
        #print obj
        obj = 0.5 * (a0 * param.KparDiff.applyK(x0,a0)).sum()
        for t in range(self.Tsize):
            rho = np.squeeze(rhot[t, :, :])            
            obj = obj + timeStep* (rho**2).sum()/2
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, at, Jt
        elif withTrajectory:
            return obj, xt, at
        else:
            return obj

    def  _objectiveFun(self, a0, rhot, withTrajectory = False):
        (obj, xt, at) = self.objectiveFunDef(a0, rhot, withTrajectory=True)
        for k in range(self.nTarg):
            self.fvDef[k].updateVertices(np.squeeze(xt[(k+1)*self.Tsize1, :, :]))
        obj0 = self.dataTerm(self.fvDef)

        if withTrajectory:
            return obj+obj0, xt, at
        else:
            return obj+obj0

    def objectiveFun(self):
        if self.obj == None:
            (self.obj, self.xt, self.at) = self.objectiveFunDef(self.a0, self.rhot, withTrajectory=True)
            self.obj0 = 0
            for k in range(self.nTarg):
                self.obj0 += self.param.fun_obj0(self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                foo = surfaces.Surface(surf=self.fvDef[k])
                self.fvDef[k].updateVertices(np.squeeze(self.xt[(k+1)*self.Tsize1, :, :]))
                foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return (self.a0, self.rhot)
    
    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        a0Try = self.a0 - eps * dir.a0
        rhotTry = self.rhot - eps * dir.rhot
        foo = self.objectiveFunDef(a0Try, rhotTry, withTrajectory=True)
        objTry += foo[0]

        ff = [] 
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(foo[1][(k+1)*self.Tsize1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.a0Try = a0Try
            self.rhotTry = rhotTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



    def endPointGradient(self):
        px = []
        for k in range(self.nTarg):
            px.append(-self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist)/ self.param.sigmaError**2)
        return px 


    def getGradient(self, coeff=1.0):
        px1 = self.endPointGradient()
        pa1 = []
        for k in range(self.nTarg):
            pa1.append(np.zeros(self.a0.shape))
            
        foo = evol.secondOrderGradient(self.x0, self.a0, self.rhot, px1, pa1,
                                        self.param.KparDiff, times = (1+np.array(range(self.nTarg)))*self.Tsize1)
        grd = Direction()
        if self.typeRegression == 'spline':
            grd.a0 = np.zeros(foo[0].shape)
            grd.rhot = foo[1]/(coeff)
            #grd.rhot = foo[1]/(coeff*self.rhot.shape[0])
        elif self.typeRegression == 'geodesic':
            grd.a0 = foo[0] / coeff
            grd.rhot = np.zeros(foo[1].shape)
        else:
            grd.a0 = foo[0] / coeff
            grd.rhot = foo[1]/(coeff)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.a0 = dir1.a0 + beta * dir2.a0
        dir.rhot = dir1.rhot + beta * dir2.rhot
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.a0 = np.copy(dir0.a0)
        dir.rhot = np.copy(dir0.rhot)

        return dir


    def randomDir(self):
        dirfoo = Direction()
        if self.typeRegression == 'spline':
            dirfoo.a0 = np.zeros([self.npt, self.dim])
            dirfoo.rhot = np.random.randn(self.Tsize, self.npt, self.dim)
        elif self.typeRegression == 'geodesic':
            dirfoo.a0 = np.random.randn(self.npt, self.dim)
            dirfoo.rhot = np.zeros([self.Tsize, self.npt, self.dim])
        else:
            dirfoo.a0 = np.random.randn(self.npt, self.dim)
            dirfoo.rhot = np.random.randn(self.Tsize, self.npt, self.dim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        gg = g1.rhot
        #gga = KparDiff.applyK(self.x0, g1.a0)
        gga = g1.a0
        ll = 0
        for gr in g2:
            ggOld = gr.rhot
            res[ll]  = (ggOld*gg).sum()/self.Tsize
            res[ll] += (gr.a0 * gga).sum()
            ll = ll+1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.a0 = np.copy(self.a0Try)
        self.rhot = np.copy(self.rhotTry)
        #print self.at

    def endOfIteration(self):
        self.iter += 1
        if (self.iter % self.saveRate == 0):
            (obj1, self.xt, self.at) = self.objectiveFunDef(self.a0, self.rhot, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[(k+1)*self.Tsize1, :, :]))
            (xt, at, zt, Jt)  = evol.secondOrderEvolution(self.x0, self.a0,  self.rhot, self.param.KparDiff,
                                                           withPointSet = self.fv0Fine.vertices, withJacobian=True)
            fvDef = surfaces.Surface(surf=self.fv0Fine)
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(zt[kk, :, :]))
                fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
        else:
            (obj1, self.xt, self.at) = self.objectiveFunDef(self.a0, self.rhot, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[(k+1)*self.Tsize1, :, :]))


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        print 'Gradient lower bound:', self.gradEps
        #print 'x0:', self.x0
        #print 'y0:', self.y0
        
        cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt

