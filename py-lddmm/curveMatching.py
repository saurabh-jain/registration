import os
import numpy as np
import scipy as sp
import matchingParam
import curves
import grid
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
class CurveMatchingParam(matchingParam.MatchingParam):
    def __init__(self, timeStep = .1, KparDiff = None, KparDist = None, sigmaKernel = 6.5, sigmaDist=2.5, sigmaError=1.0, errorType = 'measure', typeKernel='gauss'):
	matchingParam.MatchingParam.__init__(self, timeStep=timeStep, KparDiff = KparDiff, KparDist = KparDist, sigmaKernel = sigmaKernel, sigmaDist=sigmaDist,
					     sigmaError=sigmaError, errorType = errorType, typeKernel=typeKernel)
                                         
        if errorType == 'current':
            print 'Running Current Matching'
            self.fun_obj0 = curves.currentNorm0
            self.fun_obj = curves.currentNormDef
            self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            print 'Running Measure Matching'
            self.fun_obj0 = curves.measureNorm0
            self.fun_obj = curves.measureNormDef
            self.fun_objGrad = curves.measureNormGradient
        else:
            print 'Unknown error Type: ', self.errorType

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
class CurveMatching:

    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True,
                 gradLB = 0.001, saveRate=10,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                print 'Please provide a template surface'
                return
            else:
                self.fv0 = curves.Curve(filename=fileTempl)
        else:
            self.fv0 = curves.Curve(curve=Template)
        if Target==None:
            if fileTarg==None:
                print 'Please provide a target surface'
                return
            else:
                self.fv1 = curves.Curve(filename=fileTarg)
        else:
            self.fv1 = curves.Curve(curve=Target)



        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        if self.dim == 2:
            xmin = min(self.fv0.vertices[:,0].min(), self.fv1.vertices[:,0].min())
            xmax = max(self.fv0.vertices[:,0].max(), self.fv1.vertices[:,0].max())
            ymin = min(self.fv0.vertices[:,1].min(), self.fv1.vertices[:,1].min())
            ymax = max(self.fv0.vertices[:,1].max(), self.fv1.vertices[:,1].max())
            dx = 0.025*(xmax-xmin)
            dy = 0.025*(ymax-ymin)
            dxy = min(dx,dy)
            #print xmin,xmax, dxy
            [x,y] = np.mgrid[(xmin-10*dxy):(xmax+10*dxy):dxy, (ymin-10*dxy):(ymax+10*dxy):dxy]
            #print x.shape
            self.gridDef = grid.Grid(gridPoints=[x,y])
            self.gridxy = np.copy(self.gridDef.vertices)
            
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)
        self.fvDef = curves.Curve(curve=self.fv0)
        self.iter = 0 ;
        self.maxIter = maxIter
        self.verb = verb
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
        x0 = self.fv0.vertices

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, x0.shape[0], x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, x0.shape[0], x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.gradLB = gradLB
        self.saveRate = saveRate 

    def dataTerm(self, _fvDef):
        obj = self.param.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian=False, x0 = None):
        if x0 == None:
            x0 = self.fv0.vertices
        param = self.param
        timeStep = 1.0/self.Tsize
        #dim2 = self.dim**2
        A = self.affB.getTransforms(Afft)
        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
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

    def  _objectiveFun(self, at, Afft, withTrajectory = False):
        (obj, xt) = self.objectiveFunDef(at, Afft, withTrajectory=True)
        self.fvDef.updateVertices(np.squeeze(xt[-1, :, :]))
        obj0 = self.dataTerm(self.fvDef)

        if withTrajectory:
            return obj+obj0, xt
        else:
            return obj+obj0

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            print self.obj0, self.obj

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
        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry += foo[0]

        ff = curves.Curve(curve=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print 'Warning: nan in updateTry'
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry

        return objTry



    def endPointGradient(self):
        px = self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist)
        return px / self.param.sigmaError**2


    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        A = self.affB.getTransforms(self.Afft)
        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, self.Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]
        foo = evol.landmarkHamiltonianGradient(self.fv0.vertices, self.at, px1, self.param.KparDiff, self.regweight, affine=A)
        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            #dAfft = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
            grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
            grd.aff /= (coeff*self.Tsize)
            #            dAfft[:,0:self.dim**2]/=100
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
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    #print np.multiply(np.multiply(g1[1][t], gr[1][t]), self.affineWeight).shape
                    res[ll] += np.multiply(uu, gr.aff[t]).sum()
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)

    def endOfIteration(self):
        (obj1, self.xt, Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True, withJacobian=True)
        self.iter += 1

        if self.saveRate > 0 and self.iter%self.saveRate==0:
            if self.dim==2:
                A = self.affB.getTransforms(self.Afft)
                (xt,yt) = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, self.at, self.param.KparDiff, affine=A, withPointSet=self.gridxy)
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
                if self.dim == 2:
                    self.gridDef.vertices = np.copy(yt[kk, :, :])
                    self.gridDef.saveVTK(self.outputDir +'/grid'+str(kk)+'.vtk')
        else:
            self.fvDef.updateVertices(np.squeeze(self.xt[self.Tsize, :, :]))
                

    def endOptim(self):
        if self.saveRate==0 or self.iter%self.saveRate > 0:
            if self.dim==2:
                A = self.affB.getTransforms(self.Afft)
                (xt,yt) = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, self.at, self.param.KparDiff, affine=A, withPointSet=self.gridxy)
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
        self.defCost = self.obj - self.obj0 - self.dataTerm(self.fvDef)   


    def optimizeMatching(self):
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(self.gradLB, np.sqrt(grd2) / 10000)
        print 'Gradient bound:', self.gradEps
        kk = 0
        while os.path.isfile(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk'):
            os.remove(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
            kk += 1
        cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

