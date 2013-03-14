import os
import numpy as np
import scipy as sp
import surfaces
import kernelFunctions as kfun
import pointEvolution as evol
import conjugateGradient as cg
import surfaceMatching
from affineBasis import *


## Main class for surface matching
#        Template: sequence of surface classes (from surface.py); if not specified, opens files in fileTemp
#        Target: sequence of surface classes (from surface.py); if not specified, opens files in fileTarg
#        par: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        regWeightOut: multiplicative constant on background regularization
#        affineWeight: multiplicative constant on affine regularization
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        mu: initial value for quadratic penalty normalization
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        typeConstraint: 'stiched', 'sliding', 'slidingV2'
#        affine: 'affine', 'euclidean' or 'none'
#        maxIter_cg: max iterations in conjugate gradient
#        maxIter_al: max interation for augmented lagrangian

class SurfaceWithIsometries(surfaceMatching.SurfaceMatching):
    def __init__(self, Template=None, Target=None, Isometries = None, centerRadius = None, fileTempl=None, fileTarg=None, param = None, verb = True, regWeight=1.0, at = None,
                 affineWeight = 1.0, testGradient=False, mu = 0.1, outputDir='.', saveFile = 'evolution',
                 affine='none', rotWeight = None, scaleWeight = None, transWeight = None,  maxIter_cg=1000, maxIter_al=100):
        print 'Initializing class'
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

        if hasattr(self.fv0, 'edges') == False:
            self.fv0.getEdges()

        if Isometries == None:
            if centerRadius == None:
                print 'Isometries must be specified'
                return
            else:
                Isometries = []
                x0 = centerRadius[0:3]
                r = centerRadius[3]
                for edg in self.fv0.edges:
                    #print max( ((self.fv0.vertices[edg[0], :] - x0)**2).sum(), ((self.fv0.vertices[edg[1], :] - x0)**2).sum()) 
                    if (((self.fv0.vertices[edg[0], :] - x0)**2).sum() > r*r) | (((self.fv0.vertices[edg[1], :] - x0)**2).sum() > r*r):
                        Isometries.append([edg[0], edg[1]])
                


        self.c = []
        for u in Isometries:
            if (u in self.fv0.edges):
                self.c.append(u)
            else:
                if (u.reverse() in self.fv0.edges): 
                    self.c.append(u.reverse())

        self.c = np.array(self.c)
        self.nconstr = self.c.shape[0]
        print 'Number of constrained edges: ', self.c.shape[0], 'out of', len(self.fv0.edges)
        self.I0 = np.zeros([self.fv0.vertices.shape[0], self.c.shape[0]])
        #self.I1 = np.zeros(self.fv0.vertices.shape[0], self.c.shape[0])

        for k in range(self.c.shape[0]):
            self.I0[self.c[k,0]][k] = 1
            self.I0[self.c[k,1]][k] = -1
            #self.I0 = self.I0.nonzero()
            #self.I1 = self.I1.nonzero()

            #self.I0 = np.mat(self.I0)

        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        self.outputDir = outputDir  
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print 'Cannot save in ' + outputDir
                return
            else:
                os.mkdir(outputDir)

        self.fvDef = surfaces.Surface(self.fv0)
        self.maxIter_cg = maxIter_cg
        self.maxIter_al = maxIter_al
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        if param==None:
            self.param = surfaceMatching.SurfaceMatchingParam()
        else:
            self.param = param
        self.affine = affine
        affB = AffineBasis(self.dim, affine)
        self.affineDim = affB.affineDim
        self.affineBasis = affB.basis
        self.affineWeight = affineWeight * np.ones([1, self.affineDim])
        #print self.affineDim, affB.rotComp, rotWeight, self.affineWeight
        if (len(affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[0,affB.rotComp] = rotWeight
        if (len(affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[0,affB.simComp] = scaleWeight
        if (len(affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[0,affB.transComp] = transWeight

        self.x0 = self.fv0.vertices
        self.Tsize = int(round(1.0/self.param.timeStep))
        if at == None:
            self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        else:
            self.at = np.copy(at)
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.cval = np.zeros([self.Tsize+1, self.c.shape[0]])
        if self.nconstr > 0:
            self.ntau0 = np.sqrt(np.power(self.x0[self.c[:,0], :] - self.x0[self.c[:,1], :],2).sum(axis=1))
        #self.lmb = 10*np.multiply(np.random.randn(self.Tsize, 1), np.ones([self.Tsize, self.c.shape[0]]))
        self.lmb = np.ones([self.Tsize+1, self.c.shape[0]])
        self.mu = mu
        self.obj = None
        self.objTry = None
        self.saveFile = saveFile
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.color=np.ones(self.fv0.vertices.shape[0])
        u = np.fabs(self.I0).sum(axis=1)
        for kk in range(self.fv0.vertices.shape[0]):
            if u[kk] > 0:
                self.color[kk] = 2
        self.fv0.saveVTK(self.outputDir+'/Template.vtk',
        scalars=self.color, scal_name='constraints')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')


    def constraintTerm(self, xt):
        obj = 0
        timeStep = 1.0/self.Tsize
        c = self.c
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize+1):
            z = np.squeeze(xt[t, :, :]) 
            ntau = np.sqrt(np.power(z[c[:,0], :] - z[c[:,1], :],2).sum(axis=1))
            cval[t, :] = np.divide(ntau, self.ntau0) - 1
            obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() + np.multiply(cval[t, :], cval[t, :]).sum()/(2*self.mu))
        return obj, cval

    def constraintTermGrad(self, xt):
        c = self.c
        I0 = self.I0
        lmb = np.zeros(self.cval.shape)
        dxcval = np.zeros(self.xt.shape)

        for t in range(self.Tsize+1):
            z = np.squeeze(xt[t, :, :])
            tau = z[c[:,0], :] - z[c[:,1], :]
            ntau = np.sqrt(np.power(tau,2).sum(axis=1))
            lmb[t, :] = self.lmb[t, :] - (np.divide(ntau , self.ntau0)-1)/self.mu
            tau = np.divide(tau, (ntau*self.ntau0).reshape([self.nconstr, 1]))
            ltau = np.multiply(lmb[t, :].reshape([tau.shape[0], 1]), tau)
            dxcval[t, :, :] = np.dot(self.I0, ltau)
        return lmb, dxcval

    def testConstraintTerm(self, xt):
        xtTry = []
        eps = 0.00000001
        xtTry = xt + eps*np.random.randn(self.Tsize+1, self.npt, self.dim)

        u0 = self.constraintTerm(xt)
        ux = self.constraintTerm(xtTry)
        [l, dx] = self.constraintTermGrad(xt)
        vx = np.multiply(dx, xtTry-xt).sum()/eps
        print 'Testing constraints:'
        print 'var x:', self.Tsize*(ux[0]-u0[0])/(eps), -vx 

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian = False):
        param = self.param
        timeStep = 1.0/at.shape[0]
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t]) 
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        if withJacobian:
            xJ  = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, at, param.KparDiff, withJacobian = True, affine=A)
            xt = xJ[0]
            Jt = xJ[1]
        else:
            xt  = evol.landmarkDirectEvolutionEuler(self.fv0.vertices, at, param.KparDiff, affine=A)

        obj=0
        for t in range(at.shape[0]):
            z = np.squeeze(xt[t, :, :]) 
            a = np.squeeze(at[t, :, :]) 
            ra = self.param.KparDiff.applyK(z, a)
            obj += self.regweight*timeStep*np.multiply(a, ra).sum()
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()

        cstr = [[], []]
        if self.nconstr > 0:
            cstr = self.constraintTerm(xt)
            obj += cstr[0]
        
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, Jt, cstr[1]
        elif withTrajectory:
            return obj, xt, cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[self.Tsize, :, :]))
            self.obj += self.obj0 + self.param.fun_obj(self.fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.diff
        AfftTry = self.Afft - eps * dir.aff
        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry += foo[0]
        
        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        objTry +=  self.param.fun_obj(ff, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        #self.fvDef.vertices = ff

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            self.cval = foo[2]

        return objTry


    def covectorEvolution(self, at, Afft, px1):
        N = self.npt
        dim = self.dim
        M = self.Tsize
        timeStep = 1.0/M
        dim2 = self.dim**2

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t]) 
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)

        #### Restart here
        pxt = np.zeros([M, N, dim])
        if self.nconstr > 0:
            (lmb, dxcval) = self.constraintTermGrad(xt)
            pxt[M-1, :, :] = px1 + dxcval[M] *timeStep
        else:
            pxt[M-1, :, :] = px1
        # print c
        for t in range(M-1):
            px = np.squeeze(pxt[M-t-1, :, :])
            z = np.squeeze(xt[M-t-1, :, :])
            a = np.squeeze(at[M-t-1, :, :])
            if self.nconstr > 0:
                zpx = np.copy(dxcval[M-t-1])
            else:
                zpx = np.zeros(px1.shape)

            a1 = [px, a, -2*self.regweight*a]
            a2 = [a, px, a]
            zpx += self.param.KparDiff.applyDiffKT(z, a1, a2)
            if self.affineDim > 0:
                zpx += np.dot(px, A[0][M-t-1])
            pxt[M-t-2, :, :] = np.squeeze(pxt[M-t-1, :, :]) + timeStep * zpx

        return pxt, xt


    def HamiltonianGradient(self, at, Afft, px1, getCovector = False):
        (pxt, xt) = self.covectorEvolution(at, Afft, px1)
        dat = np.zeros(at.shape)
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t]) 
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        dA = np.zeros(A[0].shape)
        db = np.zeros(A[1].shape)
        for t in range(self.Tsize):
            a = np.squeeze(at[t, :, :])
            px = np.squeeze(pxt[t, :, :])
            dat[t, :, :] = (2*self.regweight*a-px)
            if not (self.affine == None):
                dA[t] = np.dot(pxt[t].T, xt[t])
                db[t] = pxt[t].sum(axis=0)

        if getCovector == False:
            return dat, dA, db, xt
        else:
            return dat, dA, db, xt, pxt
        

    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        foo = self.HamiltonianGradient(self.at, self.Afft, px1)
        grd = surfaceMatching.Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        dim2 = self.dim**2
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
        

    def endOfIteration(self):
        (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True, withJacobian=True)
        #self.testConstraintTerm(self.xt)
        if self.nconstr > 0:
            print 'mean constraint', np.sqrt((self.cval**2).sum()/self.cval.size), np.fabs(self.cval).sum() / self.cval.size
        a0, foo = self.fv0.computeVertexArea()
        for kk in range(self.Tsize+1):
            #print self.xt[kk, :, :]
            self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
            ak, foo = self.fvDef.computeVertexArea()
            JJ = np.log(np.maximum(1e-10, np.divide(ak,a0+1e-10)))
            #print ak.shape, a0.shape, JJ.shape
            self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = JJ.flatten(), scal_name='Jacobian')


    def optimizeMatching(self):
	grd = self.getGradient(self.gradCoeff)
	[grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 1000
        self.muEps = 1.0
        it = 0

        while (self.muEps > 0.005) & (it<self.maxIter_al)  :
            print 'Starting Minimization: gradEps = ', self.gradEps, ' muEps = ', self.muEps, ' mu = ', self.mu
            cg.cg(self, verb = self.verb, maxIter = self.maxIter_cg, TestGradient = self.testGradient, epsInit = 0.1)
            if self.nconstr == 0:
                break
            for t in range(self.lmb.shape[0]):
                self.lmb[t, :] -= self.cval[t, :]/self.mu
            print 'mean lambdas', np.fabs(self.lmb).sum() / self.lmb.size
            if self.converged:
                self.gradEps *= .75
                if (((self.cval**2).sum()/self.cval.size) > self.muEps**2):
                    self.mu *= 0.5
                else:
                    self.muEps = self.muEps /2
            else:
                self.mu *= 0.9
            self.obj = None
            it = it+1

            
        return self.fvDef

