# -*- coding: utf-8 *-*
import numpy
import scipy.linalg.basic
import scipy.sparse.linalg
import os
import logging
import optparse
import shutil
import regularGrid
import smoothImageConfig
import pointEvolution
import conjugateGradient
import kernelFunctions as kfun
import loggingUtils

import kernelMatrix_fort
from PIL import Image

import pdb

class SplineInterp_SVD(object):
    """
    Implements the RKHS spline interpolation problem by SVD
    (used for small dimension problems)
    """
    def __init__(self, rg, K):
        self.rg = rg
        self.K = K
        mat = self.K.precompute(self.rg.nodes, diff=False)
        (U,D,VT) = scipy.linalg.svd(mat)
        self.U = U
        self.Di = numpy.diag(1/D)
        self.VT = VT
        #self.nm = numpy.sqrt(numpy.power(mat,2).sum())
        #logging.info("eigenvalue: %f" % self.nm)

    def solve(self, rhs):
        return numpy.dot(self.VT.T, numpy.dot(self.Di, \
                                numpy.dot(self.U.T, rhs)))

class SplineInterp(object):
    """
    Implements the RKHS spline interpolation problem using cg
    """
    def __init__(self, rg, khs, kho, h, verbose=False):
        self.rg = rg
        self.cg_iter = 0
        self.khs = khs
        self.kho = kho
        self.h = h 
        self.verbose_logging = verbose
        #mat = self.K.precompute(self.rg.nodes, diff=False)
        #self.nm = numpy.sqrt(numpy.power(mat,2).sum())
        #self.nm = 30.
        #logging.info("eigenvalue: %f" % self.nm)

    def solve_mult(self, a):
        rg = self.rg
        ka = kernelMatrix_fort.applyk( \
                    rg.nodes, rg.nodes, a,
                    self.khs, self.kho, self.rg.num_nodes)
        #ka = self.K.applyK(self.rg.nodes, a)
        #ka += (1./self.nm) * a
        return ka

    def solve_callback(self, sol_k):
        self.cg_iter += 1
        ka = self.solve_mult(sol_k)
        energy = .5 * numpy.dot(sol_k, ka) - \
                             numpy.dot(sol_k, self.h)

        if (self.verbose_logging):
            logging.info("cg iteration %d: energy %f" % (self.cg_iter, energy))
            rg = self.rg
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.h, "h")
            rg.add_vtk_point_data(sol_k, "sol_k")
            rg.add_vtk_point_data(ka, "ka")
            rg.vtk_write(self.cg_iter, "cg_test", output_dir=".")

    def minimize(self):
        N = len(self.h)
        rhs = self.h
        linop = scipy.sparse.linalg.LinearOperator((N,N), self.solve_mult, \
                            dtype=float)
        sol = scipy.sparse.linalg.cg(linop, rhs, tol=1e-30, \
                            callback=self.solve_callback, maxiter=5000)
        logging.info("cg finished after %d iterations." % (self.cg_iter))
        return sol[0]

class SmoothImageMeta(object):
    """
    Author: Casey Richardson
    This class implements the non-linear conjugate gradient interface
    to match two images using RKHS metamorphosis via the shooting method.
    """
    def __init__(self, output_dir, config_name, letter_match=None):
        self.num_points = None
        self.domain_max = None
        self.dx = None
        self.verbose_file_output = False
        self.output_dir = output_dir
        self.spline_interp = False
        self.noJ = True
        self.unconstrained = True
        # used only for letter examples
        self.letter_match = letter_match

        smoothImageConfig.configure(self, config_name)
        self.khSmooth = self.khs/2

        self.rg = regularGrid.RegularGrid(self.dim, self.num_points, \
                             self.domain_max, self.dx, "meta")
        self.times = numpy.linspace(self.time_min, self.time_max, \
                            self.num_times)
        self.dt = (self.time_max - self.time_min) / (self.num_times - 1)
        self.optimize_iteration = 0
        self.g_eps = 1.

        logging.info("sigma: %f" % self.sigma)
        logging.info("sfactor: %f" % self.sfactor)
        logging.info("num_points: %s" % str(self.rg.num_points))
        logging.info("domain_max: %s" % str(self.rg.domain_max))
        logging.info("dx: %s" % str(self.rg.dx))
        logging.info("dt: %f" % self.dt)

        rg = self.rg
        self.x0 = self.rg.nodes.copy()
        self.xt = numpy.zeros((rg.num_nodes, 3, self.num_times))
        self.v = numpy.zeros((rg.num_nodes, 3, self.num_times))
        self.m = numpy.zeros((rg.num_nodes, self.num_times))
        self.z = numpy.zeros((rg.num_nodes, 3, self.num_times))
        self.id_x = self.rg.nodes[:,0].copy()
        self.id_y = self.rg.nodes[:,1].copy()
        self.m = numpy.zeros((rg.num_nodes, self.num_times))
        self.alpha = numpy.zeros(rg.num_nodes)
        self.alpha_state = numpy.zeros_like(self.alpha)
        if self.unconstrained:
            self.noJ = True
            self.z0 = numpy.zeros((rg.num_nodes, 3))
            self.z0_state = numpy.zeros((rg.num_nodes, 3))

        if not self.noJ:
            self.J = numpy.ones((rg.num_nodes, self.num_times))

        rg.create_vtk_sg()
        rg.add_vtk_point_data(self.template_in, "template_in")
        rg.add_vtk_point_data(self.target_in, "target_in")
        rg.vtk_write(0, "images", output_dir=self.output_dir)

        if self.spline_interp:
            self.khSmooth = self.khs
            si = SplineInterp(rg, self.khs, self.kho, self.template_in)
            self.dual_template = si.minimize()
            si = SplineInterp(rg, self.khs, self.kho, self.target_in)
            self.dual_target = si.minimize()
        else:
            self.dual_template = self.template_in.copy()
            self.dual_target = self.target_in.copy()

        (templ, dtempl) = kernelMatrix_fort.applyk_and_diff( \
                    rg.nodes, rg.nodes, self.dual_template,
                    self.khSmooth, self.kho, self.rg.num_nodes)
        self.target = kernelMatrix_fort.applyk( \
                    rg.nodes, rg.nodes, self.dual_target,
                    self.khSmooth, self.kho, self.rg.num_nodes)
        self.template = templ
        self.D_template = dtempl

        if True:
            temp = numpy.zeros(rg.num_nodes)
            temp[rg.num_nodes/2 + rg.num_points[1]/2] = 1.
            kvt = kernelMatrix_fort.applyk( \
                        rg.nodes, rg.nodes, temp,
                        self.kvs, self.kvo, self.rg.num_nodes)
            kht = kernelMatrix_fort.applyk( \
                        rg.nodes, rg.nodes, temp,
                        self.khs, self.kho, self.rg.num_nodes)
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.template, "template")
            rg.add_vtk_point_data(self.D_template, "D_template")
            rg.add_vtk_point_data(self.target, "target")
            rg.add_vtk_point_data(kvt, "kvt")
            rg.add_vtk_point_data(kht, "kht")
            rg.vtk_write(0, "initial_data", output_dir=self.output_dir)

        tmpMax = numpy.max(numpy.abs(self.template))
        tarMax = numpy.max(numpy.abs(self.target))
        self.D_template /= tmpMax
        self.template /= tmpMax
        self.dual_template /= tmpMax
        self.target /= tarMax
        self.dual_target /= tarMax


    def getSimData(self):
        return [self.rg, self.num_points, self.num_times]

    # **********************************************************************
    # Implementation of Callback functions for non-linear conjugate gradient
    # **********************************************************************
    def getVariable(self):
        return self

    def randomDir(self):
        if self.unconstrained:
            dirfoo = []
            dirfoo.append(numpy.random.normal(size=self.alpha.shape))
            dirfoo.append(numpy.random.normal(size=self.z0.shape))
        else:
            dirfoo = numpy.random.normal(size=self.alpha.shape)            
        return dirfoo

    def copyDir(self, dir0):
        if self.unconstrained:
            d = [[], []]
            d[0] = dir0[0].copy()
            d[1] = dir0[1].copy()
        else:
            d=dir0.copy()
        return d

    def addProd(self, dir1, dir2, beta):
        if self.unconstrained:
            d = [[], []]
            d[0] = dir1[0] + beta * dir2[0]
            d[1] = dir1[1] + beta * dir2[1]
        else:
            d = dir1 + beta * dir2
        return d




    def objectiveFun(self):
        rg, N, T = self.getSimData()
        self.shoot()
        interp_loc = self.xt[:,:,T-1].copy()
        interp_target = kernelMatrix_fort.applyk( \
                    interp_loc, rg.nodes, self.dual_target,
                    self.khSmooth, self.kho, self.rg.num_nodes)
        diff = self.m[:,T-1] - interp_target
        if self.noJ:
            objFun = numpy.multiply(diff, diff).sum()
        else:
            objFun = (numpy.multiply(diff, diff) * self.J[:,T-1]).sum()

        objFun *= rg.element_volumes[0]
        if (self.verbose_file_output):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.xt[:,:,T-1].real, "x")
            rg.add_vtk_point_data(interp_target.real, "i_target")
            rg.add_vtk_point_data(self.target, "target")
            rg.add_vtk_point_data(self.template, "template")
            rg.add_vtk_point_data(self.m[:,T-1].real, "m")
            rg.add_vtk_point_data(diff, "diff")
            rg.add_vtk_point_data(rg.integrate_dual(diff), "idiff")
            if not self.noJ:
                rg.add_vtk_point_data(self.J[:,T-1].real, "J")
            rg.vtk_write(0, "objFun_test", output_dir=self.output_dir)
        return objFun * self.g_eps

    def updateTry(self, direction, eps, objRef=None):
        rg, N, T = self.getSimData()
        alpha_old = self.alpha.copy()
        x_old = self.xt.copy()
        m_old = self.m.copy()
        z_old = self.z.copy()
        if self.unconstrained:
            z0_old = self.z0.copy()
        if not self.noJ:
            J_old = self.J.copy()
        if self.unconstrained:
            self.alpha = self.alpha_state - direction[0] * eps
            self.z0 = self.z0_state - direction[1] * eps
        else:
            self.alpha = self.alpha_state - direction * eps
        objTry = self.objectiveFun()
        if (objRef != None) and (objTry > objRef):
            self.alpha = alpha_old
            self.xt = x_old
            self.m = m_old
            self.z = z_old
            if not self.noJ:
                self.J = J_old
            if self.unconstrained:
                self.z0 = z0_old
        return objTry

    def acceptVarTry(self):
        rg, N, T = self.getSimData()
        self.alpha_state = self.alpha.copy()
        if self.unconstrained:
            self.z0_state = self.z0.copy()

    def getGradient(self, coeff=1.0):
        rg, N, T = self.getSimData()
        self.shoot()
        if self.noJ:
            dx, dm = self.endPointGradient()
            if self.unconstrained:
                ealpha0, ex0, ez0 = self.adjointSystem((dx, dm))
            else:
                ealpha0, ex0 = self.adjointSystem((dx, dm))
        else:
            dx, dm, dJ = self.endPointGradient()
            ealpha0, ex0 = self.adjointSystem((dx, dm, dJ))
        if (self.verbose_file_output):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(ealpha0, "ealpha")
            rg.vtk_write(0, "get_grad_test", output_dir=self.output_dir)
        if self.unconstrained:
            ez0 = ez0 * (1e-6 + (self.D_template**2).sum(axis=1)[..., numpy.newaxis])
            return (coeff * ealpha0, coeff*ez0)
        else:
            return coeff * ealpha0

    # uncomment the TEMP if you want to use the RKHS-based dot product
    # for non-linear conjugate gradient method
    def dotProduct_TEMP(self, g1, g2):
        x = self.rg.nodes.copy()
        res = numpy.zeros(len(g2))
        for ll in range(len(g2)):
            kg2 = self.KH.applyK(x, g2[ll])
            res[ll] += numpy.dot(g1, kg2)
        return res

    def dotProduct(self, g1, g2):
        res = numpy.zeros(len(g2))
        if self.unconstrained:
            for ll,g in enumerate(g2):
                res[ll] += (g1[0]*g[0]).sum() 
                res[ll] += (g1[1]*g[1]/(1e-6 + (self.D_template**2).sum(axis=1)[..., numpy.newaxis])).sum()
        else:
            for ll,g in enumerate(g2):
                res[ll] += numpy.dot(g1, g)
        return res

    def endPointGradient(self, testGradient=False):
        rg, N, T = self.getSimData()
        interp_loc = self.xt[:,:,T-1].copy()

        (interp_target, d_interp) = kernelMatrix_fort.applyk_and_diff( \
                    interp_loc, rg.nodes, self.dual_target,
                    self.khSmooth, self.kho, self.rg.num_nodes)

        diff = self.m[:,T-1] - interp_target

        dx = numpy.zeros((rg.num_nodes, 3))
        if self.noJ:
            dm = 2*diff
            for k in range(3):
                dx[:,k] = 2.*diff*(-1)*d_interp[:,k]

            dx *= self.g_eps * rg.element_volumes[0]
            dm *= self.g_eps * rg.element_volumes[0]
        else:
            dm = 2*diff*self.J[:,T-1]
            for k in range(3):
                dx[:,k] = 2.*diff*self.J[:,T-1] \
                  *(-1)*d_interp[:,k]

            dJ = diff * diff
            dx *= self.g_eps * rg.element_volumes[0]
            dm *= self.g_eps * rg.element_volumes[0]
            dJ *= self.g_eps * rg.element_volumes[0]

        # test the endpoint gradient
        if testGradient:
            objOld = self.objectiveFun()
            eps = 1e-8
            x = self.xt[:,:,T-1]
            m = self.m[:,T-1]
            if not self.noJ:
                J = self.J[:,T-1]
                jr = numpy.random.randn(J.shape[0])
                J1 = self.J[:,T-1] + eps * jr
                sqrtJ = numpy.sqrt(J1)
            xr = numpy.random.randn(x.shape[0], x.shape[1])
            mr = numpy.random.randn(m.shape[0])

            x1 = self.xt[:,:,T-1] + eps * xr
            m1 = self.m[:,T-1] + eps * mr
            interp_target = kernelMatrix_fort.applyk(x1, rg.nodes, self.dual_target, self.khSmooth, self.kho, self.rg.num_nodes)
            diff = m1 - interp_target.real
            if self.noJ:
                objFun = numpy.multiply(diff, diff).sum() * self.g_eps * \
                  rg.element_volumes[0]
                ip = numpy.multiply(dx, xr).sum(axis=1).sum() \
                  + numpy.dot(dm, mr)
            else:
                objFun = numpy.dot(diff*sqrtJ, sqrtJ*diff) * self.g_eps * \
                  rg.element_volumes[0]
                ip = numpy.multiply(dx, xr).sum(axis=1).sum() \
                  + numpy.dot(dm, mr)  \
                  + numpy.dot(dJ, jr)

            logging.info("Endpoint gradient test: %f, %f" % \
                                ((objFun - objOld)/eps, ip))

            if (self.verbose_file_output):
                rg.create_vtk_sg()
                rg.add_vtk_point_data(diff.real, "diff")
                rg.add_vtk_point_data(d_interp.real, "d_interp")
                rg.add_vtk_point_data(self.target, "target")
                rg.add_vtk_point_data(interp_target, "interp_target")
                rg.add_vtk_point_data(dm.real, "dm")
                rg.add_vtk_point_data(self.m[:,T-1], "m")
                rg.add_vtk_point_data(dx.real, "dx")
                rg.vtk_write(0, "grad_test", output_dir=self.output_dir)
        if self.noJ:
            return [dx, dm]
        else:
            return [dx, dm, dJ]

    def endOfIteration(self):
        self.optimize_iteration += 1
        if (self.optimize_iteration % self.write_iter == 0):
            #self.writeData("iter%d" % (self.optimize_iteration))
            self.writeData("iter")

    def endOptim(self):
            self.writeData("final")
    # ***********************************************************************
    # end of non-linear cg callbacks
    # ***********************************************************************

    def shoot(self):
        rg, N, T = self.getSimData()
        if self.noJ:
            if self.unconstrained:
                (x,m,z,v) = kernelMatrix_fort.shoot_unconstrained(self.dt, \
                                self.sfactor, self.kvs, self.kvo, \
                                self.khs, self.kho, \
                                self.alpha, self.x0, self.template, \
                                self.z0,
                                self.num_times, rg.num_nodes)
            else:
                (x,m,z,v) = kernelMatrix_fort.shoot_noj(self.dt, \
                                self.sfactor, self.kvs, self.kvo, \
                                self.khs, self.kho, \
                                self.alpha, self.x0, self.template, \
                                -1.0*self.D_template,
                                self.num_times, rg.num_nodes)
        else:
            (x,m,z,J,v) = kernelMatrix_fort.shoot(self.dt, \
                                self.sfactor, self.kvs, self.kvo, \
                                self.khs, self.kho, \
                                self.alpha, self.x0, self.template, \
                                -1.0*self.D_template,
                                self.num_times, rg.num_nodes)
            self.J = J
            
        self.xt = x
        self.m = m
        self.z = z
        self.v = v
        #print numpy.fabs(v).max()

    def shoot_numpy(self):
        rg, N, T = self.getSimData()
        x = numpy.zeros((rg.num_nodes, 3, T))
        x[:,:,0] = self.x0.copy()
        m = numpy.zeros((rg.num_nodes, T))
        z = numpy.zeros((rg.num_nodes, 3, T))
        z[:,:,0] = -1.0 * self.D_template
        m[:,0] = self.template.copy()
        for t in range(T-1):

            a = numpy.multiply(z[:,:,t], numpy.vstack( \
                                (self.alpha, self.alpha, self.alpha)).T)

            xt = x[:,:,t]
            v = self.sfactor*self.KV.applyK(xt, a)
            x[:,:,t+1] = xt + self.dt*v

            zt = z[:,:,t]
            ralpha = self.alpha.reshape([self.alpha.shape[0],1])
            rhsz = -1.0*self.dt*(self.sfactor * \
                            self.KV.applyDiffKT(xt, [zt], [a]) + \
                            self.KH.applyDiffKT(xt, [numpy.ones_like(ralpha)],\
                             [ralpha]))
            z[:,:,t+1] = zt + rhsz

            temp = self.KV.precompute(xt, diff=False)
            temp2 = numpy.dot(zt, a.T)
            m1 = self.sfactor * numpy.multiply(temp,temp2).sum(axis=1)
            m2 = self.KH.applyK(xt, self.alpha)
            m[:,t+1] = m[:,t] + self.dt * (0*m1 + m2)

            temp = self.KV.precompute(xt, diff=True)
            zx = 2*(numpy.dot(xt, a.T) - numpy.multiply(xt,a).sum(axis=1))

            if (self.verbose_file_output):
                rg.create_vtk_sg()
                ktest = self.KV.applyK(xt, numpy.ones_like(a))
                temp = numpy.zeros_like(a[:,0])
                temp[100] = 1.
                kvt = self.KV.applyK(xt, temp)
                kht = self.KH.applyK(xt, temp)
                rg.add_vtk_point_data(xt, "x")
                xtc = xt.copy()
                xtc[:,0] = xtc[:,0] - self.id_x
                xtc[:,1] = xtc[:,1] - self.id_y
                rg.add_vtk_point_data(xtc, "xtc")
                rg.add_vtk_point_data(self.alpha, "alpha")
                rg.add_vtk_point_data(kvt, "kvt")
                rg.add_vtk_point_data(kht, "kht")
                rg.add_vtk_point_data(rhsz, "rhsz")
                rg.add_vtk_point_data(self.template, "template")
                rg.add_vtk_point_data(self.target, "target")
                rg.add_vtk_point_data(zt, "z")
                rg.add_vtk_point_data(v, "v")
                rg.add_vtk_point_data(m[:,t], "m")
                rg.add_vtk_point_data(m1[:], "m1")
                rg.add_vtk_point_data(m2[:], "m2")
                rg.add_vtk_point_data(ktest[:,0], "ktest")
                rg.vtk_write(t, "shoot_test", output_dir=self.output_dir)

        self.xt = x.copy()
        self.m = m.copy()
        self.z = z.copy()

    def adjointSystem(self, dq):
        rg, N, T = self.getSimData()

        if self.noJ:
            if self.unconstrained:
                ealpha, ex, ez = kernelMatrix_fort.adjointsystem_unconstrained( \
                        self.dt, self.sfactor, self.kvs, self.kvo, \
                        self.khs, self.kho, \
                        self.alpha, self.xt, self.m, self.z, \
                        dq[0], dq[1], self.num_times, rg.num_nodes)
            else:
                ealpha, ex = kernelMatrix_fort.adjointsystem_noj( \
                        self.dt, self.sfactor, self.kvs, self.kvo, \
                        self.khs, self.kho, \
                        self.alpha, self.xt, self.m, self.z, \
                        dq[0], dq[1], self.num_times, rg.num_nodes)
        else:
            ealpha, ex = kernelMatrix_fort.adjointsystem( \
                        self.dt, self.sfactor, self.kvs, self.kvo, \
                        self.khs, self.kho, \
                        self.alpha, self.xt, self.m, self.z, self.J, \
                        dq[0], dq[1], dq[2], self.num_times, rg.num_nodes)
                    #if self.spline_interp:
        if False:
            si = SplineInterp(rg, self.KH, ealpha)
            ge = si.minimize()
        else:
            ge = ealpha
        if self.unconstrained:
            return ge, ex, ez
        else:
            return ge, ex

    def adjointSystem_numpy(self, dx, dm):
        rg, N, T = self.getSimData()
        ex = numpy.zeros((rg.num_nodes, 3, T))
        ez = numpy.zeros((rg.num_nodes, 3, T))
        em = numpy.zeros((rg.num_nodes, T))

        ealpha = numpy.zeros((rg.num_nodes, T))
        ex[:,:,T-1] = dx.real
        em[:,T-1] = dm.real

        alpha = self.alpha.copy()
        xshape = [alpha.shape[0], 1]
        ralpha = alpha.reshape(xshape)
        oa = numpy.ones_like(ralpha)

        for t in range(T-1, -1, -1):
            xt = self.xt[:,:,t]
            zt = self.z[:,:,t]
            mt = self.m[:,t]
            a = numpy.multiply(zt, numpy.vstack((alpha, alpha, alpha)).T)

            if t > 0:
                dKV = self.KV.precompute(xt, diff=True)

                #  eta_x evolution
                term1 = self.KV.applyDiffKT(xt, [ex[:,:,t]], [a])
                term2 = self.KV.applyDiffKT(xt, [a], [ex[:,:,t]])
                term3 = self.KV.applyDDiffK11(xt, zt, a, ez[:,:,t])
                term4 = self.KV.applyDDiffK12(xt, a, zt, ez[:,:,t])
                term5 = self.KH.applyDDiffK11(xt, oa, ralpha, ez[:,:,t])
                term6 = self.KH.applyDDiffK12(xt, ralpha, oa, ez[:,:,t])

                zz = numpy.dot(zt, a.T)
                g1 = numpy.multiply(dKV, zz)
                g1 = numpy.multiply(g1, em[:,t].reshape(xshape))
                term7 = 2*(numpy.multiply(xt, g1.sum(axis=1).reshape(xshape))-\
                                    numpy.dot(g1, xt))
                zz = numpy.dot(a, zt.T)
                g1 = numpy.multiply(dKV, zz)
                g1 = numpy.multiply(g1, em[:,t])
                term8 = 2*(numpy.multiply(xt, g1.sum(axis=1).reshape(xshape))-\
                                    numpy.dot(g1, xt))

                rem = em[:,t].reshape(xshape)
                term9 = self.KH.applyDiffKT(xt, [rem], [ralpha])
                term10 = self.KH.applyDiffKT(xt, [ralpha], [rem])

                ddKV = self.KV.precompute(xt, diff2=True)

                xxp = (numpy.dot(xt, zt.T) - numpy.multiply(xt, zt)\
                                    .sum(axis=1))


                xxp = (-numpy.dot(zt, xt.T) + numpy.multiply(xt,zt)\
                                    .sum(axis=1).reshape(xshape))

                ex[:,:,t-1] = ex[:,:,t] - self.dt * \
                                    (-self.sfactor*(term1+ \
                                    term2) + self.sfactor*(term3+term4) + \
                                    term5 + term6 \
                                    - self.sfactor*0*(term7+term8) - term9 - \
                                    term10  )

                if (self.verbose_file_output):
                    rg.create_vtk_sg()
                    rg.add_vtk_point_data(term1, "term1")
                    rg.add_vtk_point_data(term2, "term2")
                    rg.add_vtk_point_data(term3, "term3")
                    rg.add_vtk_point_data(term4, "term4")
                    rg.add_vtk_point_data(term5, "term5")
                    rg.add_vtk_point_data(term6, "term6")
                    rg.add_vtk_point_data(term7, "term7")
                    rg.add_vtk_point_data(term8, "term8")
                    rg.add_vtk_point_data(term9, "term9")
                    rg.add_vtk_point_data(term10, "term10")
                    rg.vtk_write(t, "ex_test", output_dir=self.output_dir)

                # eta_z evolution
                term1 = numpy.multiply(self.KV.applyK(xt, ex[:,:,t]), ralpha)

                ezx = 2*(-numpy.dot(ez[:,:,t],xt.T) + \
                                    numpy.multiply(ez[:,:,t],xt)\
                                    .sum(axis=1).reshape(xshape))
                gezx = numpy.multiply(dKV, ezx)
                term2 = numpy.dot(gezx, a)
                zx = 2*(-numpy.dot(xt, ez[:,:,t].T) + \
                                    numpy.multiply(ez[:,:,t],xt)\
                                    .sum(axis=1))
                gezx = numpy.multiply(dKV, zx)
                term3 = numpy.multiply(numpy.dot(gezx, zt), ralpha)

                ka = self.KV.applyK(xt,a)
                term4 = numpy.multiply(em[:,t].reshape(xshape), ka)

                zem = numpy.multiply(zt, em[:,t].reshape(xshape))
                kzem = self.KV.applyK(xt, zem)
                term5 = numpy.multiply(ralpha, kzem)

                ez[:,:,t-1] = ez[:,:,t] - self.dt * \
                                    self.sfactor * (-term1 + \
                                    term2 + term3 -0*term4 - 0*term5)

                if (self.verbose_file_output):
                    rg.create_vtk_sg()
                    rg.add_vtk_point_data(term1, "term1")
                    rg.add_vtk_point_data(term2, "term2")
                    rg.add_vtk_point_data(term3, "term3")
                    rg.add_vtk_point_data(term4, "term4")
                    rg.add_vtk_point_data(term5, "term5")
                    rg.vtk_write(t, "ez_test", output_dir=self.output_dir)

                gk = self.KV.precompute(xt)
                # eta_alpha evolution
                zex = numpy.dot(zt, ex[:,:,t].T)
                term1 = numpy.multiply(gk, zex).sum(axis=1)

                xe = 2 * (-numpy.dot(xt, ez[:,:,t].T) + numpy.multiply(xt, \
                                    ez[:,:,t]).sum(axis=1))
                g1 = numpy.multiply(dKV, xe)
                zz = numpy.dot(zt, zt.T)
                term2 = numpy.multiply(zz, g1).sum(axis=1)
                dKH = self.KH.precompute(xt, diff=True)
                xe = 2 * (-numpy.dot(xt, ez[:,:,t].T) + numpy.multiply(xt, \
                                    ez[:,:,t]).sum(axis=1) )
                g1 = numpy.multiply(dKH, xe)
                term3 = g1.sum(axis=1)

                zz = numpy.dot(zt, zt.T)
                g1 = numpy.multiply(gk, zz)
                term4 = numpy.dot(g1, em[:,t])
                term5 = self.KH.applyK(xt, em[:,t])

                ealpha[:,t-1] = ealpha[:,t] - self.dt * \
                                    (self.sfactor * \
                                    -term1 + self.sfactor * term2 + term3 - \
                                    self.sfactor * 0 * term4 - term5)

                if (self.verbose_file_output):
                    rg.create_vtk_sg()
                    rg.add_vtk_point_data(term1, "term1")
                    rg.add_vtk_point_data(term2, "term2")
                    rg.add_vtk_point_data(term3, "term3")
                    rg.add_vtk_point_data(term4, "term4")
                    rg.add_vtk_point_data(term5, "term5")
                    rg.vtk_write(t, "ea_test", output_dir=self.output_dir)


                # eta_m evolution
                em[:,t-1] = em[:,t]

            if (self.verbose_file_output):
                rg.create_vtk_sg()
                rg.add_vtk_point_data(xt, "x")
                rg.add_vtk_point_data(self.alpha, "alpha")
                rg.add_vtk_point_data(zt, "z")
                rg.add_vtk_point_data(mt, "m")
                rg.add_vtk_point_data(ex[:,:,t], "ex")
                rg.add_vtk_point_data(ez[:,:,t], "ez")
                rg.add_vtk_point_data(em[:,t], "em")
                rg.add_vtk_point_data(ealpha[:,t], "ealpha")
                rg.vtk_write(t, "adjoint_test_%d" % \
                                 (self.optimize_iteration), \
                                 output_dir=self.output_dir)

        if self.spline_interp:
            si = SplineInterp(rg, self.KH, ealpha[:,0])
            ge = si.minimize()
        else:
            ge = self.si.solve(ealpha[:,0])
        return ge

    def computeMatching(self):
        conjugateGradient.cg(self, True, maxIter=1000, TestGradient=False, \
                            epsInit=self.cg_init_eps)
        return self

    def writeData(self, name):
        rg, N, T = self.getSimData()
        psit = numpy.zeros((rg.num_nodes,3, self.num_times))
        #psit_y = numpy.zeros((rg.num_nodes,3, self.num_times))
        psit[:,:,0] = rg.nodes.copy()
        for t in range(1, self.num_times):
            ft = rg.nodes - self.dt * self.v[...,t-1]
            psit[:, 0, t] = rg.grid_interpolate(psit[:,0,t-1], ft).real
            psit[:, 1, t] = rg.grid_interpolate(psit[:,1,t-1], ft).real
            
        for t in range(T):
            rg.create_vtk_sg()
            xtc = self.xt[:,:,t].copy()
            interp_target = kernelMatrix_fort.applyk( \
                            xtc, rg.nodes, self.dual_target,
                            self.khSmooth, self.kho, self.rg.num_nodes)
            meta = rg.grid_interpolate(self.m[:,t], psit[...,t])
            xtc[:,0] = xtc[:,0] - self.id_x
            xtc[:,1] = xtc[:,1] - self.id_y
            rg.add_vtk_point_data(self.xt[:,:,t], "x")
            rg.add_vtk_point_data(xtc, "xtc")
            rg.add_vtk_point_data(self.z[:,:,t], "z")
            rg.add_vtk_point_data(self.m[:,t], "m")
            rg.add_vtk_point_data(self.alpha, "alpha")
            rg.add_vtk_point_data(self.v[...,t], "v")
            if not self.noJ:
                rg.add_vtk_point_data(self.J[:,t], "J")
            rg.add_vtk_point_data(self.template, "template")
            rg.add_vtk_point_data(self.target, "target")
            rg.add_vtk_point_data(interp_target.real, "deformedTarget")
            rg.add_vtk_point_data(meta.real, "metamorphosis")
            rg.vtk_write(t, name, output_dir=self.output_dir)

if __name__ == "__main__":
    # set permanent options
    output_directory_base = smoothImageConfig.compute_output_dir
    #output_directory_base = "./"
    # set options from command line
    parser = optparse.OptionParser()
    parser.add_option("-o", "--output_dir", dest="output_dir")
    parser.add_option("-c", "--config_name", dest="config_name")
    (options, args) = parser.parse_args()
    output_dir = output_directory_base + options.output_dir

    letter_match = None
    if options.config_name == "letter":
        digitId = 11
        templateId = 18
        output_dir_old = output_dir
        for targetId in range(39):
            letter_match = (digitId,templateId,targetId)
            output_dir = "%s_%d_%d_%d" % (output_dir_old, letter_match[0],\
                            letter_match[1], letter_match[2])
                # remove any old results in the output directory
            if os.access(output_dir, os.F_OK):
                shutil.rmtree(output_dir)
            os.mkdir(output_dir)
            if targetId == 0:
                loggingUtils.setup_default_logging(output_dir, smoothImageConfig)
            logging.info(options)
            sim = SmoothImageMeta(output_dir, options.config_name, letter_match)
            sim.computeMatching()
            sim.writeData("final")
    else:
        # remove any old results in the output directory
        if os.access(output_dir, os.F_OK):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        loggingUtils.setup_default_logging(output_dir, smoothImageConfig)
        logging.info(options)
        sim = SmoothImageMeta(output_dir, options.config_name, None)
        sim.computeMatching()
        sim.writeData("final")
