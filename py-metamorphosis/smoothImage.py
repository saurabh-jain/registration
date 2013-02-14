# -*- coding: utf-8 *-*

import numpy
import scipy.linalg.basic
import scipy.sparse.linalg
import os
import logging
import optparse
import shutil
import regularGrid
import pointEvolution
import conjugateGradient
import kernelFunctions as kfun

from PIL import Image

import pdb

class SplineInterp_SVD(object):

    def __init__(self, rg, K):
        self.rg = rg
        self.K = K
        mat = rg.element_volumes[0] * \
                        self.K.precompute(self.rg.nodes, diff=False)
        (U,D,VT) = scipy.linalg.svd(mat)
        self.U = U
        self.Di = numpy.diag(1/D)
        #for k in range(1100,1600):
        #    self.Di[k,k] = 0.
        self.VT = VT
        #self.nm = numpy.sqrt(numpy.power(mat,2).sum())
        #logging.info("eigenvalue: %f" % self.nm)

    def solve(self, rhs):
        return numpy.dot(self.VT.T, numpy.dot(self.Di, numpy.dot(self.U.T, rhs)))


class SplineInterp(object):

    def __init__(self, rg, K, h):
        self.rg = rg
        self.K = K
        self.h = h
        self.cg_iter = 0
        mat = rg.element_volumes[0] * \
                    self.K.precompute(self.rg.nodes, diff=False)
        self.nm = numpy.sqrt(numpy.power(mat,2).sum())
        #self.nm = 30.
        #logging.info("eigenvalue: %f" % self.nm)

    def solve_mult(self, a):
        ka = self.rg.element_volumes[0] * self.K.applyK(self.rg.nodes, a)
        ka += (1./self.nm) * a
        return ka

    def solve_callback(self, sol_k):
        self.cg_iter += 1
        ka = self.solve_mult(sol_k)
        energy = .5 * numpy.dot(sol_k, ka) - \
                             numpy.dot(sol_k, self.h)
        #logging.info("cg iteration %d: energy %f" % (self.cg_iter, energy))
        #rg = self.rg
        #rg.create_vtk_sg()
        #rg.add_vtk_point_data(self.h, "h")
        #rg.add_vtk_point_data(sol_k, "sol_k")
        #rg.add_vtk_point_data(ka, "ka")
        #rg.vtk_write(self.cg_iter, "cg_test", output_dir=".")

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

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.dim = 2
        self.sigma = 3.
        self.sfactor = 1./numpy.power(self.sigma, 2)
        self.num_points = (40,40)
        self.domain_max = (1., 1.)
        #self.domain_max = None
        #self.dx = (1.,1.)
        self.dx = None
        self.rg = regularGrid.RegularGrid(self.dim, self.num_points, \
                             self.domain_max, self.dx, "meta")
        self.num_times = 11
        self.time_min = 0.
        self.time_max = 1.
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
        self.m = numpy.zeros((rg.num_nodes, self.num_times))
        self.z = numpy.zeros((rg.num_nodes, 3, self.num_times))
        self.J = numpy.ones((rg.num_nodes, self.num_times))
        self.id_x = self.rg.nodes[:,0].copy()
        self.id_y = self.rg.nodes[:,1].copy()
        self.m = numpy.zeros((rg.num_nodes, self.num_times))

        kvn = 'laplacian'
        khn = 'laplacian'
        kvs = .07
        khs = .015
        kvo = 2
        kho = 2
        logging.info("KV params: name=%s, sigma=%f, order=%f" % (kvn,kvs,kvo))
        logging.info("KH params: name=%s, sigma=%f, order=%f" % (khn,khs,kho))
        self.KV = kfun.Kernel(name = kvn, sigma=kvs, order=kvo)
        self.KH = kfun.Kernel(name = khn, sigma=khs, order=kho)
        self.alpha = numpy.zeros(rg.num_nodes)
        self.alpha_state = numpy.zeros_like(self.alpha)

#        # initialize some test images
#        self.dual_target = numpy.zeros(rg.num_nodes)
#        self.dual_template = numpy.zeros(rg.num_nodes)
#        val = 1
#        self.dual_template[856] = 1 * val
#        self.dual_template[815] = .5 * val
#        self.dual_template[816] = .5 * val
#        self.dual_template[817] = .5 * val
#        self.dual_template[855] = .5 * val
#        self.dual_template[857] = .5 * val
#        self.dual_template[895] = .5 * val
#        self.dual_template[896] = .5 * val
#        self.dual_template[897] = .5 * val
#
#        self.dual_target[859] = 1 * val
#        self.dual_target[818] = .5 * val
#        self.dual_target[819] = .5 * val
#        self.dual_target[820] = .5 * val
#        self.dual_target[858] = .5 * val
#        self.dual_target[870] = .5 * val
#        self.dual_target[898] = .5 * val
#        self.dual_target[899] = .5 * val
#        self.dual_target[900] = .5 * val
#
#        self.template = self.KH.applyK(rg.nodes, self.dual_template)
#        self.target = self.KH.applyK(rg.nodes, self.dual_target)
#        rdt = self.dual_template.reshape([self.dual_template.shape[0],1])
#        self.D_template = self.KH.applyDiffKT(rg.nodes, \
#                            [numpy.ones_like(rdt)], [rdt])

#        size = (32, 32)
#        ims = [Image.open("eight_1.png").rotate(-90).resize(size), \
#                            Image.open("eight_2.png").rotate(-90).resize(size)]
#        tp = numpy.zeros((32,32))
#        tr = numpy.zeros((32,32))
#        for j in range(size[0]):
#            for k in range(size[0]):
#                tp[j,k] = ims[0].getpixel((j,k)) / 255
#                tr[j,k] = ims[1].getpixel((j,k)) / 255
#        tp1 = numpy.zeros((40,40))
#        tr1 = numpy.zeros((40,40))
#        tp1[4:36,4:36] = tp
#        tr1[4:36,4:36] = tr
#        self.template = tp1.ravel()
#        self.target = tr1.ravel()

        size = self.num_points
        ims = [Image.open("test_images/eight_1b.png").rotate(-90).resize(size), \
                            Image.open("test_images/eight_2b.png").rotate(-90).resize(size)]
        tp = numpy.zeros(size)
        tr = numpy.zeros(size)
        for j in range(size[0]):
            for k in range(size[1]):
                tp[j,k] = ims[0].getpixel((j,k)) / 255.
                tr[j,k] = ims[1].getpixel((j,k)) / 255.
        self.template_in = tp.ravel()
        self.target_in = tr.ravel()

        rg.create_vtk_sg()
        rg.add_vtk_point_data(self.template_in, "template_in")
        rg.add_vtk_point_data(self.target_in, "target_in")
        rg.vtk_write(0, "images", output_dir=self.output_dir)

#        si = SplineInterp(rg, self.KH, self.template_in)
#        self.dual_template = si.minimize()
#        si = SplineInterp(rg, self.KH, self.target_in)
#        self.dual_target = si.minimize()

        self.si = SplineInterp_SVD(rg, self.KH)
        self.dual_template = self.si.solve(self.template_in)
        self.dual_target = self.si.solve(self.target_in)
        rdt = self.dual_template.reshape([self.dual_template.shape[0],1])
        self.D_template = self.rg.element_volumes[0] * \
                            self.KH.applyDiffKT(rg.nodes, \
                            [numpy.ones_like(rdt)], [rdt])
        self.template = self.rg.element_volumes[0] * \
                            self.KH.applyK(rg.nodes, self.dual_template)
        self.target = self.rg.element_volumes[0] * \
                            self.KH.applyK(rg.nodes, self.dual_target)


#        loc = -2.5
#        x_sqr = numpy.power(self.rg.nodes[:,0]-loc, 2)
#        y_sqr = numpy.power(self.rg.nodes[:,1], 2)
#        nodes = numpy.where(x_sqr + y_sqr < 10**2)[0]
#        #self.template = numpy.zeros(rg.num_nodes)
#        #self.template[nodes] = 255
#        self.expf = 20
#        self.template = 10 * numpy.exp(-self.expf*(x_sqr + y_sqr))
#        self.alpha = 0. * numpy.exp(-.005*(x_sqr + y_sqr)) * 1/100.
#        loc = 0.
#        x_sqr = numpy.power(self.rg.nodes[:,0]-loc, 2)
#        y_sqr = numpy.power(self.rg.nodes[:,1], 2)
#        self.target = 10. * numpy.exp(-self.expf*(x_sqr + y_sqr))


    def get_sim_data(self):
        return [self.rg, self.num_points, self.num_times]

    def getVariable(self):
        return self

    def objectiveFun2(self):
        rg, N, T = self.get_sim_data()
        self.shoot()
        interp_loc = self.xt[:,:,T-1].copy()
        #interp_target = rg.grid_interpolate(self.target, interp_loc).real
        interp_target = self.KH.applyK(interp_loc, self.dual_target, y=rg.nodes)
        diff = self.m[:,T-1] - interp_target.real
        sqrtJ = numpy.sqrt(self.J[:,T-1])
        objFun = numpy.dot(diff*sqrtJ, rg.integrate_dual(sqrtJ*diff))
        rg.create_vtk_sg()
        rg.add_vtk_point_data(self.xt[:,:,T-1].real, "x")
        rg.add_vtk_point_data(interp_target.real, "i_target")
        rg.add_vtk_point_data(self.target, "target")
        rg.add_vtk_point_data(self.template, "template")
        rg.add_vtk_point_data(self.m[:,T-1].real, "m")
        rg.add_vtk_point_data(diff, "diff")
        rg.add_vtk_point_data(rg.integrate_dual(diff), "idiff")
        rg.add_vtk_point_data(self.J[:,T-1].real, "J")
        rg.vtk_write(0, "objFun_test", output_dir=self.output_dir)
        if numpy.isnan(objFun):
            import pdb
            pdb.set_trace()
            import sys
            sys.exit()
        return objFun

    def objectiveFun(self):
        rg, N, T = self.get_sim_data()
        self.shoot()
        interp_loc = self.xt[:,:,T-1].copy()
        interp_target = rg.element_volumes[0] * \
                            self.KH.applyK(interp_loc, self.dual_target, y=rg.nodes)
        diff = self.m[:,T-1] - interp_target
        objFun = (numpy.multiply(diff, diff) * self.J[:,T-1]).sum()
        objFun *= rg.element_volumes[0]
        rg.create_vtk_sg()
        rg.add_vtk_point_data(self.xt[:,:,T-1].real, "x")
        rg.add_vtk_point_data(interp_target.real, "i_target")
        rg.add_vtk_point_data(self.target, "target")
        rg.add_vtk_point_data(self.template, "template")
        rg.add_vtk_point_data(self.m[:,T-1].real, "m")
        rg.add_vtk_point_data(diff, "diff")
        rg.add_vtk_point_data(rg.integrate_dual(diff), "idiff")
        rg.add_vtk_point_data(self.J[:,T-1].real, "J")
        rg.vtk_write(0, "objFun_test", output_dir=self.output_dir)
        if numpy.isnan(objFun):
            import pdb
            pdb.set_trace()
            import sys
            sys.exit()
        return objFun * self.g_eps

    def updateTry(self, direction, eps, objRef=None):
        rg, N, T = self.get_sim_data()
        alpha_old = self.alpha.copy()
        x_old = self.xt.copy()
        m_old = self.m.copy()
        z_old = self.z.copy()
        J_old = self.J.copy()
        self.alpha = self.alpha_state - direction * eps
        objTry = self.objectiveFun()
        if (objRef != None) and (objTry > objRef):
            self.alpha = alpha_old
            self.xt = x_old
            self.m = m_old
            self.z = z_old
            self.J = J_old
        return objTry

    def acceptVarTry(self):
        rg, N, T = self.get_sim_data()
        self.alpha_state = self.alpha.copy()

    def getGradient(self, coeff=1.0):
        rg, N, T = self.get_sim_data()
        self.shoot()
        dx, dm, dJ = self.endPointGradient()
        ealpha0 = self.adjointSystem(dx, dm, dJ)
        rg.create_vtk_sg()
        rg.add_vtk_point_data(ealpha0, "ealpha")
        rg.vtk_write(0, "get_grad_test", output_dir=self.output_dir)
        return coeff * ealpha0

    def randomDir(self):
        dirfoo = numpy.random.randn(self.rg.num_nodes) * \
                             1./self.rg.element_volumes[0]
        return dirfoo

    def dotProduct(self, g1, g2):
        x = self.rg.nodes.copy()
        res = numpy.zeros(len(g2))
        for ll in range(len(g2)):
            kg2 = self.rg.element_volumes[0] * self.KH.applyK(x, g2[ll])
            res[ll] += numpy.dot(g1, kg2)
        return res

    def endPointGradient(self):
        rg, N, T = self.get_sim_data()
        interp_loc = self.xt[:,:,T-1].copy()
        interp_target = rg.element_volumes[0] * \
                            self.KH.applyK(interp_loc, self.dual_target, y=rg.nodes)
        diff = self.m[:,T-1] - interp_target

        rdt = self.dual_target.reshape([self.dual_target.shape[0],1])
        d_interp = rg.element_volumes[0] * self.KH.applyDiffKT(interp_loc, \
                            [numpy.ones_like(rdt)],\
                             [rdt], y=rg.nodes)

        dx = numpy.zeros((rg.num_nodes, 3))
        dm = 2*diff*self.J[:,T-1]
        for k in range(3):
            dx[:,k] = 2.*diff*self.J[:,T-1] \
                                *(-1)*d_interp[:,k]

        dJ = diff * diff
        dx *= self.g_eps * rg.element_volumes[0]
        dm *= self.g_eps * rg.element_volumes[0]
        dJ *= self.g_eps * rg.element_volumes[0]

        # test the gradient
        objOld = self.objectiveFun()
        eps = 1e-8
        x = self.xt[:,:,T-1]
        m = self.m[:,T-1]
        J = self.J[:,T-1]
        xr = numpy.random.randn(x.shape[0], x.shape[1])
        #xr[:,1] = 0.
        #temp = xr[213,0]
        #print temp * eps
        #xr[:,0] = 0.
        #xr[213,0] = temp
        mr = numpy.random.randn(m.shape[0])
        jr = numpy.random.randn(J.shape[0])

        x1 = self.xt[:,:,T-1] + eps * xr
        m1 = self.m[:,T-1] + eps * mr
        J1 = self.J[:,T-1] + eps * jr
        interp_target = rg.element_volumes[0] * \
                            self.KH.applyK(x1, self.dual_target, y=rg.nodes)
        diff = m1 - interp_target.real
        sqrtJ = numpy.sqrt(J1)
        objFun = numpy.dot(diff*sqrtJ, sqrtJ*diff) * self.g_eps * \
                                     rg.element_volumes[0]
        #objFun = numpy.dot(diff*sqrtJ, rg.integrate_dual(sqrtJ*diff))

        ip = numpy.multiply(dx, xr).sum(axis=1).sum() \
                    + numpy.dot(dm, mr)  \
                    + numpy.dot(dJ, jr)
        logging.info("Endpoint gradient test: %f, %f" % \
                            ((objFun - objOld)/eps, ip))

        rg.create_vtk_sg()
        rg.add_vtk_point_data(diff.real, "diff")
        rg.add_vtk_point_data(d_interp.real, "d_interp")
        rg.add_vtk_point_data(self.J[:,T-1], "J")
        rg.add_vtk_point_data(dJ, "dJ")
        rg.add_vtk_point_data(self.target, "target")
        rg.add_vtk_point_data(interp_target, "interp_target")
        rg.add_vtk_point_data(dm.real, "dm")
        rg.add_vtk_point_data(self.m[:,T-1], "m")
        rg.add_vtk_point_data(dx.real, "dx")
        rg.vtk_write(0, "grad_test", output_dir=self.output_dir)


        return [dx, dm, dJ]

    def shoot(self):
        rg, N, T = self.get_sim_data()
        x = numpy.zeros((rg.num_nodes, 3, T))
        x[:,:,0] = self.x0.copy()
        m = numpy.zeros((rg.num_nodes, T))
        z = numpy.zeros((rg.num_nodes, 3, T))
        J = numpy.zeros((rg.num_nodes, T))
        J[:,0] = 1.
        z[:,:,0] = -1.0 * self.D_template
        m[:,0] = self.template.copy()
        for t in range(T-1):

            a = numpy.multiply(z[:,:,t], numpy.vstack( \
                                (self.alpha, self.alpha, self.alpha)).T)

            xt = x[:,:,t]
            v = self.sfactor*rg.element_volumes[0]*self.KV.applyK(xt, a)
            x[:,:,t+1] = xt + self.dt*v

            zt = z[:,:,t]
            ralpha = self.alpha.reshape([self.alpha.shape[0],1])
            rhsz = -1.0*self.dt*rg.element_volumes[0]*(self.sfactor * \
                            self.KV.applyDiffKT(xt, [zt], [a]) + \
                            self.KH.applyDiffKT(xt, [numpy.ones_like(ralpha)],\
                             [ralpha]))
            z[:,:,t+1] = zt + rhsz

            temp = self.KV.precompute(xt, diff=False)
            temp2 = numpy.dot(zt, a.T)
            m1 = self.sfactor * numpy.multiply(temp,temp2).sum(axis=1)
            m2 = self.KH.applyK(xt, self.alpha)
            m[:,t+1] = m[:,t] + self.dt * rg.element_volumes[0] * (0*m1 + m2)

            temp = self.KV.precompute(xt, diff=True)
            zx = 2*(numpy.dot(xt, a.T) - numpy.multiply(xt,a).sum(axis=1))
            temp2 = numpy.multiply(temp, zx)
            term1 = numpy.multiply(J[:,t], temp2.sum(axis=1))
            J[:,t+1] = J[:,t] + self.dt * self.sfactor * rg.element_volumes[0]*\
                                 term1

            rg.create_vtk_sg()
            ktest = self.KV.applyK(xt, numpy.ones_like(a))
            temp = numpy.zeros_like(a[:,0])
            temp[100] = 1.
            kvt = self.KV.applyK(xt, temp)
            kht = self.KH.applyK(xt, temp)
            rg.add_vtk_point_data(xt, "x")
            rg.add_vtk_point_data(self.alpha, "alpha")
            rg.add_vtk_point_data(J[:,t], "J")
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
        self.J = J.copy()

    def adjointSystem(self, dx, dm, dJ):
        rg, N, T = self.get_sim_data()
        ex = numpy.zeros((rg.num_nodes, 3, T))
        ez = numpy.zeros((rg.num_nodes, 3, T))
        em = numpy.zeros((rg.num_nodes, T))
        eJ = numpy.zeros((rg.num_nodes, T))
        ealpha = numpy.zeros((rg.num_nodes, T))
        ex[:,:,T-1] = dx.real
        em[:,T-1] = dm.real
        eJ[:,T-1] = dJ.real
        alpha = self.alpha.copy()
        xshape = [alpha.shape[0], 1]
        ralpha = alpha.reshape(xshape)
        oa = numpy.ones_like(ralpha)

        for t in range(T-1, -1, -1):
            xt = self.xt[:,:,t]
            zt = self.z[:,:,t]
            mt = self.m[:,t]
            Jt = self.J[:,t]
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
                jej = numpy.multiply(Jt, eJ[:,t])
                rjej = jej.reshape(xshape)
                na = numpy.dot(rjej, ralpha.T)
                xpja = numpy.multiply(xxp, na)
                u = numpy.multiply(ddKV, xpja)
                t11 = 4*(numpy.multiply(u.sum(axis=1).reshape(xshape), xt) - \
                                     numpy.dot(u, xt))
                u = numpy.multiply(dKV, na)
                term11 = t11 + 2*numpy.dot(u, zt)

                xxp = (-numpy.dot(zt, xt.T) + numpy.multiply(xt,zt)\
                                    .sum(axis=1).reshape(xshape))
                jej = numpy.multiply(Jt, eJ[:,t])
                rjej = jej.reshape(xshape)
                na = numpy.dot(ralpha, rjej.T)
                xpja = numpy.multiply(xxp, na)
                u = numpy.multiply(ddKV, xpja)
                t12 = -4*(numpy.multiply(u.sum(axis=1).reshape(xshape), xt) - \
                                     numpy.dot(u, xt))
                u = numpy.multiply(dKV, na)
                term12 = t12 + -2*numpy.multiply(u.sum(axis=1).\
                                    reshape(xshape), zt)

                ex[:,:,t-1] = ex[:,:,t] - self.dt * rg.element_volumes[0] * \
                                    (-self.sfactor*(term1+ \
                                    term2) + self.sfactor*(term3+term4) + \
                                    term5 + term6 \
                                    - self.sfactor*0*(term7+term8) - term9 - \
                                    term10 - self.sfactor*(term11+term12) )

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
                rg.add_vtk_point_data(term11, "term11")
                rg.add_vtk_point_data(term12, "term12")
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

                Je = numpy.multiply(Jt, eJ[:,t]).reshape(xshape)
                aJe = numpy.dot(ralpha, Je.T)
                g1 = numpy.multiply(dKV, aJe)
                term6 = 2*(-numpy.multiply(xt, g1.sum(axis=1).reshape( \
                                    xshape)) + numpy.dot(g1, xt))
                ez[:,:,t-1] = ez[:,:,t] - self.dt * rg.element_volumes[0] * \
                                    self.sfactor * (-term1 + \
                                    term2 + term3 -0*term4 - 0*term5 - term6)

                rg.create_vtk_sg()
                rg.add_vtk_point_data(term1, "term1")
                rg.add_vtk_point_data(term2, "term2")
                rg.add_vtk_point_data(term3, "term3")
                rg.add_vtk_point_data(term4, "term4")
                rg.add_vtk_point_data(term5, "term5")
                rg.add_vtk_point_data(term6, "term6")
                #rg.add_vtk_point_data(term7, "term7")
                #rg.add_vtk_point_data(term8, "term8")
                #rg.add_vtk_point_data(term9, "term9")
                #rg.add_vtk_point_data(term10, "term10")
                #rg.add_vtk_point_data(term11, "term11")
                #rg.add_vtk_point_data(term12, "term12")
                rg.vtk_write(t, "ez_test", output_dir=self.output_dir)

                gk = self.KV.precompute(xt)
                # eta_alpha evolution

                #kex = self.KV.applyK(xt, ex[:,:,t])
                #term1 = numpy.multiply(zt, kex).sum(axis=1)
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

                Je = numpy.multiply(Jt, eJ[:,t])
                zx = 2*(-numpy.multiply(zt,xt).sum(axis=1).reshape(xshape) \
                                     + numpy.dot(zt, xt.T) )
                g1 = numpy.multiply(dKV, zx)
                g1 = numpy.multiply(g1, Je)
                term6 = g1.sum(axis=1)
                ealpha[:,t-1] = ealpha[:,t] - self.dt * \
                                    rg.element_volumes[0] * (self.sfactor * \
                                    -term1 + self.sfactor * term2 + term3 - \
                                    self.sfactor * 0 * term4 - term5 - \
                                    self.sfactor * term6 )

                rg.create_vtk_sg()
                rg.add_vtk_point_data(term1, "term1")
                rg.add_vtk_point_data(term2, "term2")
                rg.add_vtk_point_data(term3, "term3")
                rg.add_vtk_point_data(term4, "term4")
                rg.add_vtk_point_data(term5, "term5")
                rg.add_vtk_point_data(term6, "term6")
                #rg.add_vtk_point_data(term7, "term7")
                #rg.add_vtk_point_data(term8, "term8")
                #rg.add_vtk_point_data(term9, "term9")
                #rg.add_vtk_point_data(term10, "term10")
                #rg.add_vtk_point_data(term11, "term11")
                #rg.add_vtk_point_data(term12, "term12")
                rg.vtk_write(t, "ea_test", output_dir=self.output_dir)

                # eta_J evolution
                zx = 2*(-numpy.multiply(zt,xt).sum(axis=1) \
                                     + numpy.dot(xt, zt.T) )
                g1 = numpy.multiply(dKV, zx)
                aje = numpy.dot(eJ[:,t].reshape(xshape), ralpha.T)
                term1 = numpy.multiply(g1, aje).sum(axis=1)
                eJ[:,t-1] = eJ[:,t] - self.dt * self.sfactor * \
                                    rg.element_volumes[0] * -term1

                # eta_m evolution
                em[:,t-1] = em[:,t]

            rg.create_vtk_sg()
            rg.add_vtk_point_data(xt, "x")
            rg.add_vtk_point_data(self.alpha, "alpha")
            rg.add_vtk_point_data(Jt, "J")
            rg.add_vtk_point_data(zt, "z")
            rg.add_vtk_point_data(mt, "m")
            rg.add_vtk_point_data(ex[:,:,t], "ex")
            rg.add_vtk_point_data(ez[:,:,t], "ez")
            rg.add_vtk_point_data(em[:,t], "em")
            rg.add_vtk_point_data(ealpha[:,t], "ealpha")
            rg.add_vtk_point_data(eJ[:,t], "eJ")
            rg.vtk_write(t, "adjoint_test_%d" % \
                             (self.optimize_iteration), \
                             output_dir=self.output_dir)

        si = SplineInterp(rg, self.KH, ealpha[:,0])
        #ge = si.minimize()
        ge = self.si.solve(ealpha[:,0])
        #ge = ealpha[:,0]

        rg.create_vtk_sg()
        rg.add_vtk_point_data(ge, "ge")
        ak = self.KH.applyK(self.xt[:,:,0], ge)
        rg.add_vtk_point_data(ak, "ak")
        rg.add_vtk_point_data(ealpha[:,0], "ea")
        rg.add_vtk_point_data(ealpha[:,0]-ak, "diff")
        rg.vtk_write(0, "casey", output_dir=self.output_dir)
        return ge

    def computeMatching(self):
        conjugateGradient.cg(self, True, maxIter=1000, TestGradient=True, \
                            epsInit=2.)
        return self

    def writeData(self, name):
        rg, N, T = self.get_sim_data()
        for t in range(T):
            rg.create_vtk_sg()
            xtc = self.xt[:,:,t].copy()
            xtc[:,0] = xtc[:,0] - self.id_x
            xtc[:,1] = xtc[:,1] - self.id_y
            rg.add_vtk_point_data(self.xt[:,:,t], "x")
            rg.add_vtk_point_data(xtc, "xtc")
            rg.add_vtk_point_data(self.z[:,:,t], "z")
            rg.add_vtk_point_data(self.m[:,t], "m")
            rg.add_vtk_point_data(self.J[:,t], "J")
            rg.add_vtk_point_data(self.alpha, "alpha")
            rg.add_vtk_point_data(self.template, "template")
            rg.add_vtk_point_data(self.target, "target")

            a = numpy.multiply(self.z[:,:,t], numpy.vstack( \
                                (self.alpha, self.alpha, self.alpha)).T)
            vt = self.sfactor*self.KV.applyK(self.xt[:,:,t], a)
            rg.add_vtk_point_data(vt, "v")
            kha = self.KH.applyK(self.xt[:,:,t], self.alpha)
            rg.add_vtk_point_data(kha, "kha")
            rg.vtk_write(t, name, output_dir=self.output_dir)

    def endOfIteration(self):
        self.optimize_iteration += 1
        self.writeData("iter%d" % (self.optimize_iteration))

def setup_default_logging(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    fh = logging.FileHandler("%s/lddmm.log" % (output_dir))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if __name__ == "__main__":

    # set permanent options
    output_directory_base = "/cis/home/clr/compute/smoothImage_meta/"
    #output_directory_base = "./"
    # set options from command line
    parser = optparse.OptionParser()
    parser.add_option("-o", "--output_dir", dest="output_dir")
    (options, args) = parser.parse_args()
    output_dir = output_directory_base + options.output_dir
    # remove any old results in the output directory
    if os.access(output_dir, os.F_OK):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    setup_default_logging(output_dir)
    logging.info(options)
    sim = SmoothImageMeta(output_dir)
    sim.computeMatching()
    sim.writeData("final")
