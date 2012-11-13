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

class SplineInterp(object):

    def __init__(self, rg, K, h):
        self.rg = rg
        self.K = K
        self.h = h
        self.cg_iter = 0

    def solve_mult(self, a):
        ka = self.K.applyK(self.rg.nodes, a)
        return ka

    def solve_callback(self, sol_k):
        self.cg_iter += 1
        ka = self.solve_mult(sol_k)
        energy = .5 * numpy.dot(sol_k, ka) - \
                             numpy.dot(sol_k, self.h)
        logging.info("cg iteration %d: energy %f" % (self.cg_iter, energy))
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
        sol = scipy.sparse.linalg.cg(linop, rhs, \
                            callback=self.solve_callback, maxiter=1000)
        return sol[0]

class SmoothImageMeta(object):

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.dim = 2
        self.sigma = .3
        self.sfactor = 1./numpy.power(self.sigma, 2)
        self.num_points = (20,20)
        self.domain_max = (20.,20.)
        self.dx = None
        self.rg = regularGrid.RegularGrid(self.dim, self.num_points, \
                             self.domain_max, self.dx, "meta")
        self.num_times = 21
        self.time_min = 0.
        self.time_max = 1.
        self.times = numpy.linspace(self.time_min, self.time_max, \
                            self.num_times)
        self.dt = (self.time_max - self.time_min) / (self.num_times - 1)
        self.optimize_iteration = 0
        self.gradEps = 1e-20

        logging.info("sigma: %f" % self.sigma)
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
        #self.alpha = numpy.zeros(self.num_points**2)
        self.m = numpy.zeros((rg.num_nodes, self.num_times))
        self.KV = kfun.Kernel(name = 'gauss', sigma = 3)
        self.KH= kfun.Kernel(name = 'gauss', sigma = 1.5)
        #self.KV = kfun.Kernel(name = 'laplacian', sigma=3.)
        #self.KV= kfun.Kernel(name = 'laplacian', sigma=3, order=3)
        #self.KH= kfun.Kernel(name = 'laplacian', sigma=5, order=3)
        self.alpha = numpy.ones(rg.num_nodes) * 0.
        self.alpha_state = numpy.zeros_like(self.alpha)

        # initialize some test images
        self.dual_target = numpy.zeros(rg.num_nodes)
        self.dual_template = numpy.zeros(rg.num_nodes)
        self.dual_template[210] = 1.
        self.dual_target[212] = 1.
        self.template = self.KH.applyK(rg.nodes, self.dual_template)
        self.target = self.KH.applyK(rg.nodes, self.dual_target)
        rdt = self.dual_template.reshape([self.dual_template.shape[0],1])
        self.D_template = self.KH.applyDiffKT(rg.nodes, \
                            [numpy.ones_like(rdt)], [rdt])

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

#        size = self.num_points
#        ims = [Image.open("eight_1a.png").rotate(-90).resize(size), \
#                            Image.open("eight_2a.png").rotate(-90).resize(size)]
#        tp = numpy.zeros(size)
#        tr = numpy.zeros(size)
#        for j in range(size[0]):
#            for k in range(size[0]):
#                tp[j,k] = ims[0].getpixel((j,k)) / 100.
#                tr[j,k] = ims[1].getpixel((j,k)) / 100.
#        self.template = tp.ravel()
#        self.target = tr.ravel()
#
#        si = SplineInterp(rg, self.KH, self.template)
#        self.dual_template = si.minimize()
#        si = SplineInterp(rg, self.KH, self.target)
#        self.dual_target = si.minimize()
#        rdt = self.dual_template.reshape([self.dual_template.shape[0],1])
#        self.D_template = self.KH.applyDiffKT(rg.nodes, \
#                            [numpy.ones_like(rdt)], [rdt])

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

    def objectiveFun(self):
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
        #si = SplineInterp(rg, self.KH, ealpha[:,0])
        #ge = si.minimize()
        return coeff * ealpha0

    def dotProduct(self, g1, g2):
        x = self.rg.nodes.copy()
        res = numpy.zeros(len(g2))
        for ll in range(len(g2)):
            kg2 = self.KH.applyK(x, g2[ll])
            res[ll] += numpy.dot(g1, kg2)
        return res

    def endPointGradient(self):
        rg, N, T = self.get_sim_data()
        interp_loc = self.xt[:,:,T-1].copy()
        #interp_target = rg.grid_interpolate(self.target, interp_loc).real
        interp_target = self.KH.applyK(interp_loc, self.dual_target, y=rg.nodes)
        diff = self.m[:,T-1] - interp_target
        #d_interp = rg.grid_interpolate_gradient_2d(self.target, interp_loc).real

        rdt = self.dual_target.reshape([self.dual_target.shape[0],1])
        d_interp = self.KH.applyDiffKT(interp_loc, \
                            [numpy.ones_like(rdt)],\
                             [rdt])

        sJ = numpy.sqrt(self.J[:,T-1])
        dm = 2*sJ*rg.integrate_dual(sJ * diff).real
        dx = numpy.zeros((rg.num_nodes, 3))

        for k in range(3):
            dx[:,k] = 2.*sJ*rg.integrate_dual(sJ*diff).real \
                                *(-1)*d_interp[:,k].real
        dJ = rg.integrate_dual(sJ*diff).real * diff * (1./sJ)

        # test the gradient
        print "endpoint grad test"
        objOld = self.objectiveFun()
        eps = 1e-12
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
        interp_target = self.KH.applyK(x1, self.dual_target, y=rg.nodes)
        diff = m1 - interp_target.real
        sqrtJ = numpy.sqrt(J1)
        objFun = numpy.dot(diff*sqrtJ, rg.integrate_dual(sqrtJ*diff))
        ip = numpy.multiply(dx, xr).sum(axis=1).sum() \
                    + numpy.dot(dm, mr)  \
                    + numpy.dot(dJ, jr)
        print (objFun - objOld)/eps, ip

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

        pdb.set_trace()

        return [dx, dm, dJ]

    def shoot(self):
        rg, N, T = self.get_sim_data()
        x = numpy.zeros((rg.num_nodes, 3, T))
        x[:,:,0] = self.x0.copy()
        m = numpy.zeros((rg.num_nodes, T))
        z = numpy.zeros((rg.num_nodes, 3, T))
        J = numpy.zeros((rg.num_nodes, T))
        J[:,0] = 1.
        #z[:,:,0] = -1.0 * rg.gradient(self.template).real
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
            z[:,:,t+1] = zt + -1.0*self.dt*(self.sfactor * \
                            self.KV.applyDiffKT(xt, [zt], [a]) + \
                            self.KH.applyDiffKT(xt, [numpy.ones_like(ralpha)],\
                             [ralpha]))

            #temp = self.KV.applyK(xt, a)
            #dtemp = numpy.zeros(N**2)
            #for d in range(3):
            #    dtemp[:] += zt[:,d] * temp[:,d]
            #m1 = self.sfactor * dtemp
            #m2 = self.KH.applyK(xt, self.alpha)
            #m[:,t+1] = m[:,t] + self.dt * (m1 + m2)
            temp = self.KV.precompute(xt, diff=False)
            temp2 = numpy.dot(zt, a.T)
            m1 = self.sfactor * (temp*temp2).sum(axis=1)
            m2 = self.KH.applyK(xt, self.alpha)
            m[:,t+1] = m[:,t] + self.dt * (m1 + m2)

            temp = self.KV.precompute(xt, diff=True)
            zx = 2*(numpy.dot(xt, a.T) - numpy.multiply(xt,a).sum(axis=1))
            term1 = (temp*zx).sum(axis=1) * J[:,t]
            J[:,t+1] = J[:,t] + self.dt * self.sfactor * term1


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

        for t in range(T-1, -1, -1):
            xt = self.xt[:,:,t]
            zt = self.z[:,:,t]
            mt = self.m[:,t]
            Jt = self.J[:,t]
            a = numpy.multiply(zt, numpy.vstack((alpha, alpha, alpha)).T)

            if t > 0:
                #  eta_x evolution
                term1 = self.KV.applyDiffKT(xt, [ex[:,:,t]], [a])
                term2 = self.KV.applyDiffKT(xt, [a], [ex[:,:,t]])
                term3 = self.KV.applyDDiffK11(xt, zt, a, ez[:,:,t])
                term4 = self.KV.applyDDiffK12(xt, a, zt, ez[:,:,t])
                oa = numpy.ones_like(alpha)
                term5 = self.KH.applyDDiffK11(xt, oa.reshape( \
                                    [oa.shape[0], 1]), \
                                    alpha.reshape([alpha.shape[0],1]), \
                                    ez[:,:,t])
                term6 = self.KH.applyDDiffK12(xt, \
                                    alpha.reshape([alpha.shape[0],1]), \
                                    oa.reshape([oa.shape[0],1]), ez[:,:,t])
                zem = numpy.multiply(zt, em[:,t].reshape([xt.shape[0],1]))
                term7 = self.KV.applyDiffKT(xt, [zem],[a])
                term8 = self.KV.applyDiffKT(xt, [a], [zem])
                rem = em[:,t].reshape([em[:,t].shape[0],1])
                ra = alpha.reshape([alpha.shape[0],1])
                term9 = self.KH.applyDiffKT(xt, [rem], [ra])
                term10 = self.KH.applyDiffKT(xt, [ra], [rem])
                jej = numpy.multiply(Jt, eJ[:,t])
                rjej = jej.reshape([jej.shape[0], 1])
                ra = alpha.reshape([alpha.shape[0], 1])
                term11 = -1.0*self.KV.applyDDiffK12(xt, rjej, ra, zt)
                term12 = -1.0*self.KV.applyDDiffK11(xt, ra, rjej, zt)
                ex[:,:,t-1] = ex[:,:,t] - self.dt * (-self.sfactor*(term1+ \
                                    term2) + self.sfactor*(term3+term4) + \
                                     term5 + term6 \
                                    - self.sfactor*(term7+term8) - term9 - \
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
                term1 = numpy.multiply(self.KV.applyK(xt, ex[:,:,t]), \
                                    alpha.reshape([xt.shape[0],1]))
                g = self.KV.precompute(xt, diff=True)
                xe = 2. * (-numpy.dot(xt, ez[:,:,t].T) + numpy.multiply(xt, \
                                    ez[:,:,t]).sum(axis=1) )
                g1 = numpy.multiply(g, xe)
                g1 = numpy.multiply(g1, numpy.dot(
                                    alpha.reshape([alpha.shape[0],1]), \
                                    oa.reshape([oa.shape[0],1]).T))
                term2 = numpy.dot(g1, zt)

                #xe = 2. * (numpy.multiply(xt, ez[:,:,t]).sum(axis=1)[:, \
                #                    numpy.newaxis]-numpy.dot(ez[:,:,t], xt.T) )
                xe = 2. * (numpy.multiply(xt, ez[:,:,t]).sum(axis=1) \
                                    -numpy.dot(ez[:,:,t], xt.T) )
                g1 = numpy.multiply(g, xe)
                ezz = numpy.dot(g1, a)
                term3 = ezz.copy()
                ae = numpy.dot(em[:,t].reshape([em[:,t].shape[0],1]), \
                                     alpha.reshape([alpha.shape[0],1]).T)
                gk = self.KV.precompute(xt)
                g1 = numpy.multiply(gk, ae)
                term4 = numpy.dot(g1, zt)
                zem = numpy.multiply(zt, em[:,t].reshape([em[:,t].shape[0],1]))
                kzem = self.KV.applyK(xt, zem)
                term5 = numpy.multiply(alpha.reshape([alpha.shape[0],1]), kzem)
                Je = numpy.multiply(Jt, eJ[:,t]).reshape([Jt.shape[0],1])
                ra = self.alpha.reshape([self.alpha.shape[0],1])
                aJe = numpy.dot(ra, Je.T)
                g1 = numpy.multiply(g, aJe)
                term6 = 2*(-numpy.multiply(xt, g1.sum(axis=1).reshape( \
                                    [xt.shape[0],1])) + numpy.dot(g1, xt))
                ez[:,:,t-1] = ez[:,:,t] - self.dt * self.sfactor * (-term1 + \
                                    term2 + term3 + (-term4 - term5) - term6)

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

                # eta_alpha evolution
                #ze = numpy.dot(zt, ex[:,:,t].T)
                #g1 = numpy.multiply(gk, ze)
                #term1 = g1.sum(axis=1)
                ke = self.KV.applyK(xt, ex[:,:,t])
                term1 = (ke * zt).sum(axis=1)
                xe = 2 * (-numpy.dot(xt, ez[:,:,t].T) + numpy.multiply(xt, \
                                    ez[:,:,t]).sum(axis=1) )
                g1 = numpy.multiply(g, xe)
                zz = numpy.dot(zt, zt.T)
                term2 = numpy.multiply(zz, g1).sum(axis=1)
                gh = self.KH.precompute(xt, diff=True)
                xe = 2* (-numpy.dot(xt, ez[:,:,t].T) + numpy.multiply(xt, \
                                    ez[:,:,t]).sum(axis=1) )
                g1 = numpy.multiply(gh, xe)
                term3 = g1.sum(axis=1)
                zz = numpy.dot(zt, zt.T)
                g1 = numpy.multiply(gk, zz)
                g1 = numpy.multiply(g1, em[:,t])
                term4 = g1.sum(axis=1)
                #term5 = numpy.multiply(gh, em[:,t]).sum(axis=1)
                term5 = self.KH.applyK(xt, em[:,t])
                Je = numpy.multiply(Jt, eJ[:,t])
                zx = 2*(numpy.multiply(xt,zt).sum(axis=1) - numpy.dot( \
                                    xt, zt.T) )
                g1 = numpy.multiply(g, zx)
                g1 = numpy.multiply(g1, Je)
                term6 = g1.sum(axis=1)
                ealpha[:,t-1] = ealpha[:,t] - self.dt * (self.sfactor * \
                                    -term1 + self.sfactor * term2 + term3 - \
                                    self.sfactor * term4 - term5 - \
                                    self.sfactor * term6 )

                rg.create_vtk_sg()
                rg.add_vtk_point_data(ke, "ke")
                rg.add_vtk_point_data(zt, "zt")
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
                zx = 2*(numpy.multiply(zt,xt).sum(axis=1) - numpy.dot( \
                                    zt, xt.T) )
                g1 = numpy.multiply(g, zx)
                aje = numpy.dot(eJ[:,t].reshape([eJ[:,t].shape[0],1]), \
                                     alpha.reshape([alpha.shape[0],1]).T)
                term1 = numpy.multiply(g1, aje).sum(axis=1)
                eJ[:,t-1] = eJ[:,t] - self.dt * self.sfactor * -term1

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
            rg.vtk_write(t, "adjoint_test", output_dir=self.output_dir)

        si = SplineInterp(rg, self.KH, ealpha[:,0])
        ge = si.minimize()

        rg.create_vtk_sg()
        rg.add_vtk_point_data(ge, "ge")
        ak = self.KH.applyK(self.xt[:,:,0], ge)
        rg.add_vtk_point_data(ak, "ak")
        rg.add_vtk_point_data(ealpha[:,0], "ea")
        rg.vtk_write(0, "casey", output_dir=self.output_dir)
        return ge

    def computeMatching(self):
        conjugateGradient.cg(self, True, maxIter=1000, TestGradient=True, \
                            epsInit=1e-5)
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
