import regularGrid
import semiLagrangian
import diffeomorphisms
import conjugateGradient
import numpy
import optparse
import os
import shutil
import logging

import pyfftw
import time

class ImageTimeSeries(object):

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.initialize_lung()
        #self.initialize_biocard()
        self.mu = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.mu_state = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.v = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.objTry = 0.
        self.mu_state = self.mu.copy()
        self.optimize_iteration = 0

        # initialize fftw information
        self.fft_thread_count = 16
        self.fft_vec_shape = self.rg.dims
        self.in_vec = pyfftw.n_byte_align_empty(self.fft_vec_shape, 16, \
                            dtype=numpy.complex)
        self.fft_vec = pyfftw.n_byte_align_empty(self.fft_vec_shape, 16, \
                            dtype=numpy.complex)
        self.out_vec = pyfftw.n_byte_align_empty(self.fft_vec_shape, 16, \
                            dtype=numpy.complex)
        self.wfor = pyfftw.FFTW(self.in_vec, self.fft_vec, \
                            axes=range(self.dim), \
                            threads=self.fft_thread_count)
        self.wback = pyfftw.FFTW(self.in_vec, self.fft_vec, \
                            axes=range(self.dim), \
                            direction='FFTW_BACKWARD', \
                            threads=self.fft_thread_count)

        #self.pool_size = 16
        #self.pool = multiprocessing.Pool(self.pool_size)

        self.update_evolutions()

    def initialize_lung(self):
        self.dim = 3
        self.num_target_times = 5
        self.num_times_disc = 10
        self.num_times = self.num_times_disc * self.num_target_times + 1
        self.times = numpy.linspace(0, 1, self.num_times)
        self.dt = 1. / (self.num_times - 1)
        self.sigma = .1
        self.num_points_data = numpy.array([256, 184, 160])
        #self.num_points_data = numpy.array([64,64,64])
        #self.num_points_data = numpy.array([2,4,6])
        self.mults = numpy.array([2,2,2]).astype(int)
        self.num_points = (self.num_points_data/self.mults).astype(int)
        #self.num_points = (32,32,32)
        self.dx_data = (1., 1., 1.)
        self.domain_max_data = None
        self.dx = None # (1.,1.,1.)
        self.domain_max = numpy.array([128., 92., 80.])
        #self.domain_max = (domain_max, domain_max, domain_max)
        self.gradEps = 1e-8
        self.rg = regularGrid.RegularGrid(self.dim, \
                            num_points=self.num_points, \
                            domain_max=self.domain_max, \
                            dx=self.dx, mesh_name="lddmm")
        self.rg_data = regularGrid.RegularGrid(self.dim, \
                            num_points=self.num_points_data, \
                            domain_max=self.domain_max_data, \
                            dx=self.dx_data, mesh_name="lddmm_data")
        #import pdb
        #pdb.set_trace()
        self.I = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.I_interp = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.p = numpy.zeros((self.rg.num_nodes, self.num_times))

        sc = diffeomorphisms.gridScalars()
        sc.loadAnalyze("/cis/home/clr/cawork/lung/test007.hdr")
        #test_data = sc.data[:,0:184,:].copy()
        #test_data = numpy.zeros((256,192,160))
        #test_data = numpy.zeros((64,64,64))
        #test_data[0:256,0:190,0:160] = sc.data.copy()
        #test_data.shape = test_data.size

        #test_data = sc.data.copy()
        #test_data.shape = self.rg_data.num_nodes

        #self.rg_data.create_vtk_sg()
        #self.rg_data.add_vtk_point_data(test_data,"I")
        #self.rg_data.vtk_write(0,"test", output_dir=self.output_dir)

        #import pdb
        #pdb.set_trace()

        self.I[:,0.*self.num_times_disc] = self.apply_sync_filter( \
                            sc.data.reshape(self.rg_data.num_nodes),self.mults)
        sc.loadAnalyze("/cis/home/clr/cawork/lung/test009.hdr")
        self.I[:,1.*self.num_times_disc] = self.apply_sync_filter( \
                            sc.data.reshape(self.rg_data.num_nodes),self.mults)
        sc.loadAnalyze("/cis/home/clr/cawork/lung/test011.hdr")
        self.I[:,2.*self.num_times_disc] = self.apply_sync_filter( \
                            sc.data.reshape(self.rg_data.num_nodes),self.mults)
        sc.loadAnalyze("/cis/home/clr/cawork/lung/test013.hdr")
        self.I[:,3.*self.num_times_disc] = self.apply_sync_filter( \
                            sc.data.reshape(self.rg_data.num_nodes),self.mults)
        sc.loadAnalyze("/cis/home/clr/cawork/lung/test015.hdr")
        self.I[:,4.*self.num_times_disc] = self.apply_sync_filter( \
                            sc.data.reshape(self.rg_data.num_nodes),self.mults)

        self.I /= 255.

        for t in range(self.num_target_times):
            self.rg.create_vtk_sg()
            self.rg.add_vtk_point_data(self.I[:,t*self.num_times_disc], "I")
            self.rg.vtk_write(t, "targets", output_dir=self.output_dir)

        logging.info("lung data image parameters: ")
        logging.info("dimension: %d" % (self.dim))
        logging.info("num_points: %s" % (str(self.rg.num_points)))
        logging.info("domain_max: %s" % (str(self.rg.domain_max)))
        logging.info("dx: %s" % (str(self.rg.dx)))
        logging.info("sigma: %f" % (self.sigma))
        logging.info("dt: %f" % (self.dt))

    def initialize_biocard(self):
        self.dim = 3
        self.num_target_times = 3
        self.num_times_disc = 10
        self.num_times = self.num_times_disc * self.num_target_times + 1
        self.times = numpy.linspace(0, 1, self.num_times)
        self.dt = 1. / (self.num_times - 1)
        self.sigma = .01
        self.num_points = (40, 32, 40)
        self.dx = (.9375, 2., .9375)
        self.domain_max = None
        self.gradEps = 1e-8
        self.rg = regularGrid.RegularGrid(self.dim, \
                            num_points=self.num_points, \
                            domain_max=self.domain_max, \
                            dx=self.dx, mesh_name="lddmm")
        self.I = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.I_interp = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.p = numpy.zeros((self.rg.num_nodes, self.num_times))

        sc = diffeomorphisms.gridScalars()
        sc.loadAnalyze("/cis/home/clr/cawork/biocard/regR2_cut.hdr")
        self.I[:,0.*self.num_times_disc] = sc.data.reshape(self.rg.num_nodes)
        sc.loadAnalyze("/cis/home/clr/cawork/biocard/regR3_cut.hdr")
        self.I[:,1.*self.num_times_disc] = sc.data.reshape(self.rg.num_nodes)
        sc.loadAnalyze("/cis/home/clr/cawork/biocard/regR4_cut.hdr")
        self.I[:,2.*self.num_times_disc] = sc.data.reshape(self.rg.num_nodes)
        sc.loadAnalyze("/cis/home/clr/cawork/biocard/regR5_cut.hdr")
        self.I[:,3.*self.num_times_disc] = sc.data.reshape(self.rg.num_nodes)
        logging.info("Biocard image parameters: ")
        logging.info("dimension: %d" % (self.dim))
        logging.info("num_points: %s" % (str(self.rg.num_points)))
        logging.info("domain_max: %s" % (str(self.rg.domain_max)))
        logging.info("dx: %s" % (str(self.rg.dx)))
        logging.info("sigma: %f" % (self.sigma))
        logging.info("dt: %f" % (self.dt))

    def initialize_lddmm(self):
        self.dim = 3
        self.num_target_times = 1
        self.num_times_disc = 10
        self.num_times = self.num_times_disc * self.num_target_times + 1
        self.times = numpy.linspace(0, 1, self.num_times)
        self.dt = 1. / (self.num_times - 1)
        self.sigma = .01
        self.num_points = (32, 32, 32)
        self.dx = (.9375, 2., .9375)
        self.domain_max = None
        #self.domain_max = (domain_max, domain_max, domain_max)
        self.gradEps = 1e-8
        self.rg = regularGrid.RegularGrid(self.dim, \
                            num_points=self.num_points, \
                            domain_max=self.domain_max, \
                            dx=self.dx, mesh_name="lddmm")
        self.I = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.I_interp = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.p = numpy.zeros((self.rg.num_nodes, self.num_times))

        sc = diffeomorphisms.gridScalars()
        sc.loadAnalyze("/cis/home/clr/cawork/biocard/reg2_cut.hdr")
        self.I[:,0.*self.num_times_disc] = sc.data.reshape(self.rg.num_nodes)
        sc.loadAnalyze("/cis/home/clr/cawork/biocard/reg5_cut.hdr")
        self.I[:,1.*self.num_times_disc] = sc.data.reshape(self.rg.num_nodes)
        logging.info("Biocard image parameters: ")
        logging.info("dimension: %d" % (self.dim))
        logging.info("num_points: %s" % (str(self.rg.num_points)))
        logging.info("domain_max: %s" % (str(self.rg.domain_max)))
        logging.info("dx: %s" % (str(self.rg.dx)))
        logging.info("sigma: %f" % (self.sigma))
        logging.info("dt: %f" % (self.dt))

    def initialize_test_3d(self):
        self.dim = 3
        self.num_target_times = 4
        num_points = 64
        domain_max = 40
        self.num_times_disc = 20
        self.num_times = self.num_times_disc * self.num_target_times + 1
        self.times = numpy.linspace(0, 1, self.num_times)
        self.dt = 1. / (self.num_times - 1)
        self.sigma = .1
        self.num_points = (num_points, num_points, num_points)
        self.dx = None
        self.domain_max = (domain_max, domain_max, domain_max)
        self.gradEps = 1e-8
        self.rg = regularGrid.RegularGrid(self.dim, \
                            num_points=self.num_points, \
                            domain_max=self.domain_max, \
                            dx=self.dx, mesh_name="lddmm")
        self.I = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.I_interp = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.p = numpy.zeros((self.rg.num_nodes, self.num_times))
        for t in range(self.num_times):
            loc = -1. + 0 * self.dt * (t -(t % self.num_times_disc))
            x_sqr = numpy.power(self.rg.nodes[:,0]-loc, 2)
            #y_sqr = numpy.power(self.rg.nodes[:,1], 2)
            z_sqr = numpy.power(self.rg.nodes[:,2], 2)
            r = 13 - 8 * self.dt * (t - (t % self.num_times_disc))
            nodes = numpy.where(x_sqr + y_sqr + z_sqr < r**2)[0]
            self.I[nodes, t] = 50
        logging.info("Test 3d parameters: ")
        logging.info("dimension: %d" % (self.dim))
        logging.info("num_points: %s" % (str(self.rg.num_points)))
        logging.info("domain_max: %s" % (str(self.rg.domain_max)))
        logging.info("dx: %s" % (str(self.rg.dx)))
        logging.info("sigma: %f" % (self.sigma))
        logging.info("dt: %f" % (self.dt))

    def initialize_test_2d(self):
        self.dim = 2
        self.num_target_times = 4
        num_points = 64
        domain_max = 40
        self.num_times_disc = 20
        self.num_times = self.num_times_disc * self.num_target_times + 1
        self.times = numpy.linspace(0, 1, self.num_times)
        self.dt = 1. / (self.num_times - 1)
        self.sigma = .1
        self.num_points = (num_points, num_points)
        self.dx = None
        self.domain_max = (domain_max, domain_max)
        self.gradEps = 1e-8
        self.rg = regularGrid.RegularGrid(self.dim, \
                            num_points=self.num_points, \
                            domain_max=self.domain_max, \
                            dx=self.dx, mesh_name="lddmm")
        self.I = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.I_interp = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.p = numpy.zeros((self.rg.num_nodes, self.num_times))
        for t in range(self.num_times):
            loc = -1. + 0 * self.dt * (t -(t % self.num_times_disc))
            x_sqr = numpy.power(self.rg.nodes[:,0]-loc, 2)
            y_sqr = numpy.power(self.rg.nodes[:,1], 2)
            z_sqr = numpy.power(self.rg.nodes[:,2], 2)
            r = 13 - 8 * self.dt * (t - (t % self.num_times_disc))
            nodes = numpy.where(x_sqr + y_sqr + z_sqr < r**2)[0]
            self.I[nodes, t] = 50
        logging.info("Test 2d parameters: ")
        logging.info("dimension: %d" % (self.dim))
        logging.info("num_points: %s" % (str(self.rg.num_points)))
        logging.info("domain_max: %s" % (str(self.rg.domain_max)))
        logging.info("dx: %s" % (str(self.rg.dx)))
        logging.info("sigma: %f" % (self.sigma))
        logging.info("dt: %f" % (self.dt))

    def get_sim_data(self):
        return [self.rg, self.num_points, self.num_times]

    def get_kernelv(self):
        rg, N, T = self.get_sim_data()
        a = 1
        b = 1./a
        r_sqr_xsi = (numpy.power(rg.xsi_1,2) + numpy.power(rg.xsi_2,2) + \
                            numpy.power(rg.xsi_3,2))
        #r_sqr_xsi = (numpy.power(fnodes[:,:,0],2) + numpy.power(fnodes[:,:,1],2) + \
        #                    numpy.power(fnodes[:,:,2],2))
        #Kv = 2.*numpy.pi * b * numpy.exp(-b/2. * r_sqr_xsi)
        alpha = 2
        Kv = 1.0 / numpy.power(1 + alpha*(r_sqr_xsi),2)
        return Kv

    def apply_sync_filter(self, right, mults):
        rg, N, T = self.get_sim_data()
        sf = self.rg_data.sync_filter(mults)
        sf = numpy.fft.fftshift(sf)
        rr = right.copy().astype(complex)
        fr = numpy.reshape(rr, self.rg_data.dims)
        fr = numpy.fft.fftshift(fr)
        fr = numpy.fft.fft2(fr * self.rg_data.element_volumes[0])
        fr = fr * sf
        out = numpy.fft.ifft2(fr) * 1./self.rg_data.element_volumes[0]
        out = numpy.fft.fftshift(out)
        out = out[range(0,self.num_points_data[2],mults[2]),:,:]
        out = out[:,range(0,self.num_points_data[1],mults[1]), :]
        out = out[:,:,range(0,self.num_points_data[0],mults[0])]
        out = out.real.ravel()
        return out

    def apply_kernel_V_fftw(self, right):
        rg, N, T = self.get_sim_data()
        krho = numpy.zeros((rg.num_nodes, 3))
        for j in range(self.dim):
            rr = right[:,j].copy().astype(complex)
            fr = numpy.reshape(rr, rg.dims)
            fr = numpy.fft.fftshift(fr)
            fr *= rg.element_volumes[0]
            self.in_vec[...] = fr[...]
            self.wfor.execute()
            fr[...] = self.fft_vec[...]
            Kv = self.get_kernelv()
            Kv = numpy.fft.fftshift(Kv)
            fr = fr * Kv
            self.fft_vec[...] = fr[...]
            self.wback.execute()
            out = self.out_vec[...] / self.in_vec.size
            out *= 1/rg.element_volumes[0]
            out = numpy.fft.fftshift(out)
            krho[:,j] = out.real.ravel()
        return krho

    def apply_kernel_V(self, right):
        rg, N, T = self.get_sim_data()
        krho = numpy.zeros((rg.num_nodes, 3))
        if self.dim==2:
            for j in range(self.dim):
              rr = right[:,j].copy().astype(complex)
              fr = numpy.reshape(rr, rg.dims)
              fr = numpy.fft.fftshift(fr)
              fr = numpy.fft.fft2(fr * rg.element_volumes[0])
              Kv = self.get_kernelv()
              Kv = numpy.fft.fftshift(Kv)
              fr = fr * Kv
              out = numpy.fft.ifft2(fr) * 1./rg.element_volumes[0]
              out = numpy.fft.fftshift(out)
              krho[:,j] = out.real.ravel()
        elif self.dim==3:
            for j in range(self.dim):
              rr = right[:,j].copy().astype(complex)
              fr = numpy.reshape(rr, rg.dims)
              fr = numpy.fft.fftshift(fr)
              fr = numpy.fft.fftn(fr * rg.element_volumes[0])
              Kv = self.get_kernelv()
              Kv = numpy.fft.fftshift(Kv)
              fr = fr * Kv
              out = numpy.fft.ifftn(fr) * 1./rg.element_volumes[0]
              out = numpy.fft.fftshift(out)
              krho[:,j] = numpy.reshape(out.real, (rg.num_nodes))
        return krho

    def kernel_norm_V(self, right):
        rg, N, T = self.get_sim_data()
        kright = self.apply_kernel_V(right)
        kn = 0.
        kn += numpy.dot(right[:,0], rg.integrate_dual(kright[:,0]))
        kn += numpy.dot(right[:,1], rg.integrate_dual(kright[:,1]))
        kn += numpy.dot(right[:,2], rg.integrate_dual(kright[:,2]))
        return kn

    def k_mu(self):
        rg, N, T = self.get_sim_data()
        start = time.time()
        for t in range(T):
          self.v[:,:,t] = self.apply_kernel_V(self.mu[:,:,t])
        self.v = self.v.real
        logging.info("k_mu time: %f" % (time.time()-start))

    def update_evolutions(self):
        rg, N, T = self.get_sim_data()
        self.k_mu()
        start = time.time()
        self.v[rg.edge_nodes,:,:] = 0.
        # compute the flows needed to update evolutions
        intv = semiLagrangian.integrate(rg, N, T, self.v, 3, self.dt)
        logging.info("outer integrate: %f" % (time.time()-start))
        self.h = intv.copy()
        self.h2 = numpy.zeros((rg.num_nodes, 3, T))
        # compute image and covector push-forwards
        dj = numpy.zeros((rg.num_nodes, T))
        #for t in range(T):
        #    self.I_interp[:,t] = self.rg.grid_interpolate(self.I[:,0],
        #                    self.h[:,:,t]).real
        logging.info("i interp: %f" % (time.time()-start))
        for targ in range(self.num_target_times-1, -1, -1):
            t1 = targ * self.num_times_disc
            t2 = (targ + 1) * self.num_times_disc + 1
            flip = numpy.flipud(range(t1,t2))
            start_i = time.time()
            intv = semiLagrangian.integrate(rg, N, len(flip), \
                                self.v[:,:,flip], 3, self.dt)
            logging.info("inner integrate: %f" % (time.time()-start_i))
            start_i = time.time()
            h2 = intv
            flip2 = numpy.flipud(range(len(flip)))
            h2 = h2[:,:,flip2]
            self.h2[:,:,t1:t2] = h2[:,:]
            if t2==T:
                self.I_interp[:,T-1] = self.rg.grid_interpolate(self.I[:,0], \
                                    self.h[:,:,T-1]).real
                p1 = 2./numpy.power(self.sigma,2) * \
                                (self.I_interp[:,T-1]-self.I[:,T-1])
            else:
                self.I_interp[:,t2-1] = self.rg.grid_interpolate(self.I[:,0], \
                                    self.h[:,:,t2-1]).real
                p1 = self.p[:, t2-1] - 2./numpy.power(self.sigma,2) * \
                                (self.I[:,t2-1] - self.I_interp[:,t2-1])
            logging.info("other stuff: %f" % (time.time()-start_i))
            start_i = time.time()
            for t in range(t1, t2-1):
                if self.dim==2:
                    gh1 = rg.gradient(self.h2[:,0,t])
                    gh2 = rg.gradient(self.h2[:,1,t])
                    dj[:,t] = (gh1[:,0] * gh2[:,1] - gh2[:,0] * gh1[:,1]).real
                elif self.dim==3:
                    gh1 = rg.gradient(self.h2[:,0,t]).real
                    gh2 = rg.gradient(self.h2[:,1,t]).real
                    gh3 = rg.gradient(self.h2[:,2,t]).real
                    m1 = gh2[:,1]*gh3[:,2] - gh3[:,1]*gh2[:,2]
                    m2 = gh2[:,0]*gh3[:,2] - gh3[:,0]*gh2[:,2]
                    m3 = gh2[:,0]*gh3[:,1] - gh3[:,0]*gh2[:,1]
                    dj[:,t] = gh1[:,0]*m1 - gh1[:,1]*m2 + gh1[:,2]*m3
                self.p[:,t] = dj[:,t] * self.rg.grid_interpolate(p1, \
                                self.h2[:,:,t]).real
            logging.info("inner loop: %f" % (time.time()-start_i))
        logging.info("update evo: %f" % (time.time() - start))
        #self.writeData("debug")

    def writeData(self, name):
        rg, N, T = self.get_sim_data()
        grd = self.getGradient()
        for t in range(T):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.v[:,:,t], "v")
            h = numpy.vstack((self.h[:,0,t], self.h[:,1,t], self.h[:,2,t])).T
            rg.add_vtk_point_data(h[:,:], "h")
            h2 = numpy.vstack((self.h2[:,0,t], self.h2[:,1,t], self.h2[:,2,t])).T
            rg.add_vtk_point_data(h2[:,:], "h2")
            rg.add_vtk_point_data(self.I[:,t], "I")
            #rg.add_vtk_point_data(dj[:,t], "dj")
            rg.add_vtk_point_data(self.I_interp[:,t], "I_interp")
            rg.add_vtk_point_data(self.I[:,t]-self.I_interp[:,t], "diff")
            gI = rg.gradient(self.I_interp[:,t])
            rg.add_vtk_point_data(gI.real, "gI")
            rg.add_vtk_point_data(self.p[:,t], "p")
            rg.add_vtk_point_data(grd[:,:,t], "grad")
            rg.add_vtk_point_data(self.mu[:,:,t], "mu")
            rg.vtk_write(t, name, output_dir=self.output_dir)

    def getVariable(self):
        return self

    def objectiveFun(self):
        rg, N, T = self.get_sim_data()
        obj = 0.
        term2 = 0.
        for t in range(T-1):
            obj += self.dt * self.kernel_norm_V(self.mu[:,:,t])
        for t in range(self.num_target_times):
            term2 += numpy.dot(self.I[:,(t+1)*self.num_times_disc]
                    - self.I_interp[:,(t+1)*self.num_times_disc],
                    rg.integrate_dual(self.I[:,(t+1)*self.num_times_disc]
                    - self.I_interp[:,(t+1)*self.num_times_disc]))
        total_fun = obj + 1./numpy.power(self.sigma,2) * term2
        logging.info("term1: %f, term2: %f, tot: %f" % (obj, term2, total_fun))
        return total_fun

    def updateTry(self, direction, eps, objRef=None):
        rg, N, T = self.get_sim_data()
        self.last_dir = eps * direction
        mu_old = self.mu.copy()
        for t in range(T):
            self.mu[:,:,t] = self.mu_state[:,:,t] - direction[:,:,t] * eps
        self.update_evolutions()
        objTry = self.objectiveFun()
        if (objRef != None) and (objTry > objRef):
            self.mu = mu_old
            self.update_evolutions()
        return objTry

    def acceptVarTry(self):
        rg, N, T = self.get_sim_data()
        for t in range(T):
            self.mu_state[:,:,t] = self.mu[:,:,t].copy()
        #self.mu_state = self.mu.copy()
        #self.update_evolutions()

    def getGradient(self, coeff=1.0):
        rg, N, T = self.get_sim_data()
        grad = numpy.zeros((rg.num_nodes, 3, T))
        for t in range(T):
            gI_interp = rg.gradient(self.I_interp[:,t]).real
            #gtemp = rg.gradient(self.I[:,0]).real
            #gtemp0 = self.rg.grid_interpolate(gtemp[:,0], self.h[:,:,t]).real
            #gtemp1 = self.rg.grid_interpolate(gtemp[:,1], self.h[:,:,t]).real
            #gh1 = rg.gradient(self.h2[:,0,t])
            #gh2 = rg.gradient(self.h2[:,1,t])
            #gtemp[:,0] = gtemp0 * gh1[:,0] + gtemp0 * gh1[:,1]
            #gtemp[:,1] = gtemp1 * gh2[:,1] + gtemp1 * gh2[:,1]
            #gI_interp = gtemp.copy()
            grad[:,0,t] = rg.integrate_dual((2*self.mu[:,0,t] - self.p[:,t] * \
                                gI_interp[:,0]))
            grad[:,1,t] = rg.integrate_dual((2*self.mu[:,1,t] - self.p[:,t] * \
                                gI_interp[:,1]))
            grad[:,2,t] = rg.integrate_dual((2*self.mu[:,2,t] - self.p[:,t] * \
                                gI_interp[:,2]))

            #rg.create_vtk_sg()
            #rg.add_vtk_point_data(grad[:,:,t], "grad")
            #rg.add_vtk_point_data(gI_interp, "gI_interp")
            #rg.vtk_write(t, "grad_test", output_dir=self.output_dir)
        return coeff * self.dt * grad

    def computeMatching(self):
        conjugateGradient.cg(self, True, maxIter = 500, TestGradient=False,
                                epsInit=1e-3)
        return self

    def dotProduct(self, g1, g2):
        rg, N, T = self.get_sim_data()
        res = numpy.zeros(len(g2))
        for t in range(T):
            for ll in range(len(g2)):
                gr = g2[ll]
                kgr = self.apply_kernel_V(gr[:,:,t])
                res[ll] += self.dt * numpy.dot(g1[:,0,t], \
                                    rg.integrate_dual(kgr[:,0]))
                res[ll] += self.dt * numpy.dot(g1[:,1,t], \
                                    rg.integrate_dual(kgr[:,1]))
                res[ll] += self.dt * numpy.dot(g1[:,2,t], \
                                    rg.integrate_dual(kgr[:,2]))
        return res

    def endOfIteration(self):
        self.optimize_iteration += 1
        self.writeData("iter%d" % (self.optimize_iteration))

def initialize_v(rg, N, T):
    dt = 1./(T-1)
    # initialze the identity maps
    #for t in range(T):
    #  self.id_x[:,t] = rg.nodes[:,0]
    #  self.id_y[:,t] = rg.nodes[:,1]
    v0 = numpy.zeros((len(rg.nodes), 3, T))
    #v1 = 2.0 * numpy.ones((len(rg.nodes), 3, T))
    v1 = numpy.zeros((len(rg.nodes), 3, T))
    v = numpy.zeros((len(rg.nodes), 3, T))
    v2 = numpy.zeros((len(rg.nodes), 3, T))
    #set1 = numpy.array(r>=2).astype(int)
    #set2 = numpy.array(r<5).astype(int)
    #set3 = ((set1-set2)==0)
    for t in range(T):
      loc = -2 + 4*dt * t
      x_sqr = numpy.power(rg.nodes[:,0]-loc, 2)
      y_sqr = numpy.power(rg.nodes[:,1], 2)
      #r = numpy.sqrt(r_sqr)
      v[:,0,t] = 4 * numpy.exp(-.5*(.1*x_sqr + .1*y_sqr))
      v[:,1,t] = 1e-30 * numpy.exp(-.5*(.1*x_sqr + .1*y_sqr))
      v1[:,0,t] = 4
      #v2[set3,0,t] = -(2./3.)*(r[set3]-5)
    return v

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
    output_directory_base = "/cis/home/clr/compute/time_series/"
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
    its = ImageTimeSeries(output_dir)
    its.computeMatching()
