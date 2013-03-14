import regularGrid
import diffeomorphisms
import conjugateGradient
import numpy
import optparse
import os
import shutil
import logging
import time
import multiprocessing
import rg_fort
#import fftw3
import imageTimeSeriesConfig
import gradientDescent

import numexpr as ne

def apply_kernel_V_for_async(right, dims, num_nodes, Kv, el_vol):
    krho = numpy.zeros((num_nodes, 3))
    # for now assume 3 dimensions
    for j in range(3):
      rr = right[:,j].copy().astype(complex)
      fr = numpy.reshape(rr, dims)
      fr = numpy.fft.fftshift(fr)
      fr = numpy.fft.fftn(fr)
      Kv = numpy.fft.fftshift(Kv)
      fr = fr * Kv
      out = numpy.fft.ifftn(fr) * 1./el_vol
      out = numpy.fft.fftshift(out)
      krho[:,j] = numpy.reshape(out.real, (num_nodes))
    return krho

def apply_kernel_V_for_async_w(right, dims, num_nodes, Kv, el_vol):

    in_vec = numpy.zeros(dims, dtype=complex)
    fft_vec = numpy.zeros(dims, dtype=complex)
    out_vec = numpy.zeros(dims, dtype=complex)

    wfor = fftw3.Plan(in_vec, fft_vec, \
                            direction='forward', flags=['estimate'], \
                            )
    wback = fftw3.Plan(fft_vec, out_vec, \
                            direction='backward', flags=['estimate'], \
                            )
    krho = numpy.zeros((num_nodes, 3))
    for j in range(3):
        rr = right[:,j].copy().astype(complex)
        fr = numpy.reshape(rr, dims)
        fr = numpy.fft.fftshift(fr)
        fr *= el_vol
        in_vec[...] = fr[...]
        wfor.execute()
        fr[...] = fft_vec[...]
        Kv = numpy.fft.fftshift(Kv)
        fr = fr * Kv
        fft_vec[...] = fr[...]
        wback.execute()
        out = out_vec[...] / in_vec.size
        out *= 1/el_vol
        out = numpy.fft.fftshift(out)
        krho[:,j] = out.real.ravel()
    return krho


class ImageTimeSeries(object):

    def __init__(self, output_dir, config_name):
        # can override these in configuration scripts
        self.output_dir = output_dir
        self.write_iter = 25
        self.verbose_file_output = False
        imageTimeSeriesConfig.configure(self, config_name)

        # general configuration
        self.mu = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.mu_state = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.v = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.objTry = 0.
        self.mu_state = self.mu.copy()
        self.optimize_iteration = 0

        # initialize fftw information
#        self.fft_thread_count = 8
#        self.in_vec = numpy.zeros(self.rg.dims, dtype=complex)
#        self.fft_vec = numpy.zeros(self.rg.dims, dtype=complex)
#        self.out_vec = numpy.zeros(self.rg.dims, dtype=complex)
#        self.wfor = fftw3.Plan(self.in_vec, self.fft_vec, \
#                                direction='forward', flags=['measure'], \
#                                )
#        self.wback = fftw3.Plan(self.fft_vec, self.out_vec, \
#                                direction='backward', flags=['measure'], \
#                                )

        self.pool_size = 16
        self.pool = multiprocessing.Pool(self.pool_size)
        self.pool_timeout = 5000

        test_mu = numpy.zeros_like(self.mu[...,0])
        test_mu[1850,0] = 1.0
        test_v = apply_kernel_V_for_async(test_mu, self.rg.dims, \
                            self.rg.num_nodes, self.get_kernelv(), \
                            self.rg.element_volumes[0])
        self.rg.create_vtk_sg()
        self.rg.add_vtk_point_data(test_v.real, "test_v")
        self.rg.vtk_write(0, "kernel_test", self.output_dir)

        self.update_evolutions()

    def get_sim_data(self):
        return [self.rg, self.num_points, self.num_times]

    def get_kernelv(self):
        rg, N, T = self.get_sim_data()
        i = complex(0,1)
        r_sqr_xsi = (numpy.power(i*rg.xsi_1,2) + numpy.power(i*rg.xsi_2,2) + \
                            numpy.power(i*rg.xsi_3,2))
        Kv = 1.0 / numpy.power(self.gamma - self.alpha * (r_sqr_xsi), \
                            self.Lpower)
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

    def apply_kernel_V(self, right):
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

    def apply_kernel_V_numpy(self, right):
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

    def k_mu_async(self):
        rg, N, T = self.get_sim_data()
        Kv = self.get_kernelv()
        res = []
        for t in range(T):
            res.append(self.pool.apply_async(apply_kernel_V_for_async, \
                            args=(self.mu[:,:,t].copy(), rg.dims, rg.num_nodes,\
                            Kv, rg.element_volumes[0])))
        for t in range(T):
            self.v[:,:,t] = res[t].get(timeout=self.pool_timeout)

        self.v = self.v.real

    def k_mu(self):
        rg, N, T = self.get_sim_data()
        start = time.time()
        for t in range(T):
          self.v[:,:,t] = self.apply_kernel_V(self.mu[:,:,t])
        self.v = self.v.real
        logging.info("k_mu time: %f" % (time.time()-start))

    def update_evolutions(self):
        rg, N, T = self.get_sim_data()
        self.k_mu_async()
        self.I_interp[:,0] = self.I[:,0].copy()
        for t in range(T-1):
            vt = self.v[:,:,t]
            dt = self.dt
            w = ne.evaluate("-1.0 *  vt * dt")
            w = w.reshape((rg.dims[0], rg.dims[1], rg.dims[2], 3))
            self.I_interp[:,t+1] = rg_fort.interpolate_3d( \
                            self.I_interp[:,t], w, \
                            rg.num_points[0], rg.num_points[1], \
                            rg.num_points[2], \
                            rg.interp_mesh[0], \
                            rg.interp_mesh[1], rg.interp_mesh[2], \
                            rg.dx[0], rg.dx[1], rg.dx[2], rg.num_nodes,
                            rg.dims[0], rg.dims[1], \
                            rg.dims[2]).reshape(rg.num_nodes)
        if self.verbose_file_output:
            self.writeData("evolutions%d" % (self.optimize_iteration))

    def writeData(self, name):
        rg, N, T = self.get_sim_data()
        for t in range(T):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.v[:,:,t], "v")
            rg.add_vtk_point_data(self.I[:,t], "I")
            rg.add_vtk_point_data(self.I_interp[:,t], "I_interp")
            rg.add_vtk_point_data(self.I[:,t]-self.I_interp[:,t], "diff")
            rg.add_vtk_point_data(self.p[:,t], "p")
            rg.add_vtk_point_data(self.mu[:,:,t], "mu")
            rg.vtk_write(t, name, output_dir=self.output_dir)
            self.sc.data = self.I_interp[:,t]
            self.sc.saveAnalyze("%s/%s_I_%d" % (self.output_dir, name, \
                                 t), rg.num_points)

    def getVariable(self):
        return self

    def objectiveFun(self):
        rg, N, T = self.get_sim_data()
        obj = 0.
        term2 = 0.
        for t in range(T):
            if t<T-1:
                kn = 0.
                kn += numpy.dot(self.mu[:,0,t], self.v[:,0,t])
                kn += numpy.dot(self.mu[:,1,t], self.v[:,1,t])
                kn += numpy.dot(self.mu[:,2,t], self.v[:,2,t])
                obj += self.dt * kn
            if t in range(0, self.num_times, self.num_times_disc):
                term2 += numpy.power(self.I_interp[:,t] - self.I[:,t],2).sum()
        term2 *= rg.element_volumes[0]
        total_fun = obj + 1./numpy.power(self.sigma,2) * term2
        logging.info("term1: %e, term2: %e, tot: %e" % (obj, term2, total_fun))
        return total_fun

    def updateTry(self, direction, eps, objRef=None):
        rg, N, T = self.get_sim_data()
        self.last_dir = eps * direction
        mu_old = self.mu.copy()
        v_old = self.v.copy()
        Ii_old = self.I_interp.copy()
        for t in range(T):
            self.mu[:,:,t] = self.mu_state[:,:,t] - direction[:,:,t] * eps
        self.update_evolutions()
        objTry = self.objectiveFun()
        if (objRef != None) and (objTry > objRef):
            self.mu = mu_old
            self.v = v_old
            self.I_interp = Ii_old
        return objTry

    def acceptVarTry(self):
        rg, N, T = self.get_sim_data()
        for t in range(T):
            self.mu_state[:,:,t] = self.mu[:,:,t].copy()

    def getGradient(self, coeff=1.0):
        rg, N, T = self.get_sim_data()
        dt = self.dt
        self.p[...] = 0.
        vol = rg.element_volumes[0]
        gIi = numpy.empty((rg.num_nodes,3,T))
        pr = numpy.empty((rg.num_nodes,1,T))
        pr[:,0,T-1] = 2./numpy.power(self.sigma, 2) * vol * \
                                    (-self.I[:,T-1]+self.I_interp[:,T-1])
        for t in range(T-1,-1,-1):
            p1 = pr[:,0,t]
            if (t in range(0, self.num_times, self.num_times_disc)):
                if t!=T-1:
                    s = 2./numpy.power(self.sigma,2)
                    It = self.I[:,t]
                    Iit = self.I_interp[:,t]
                    p1 = ne.evaluate("(p1 - s * vol * (It - Iit))")
            v = self.v[:,:,t]
            v.shape = ((rg.dims[0], rg.dims[1], rg.dims[2], 3))
            (p_new, gI_interp) = rg_fort.interp_dual_and_grad( \
                            p1.reshape(rg.dims), self.I_interp[:,t], v, \
                            rg.num_points[0], rg.num_points[1], \
                            rg.num_points[2], \
                            rg.interp_mesh[0], \
                            rg.interp_mesh[1], rg.interp_mesh[2], \
                            rg.dx[0], rg.dx[1], rg.dx[2], dt, rg.num_nodes,
                            rg.dims[0], rg.dims[1], \
                            rg.dims[2])
            if t>0:
                pr[:,0,t-1] = p_new[...]
            gIi[...,t] = gI_interp[...]
        mu = self.mu
        grad = ne.evaluate("2*mu - pr * gIi")
        self.p = pr[:,0,:]
        retGrad = ne.evaluate("coeff * grad")
        if self.verbose_file_output:
            for t in range(T):
                rg.create_vtk_sg()
                rg.add_vtk_point_data(gIi[...,t], "gIi")
                rg.vtk_write(t, "gradE", self.output_dir)
        return retGrad

    def dotProduct(self, g1, g2):
        rg, N, T = self.get_sim_data()
        prod = numpy.zeros(len(g2))
        vol = rg.element_volumes[0]
        Kv = self.get_kernelv()
        for ll in range(len(g2)):
            gr = g2[ll]
            res = []
            for t in range(T):
                res.append(self.pool.apply_async(apply_kernel_V_for_async, \
                        args=(gr[:,:,t].copy(), rg.dims, rg.num_nodes,\
                        Kv, rg.element_volumes[0])))
            for t in range(T):
                kgr = res[t].get(timeout=self.pool_timeout)
                prod[ll] += self.dt * numpy.dot(g1[:,0,t], kgr[:,0])
                prod[ll] += self.dt * numpy.dot(g1[:,1,t], kgr[:,1])
                prod[ll] += self.dt * numpy.dot(g1[:,2,t], kgr[:,2])
        return prod

    def endOfIteration(self):
        self.optimize_iteration += 1
        if (self.optimize_iteration % self.write_iter == 0):
            self.writeData("iter%d" % (self.optimize_iteration))

    def endOptim(self):
        self.writeData("final")

    def computeMatching(self):
        conjugateGradient.cg(self, True, maxIter = 500, TestGradient=False, \
                           epsInit=self.cg_init_eps)
        #gradientDescent.descend(self, True, maxIter=1000, TestGradient=False,\
        #                    epsInit=self.cg_init_eps)
        return self

    def reset(self):
        from tvtk.api import tvtk
        rg, N, T = self.get_sim_data()
        fbase = "/cis/home/clr/compute/time_series/lung_data_1/iter250_mesh256_"
        for t in range(T):
            r = tvtk.XMLStructuredGridReader(file_name="%s%d.vts" % (fbase, t))
            r.update()
            self.v[...,t] = numpy.array(r.output.point_data.get_array("v")).astype(float)
            self.I[...,t] = numpy.array(r.output.point_data.get_array("I")).astype(float)
            self.I_interp[...,t] = numpy.array(r.output.point_data.get_array("I_interp")).astype(float)
            self.p[...,t] = numpy.array(r.output.point_data.get_array("p")).astype(float)
            self.mu[...,t] = numpy.array(r.output.point_data.get_array("mu")).astype(float)
            self.mu_state[...,t] = numpy.array(r.output.point_data.get_array("mu")).astype(float)
            logging.info("reloaded time %d." % (t))
        self.update_evolutions()

def setup_default_logging(output_dir, config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    log_file_name = config.log_file_name
    fh = logging.FileHandler("%s/%s" % (output_dir, log_file_name))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if __name__ == "__main__":
    output_directory_base = imageTimeSeriesConfig.compute_output_dir
    # set options from command line
    parser = optparse.OptionParser()
    parser.add_option("-o", "--output_dir", dest="output_dir")
    parser.add_option("-c", "--config_name", dest="config_name")
    (options, args) = parser.parse_args()
    output_dir = output_directory_base + options.output_dir
    # remove any old results in the output directory
    if os.access(output_dir, os.F_OK):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    setup_default_logging(output_dir, imageTimeSeriesConfig)
    logging.info(options)
    its = ImageTimeSeries(output_dir, options.config_name)
    #its.reset()
    its.computeMatching()
