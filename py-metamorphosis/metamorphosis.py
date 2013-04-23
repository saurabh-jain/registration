import metamorphosisConfig
import optparse
import os
import shutil
import logging
import loggingUtils
import numpy
import scipy
import scipy.sparse.linalg
import fftHelper
import multiprocessing
import conjugateGradient

class ImageOptimizeCallbacks(object):

    def __init__(self, meta):
        self.meta = meta
        self.n_state = meta.n.copy()
        self.n = meta.n.copy()

    def solveCallback(self, vec):
        pass

    def kernelMult(self, in_vec, full=True, ic=False):
        rg, N, T = self.meta.getSimData()
        vec = numpy.zeros_like(in_vec)
        v = self.meta.v
        dt = self.meta.dt
        for t in range(0,T-1):
          n_t = in_vec[t*rg.num_nodes:(t+1)*rg.num_nodes].copy()
          n_t_1 = in_vec[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes].copy()
          dtv = rg.nodes + 1.0*dt*v[:,:,t]
          interp_n_1 = rg.grid_interpolate_dual_2d(n_t_1, dtv).real
          #divv = rg.divergence(self.mm.v[:,:,t])
          if (full or t>0):
            right = interp_n_1 - n_t
          else:
            right = interp_n_1
          if ic:
            right = -2.0*n_t
          right = rg.integrate_dual(right)
          right = rg.grid_interpolate(right, dtv).real
          vec[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes] += right
          if (full or t<T-2):
            right = -interp_n_1 + n_t
          else:
            right = n_t
          if ic:
            right = -2.0*interp_n_1
          right = rg.integrate_dual(right)
          vec[t*rg.num_nodes:(t+1)*rg.num_nodes] += right
        return vec

    def solveMult(self, vec):
        rg, N, T = self.meta.getSimData()
        full_state = self.meta.input_state
        full_state[1*rg.num_nodes:(T-1)*rg.num_nodes] = vec.copy()
        mult = self.kernelMult(full_state, full=False)
        return 2.0*mult[1*rg.num_nodes:(T-1)*rg.num_nodes]

class Metamorphosis(object):
    def __init__(self, output_dir, config_name):
        self.num_points = None
        self.domain_max = None
        self.dx = None
        self.verbose_file_output = False
        self.output_dir = output_dir
        metamorphosisConfig.configure(self, config_name)
        self.optimize_iteration = 0

        self.n = numpy.zeros((self.rg.num_nodes, self.num_times))
        self.mu = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.mu_state = self.mu.copy()
        self.v = numpy.zeros((self.rg.num_nodes, 3, self.num_times))

        # initialize the V kernel multiplier
        i = complex(0,1)
        r_sqr_xsi = (numpy.power(i*self.rg.xsi_1,2) + \
                            numpy.power(i*self.rg.xsi_2,2) + \
                            numpy.power(i*self.rg.xsi_3,2))
        self.KernelV = 1.0 / numpy.power(self.gamma - self.alpha * (r_sqr_xsi), \
                            self.Lpower)

        self.n[...,0] = self.template_in
        self.n[...,self.num_times-1] = self.target_in

        self.cache_v = False
        self.pool = multiprocessing.Pool(self.pool_size)

    def optimizeN(self):
        (rg,N,T) = self.getSimData()
        logging.info("Minimize n.")
        self.input_state = self.n.ravel(order="F")
        logging.info("Setting up right hand side.")
        ioc = ImageOptimizeCallbacks(self)
        initial_mult = ioc.kernelMult(self.input_state, full=False, ic=True)
        rhs = -1.0 * initial_mult
        input_state2 = self.n.ravel()[1*rg.num_nodes:(T-1)*rg.num_nodes].copy()
        logging.info("Solving system with cg.")
        linop = scipy.sparse.linalg.LinearOperator((len(input_state2), \
                            len(input_state2)), ioc.solveMult, dtype=float)
        input_rhs = rhs[1*rg.num_nodes:(T-1)*rg.num_nodes]
        self.irhs = input_rhs.copy()
        sol = scipy.sparse.linalg.cg(linop, input_rhs, x0=input_state2, \
                            maxiter=self.cg_max_iter, \
                            callback=ioc.solveCallback)[0]
        self.n = self.input_state.copy()
        self.n[1*rg.num_nodes:(T-1)*rg.num_nodes] = sol.copy()
        self.n = numpy.reshape(self.n, (self.rg.num_nodes, self.num_times), order="F")

    def getSimData(self):
        return [self.rg, self.num_points, self.num_times]

    def kMu(self, async=False):
        rg, N, T = self.getSimData()
        if async:
            res = []
            for t in range(T):
                res.append(self.pool.apply_async(fftHelper.applyKernel, \
                        args=(self.mu[:,:,t].copy(), rg.dims, rg.num_nodes,\
                        self.KernelV, rg.element_volumes[0])))
            for t in range(T):
                self.v[:,0:self.dim,t] = res[t].get( \
                                    timeout=self.pool_timeout).real
        else:
            for t in range(T):
              self.v[:,0:self.dim,t] = fftHelper.applyKernel(self.mu[:,:,t], \
                              rg.dims, \
                              rg.num_nodes, self.KernelV, \
                              rg.element_volumes[0]).real
        self.v = self.v.real

    # **********************************************************************
    # Implementation of Callback functions for non-linear conjugate gradient
    # **********************************************************************
    def getVariable(self):
        return self

    def objectiveFun(self):
        rg, N, T = self.getSimData()
        obj = 0.
        term2 = 0.
        for t in range(T-1):
            kn = 0.
            kn += numpy.dot(self.mu[:,0,t], self.v[:,0,t])
            kn += numpy.dot(self.mu[:,1,t], self.v[:,1,t])
            kn += numpy.dot(self.mu[:,2,t], self.v[:,2,t])
            obj += self.dt * kn
            dtv = rg.nodes + self.dt * self.v[...,t]
            interp_n_1 = rg.grid_interpolate(self.n[:,t+1], dtv).real
            term2 += numpy.power(1./self.dt*(interp_n_1 - self.n[:,t]), 2).sum()
            term2 *= self.dt
        term2 *= rg.element_volumes[0]
        total_fun = obj + 1./numpy.power(self.sigma,2) * term2
        logging.info("term1: %e, term2: %e, tot: %e" % (obj, term2, total_fun))
        return total_fun

    def updateTry(self, direction, eps, objRef=None):
        rg, N, T = self.getSimData()
        self.last_dir = eps * direction
        mu_old = self.mu.copy()
        v_old = self.v.copy()
        self.mu = self.mu_state - direction * eps
        if not self.cache_v:
            self.kMu(async=True)
        else:
            self.v = self.v_state - self.cache_v_dir * eps
        objTry = self.objectiveFun()
        if (objRef != None) and (objTry > objRef):
            self.mu = mu_old
            self.v = v_old
        return objTry

    def acceptVarTry(self):
        rg, N, T = self.getSimData()
        self.mu_state = self.mu.copy()
        self.kMu(async=True)

    def getGradient(self, coeff=1.0):
        rg, N, T = self.getSimData()
        gE = numpy.zeros((rg.num_nodes, 3, T))
        for t in range(T-1):
            dtv = rg.nodes + self.v[...,t] * self.dt
            interp_grad = rg.grid_interpolate_gradient_2d(self.n[:,t+1], dtv).real
            interp_n_1 = rg.grid_interpolate(self.n[:,t+1], dtv).real
            diff = numpy.reshape(1./self.dt*(interp_n_1 - self.n[:,t]), \
                                (rg.num_nodes,1))
            gE[...,t] = self.mu[...,t] + self.sfactor * diff * interp_grad
        return coeff * gE

    def dotProduct(self, g1, g2):
        rg, N, T = self.getSimData()
        prod = numpy.zeros(len(g2))
        vol = rg.element_volumes[0]
        for ll in range(len(g2)):
            gr = g2[ll]
            res = []
            for t in range(T):
                res.append(self.pool.apply_async(fftHelper.applyKernel, \
                        args=(gr[:,:,t].copy(), rg.dims, rg.num_nodes,\
                        self.KernelV, vol)))
            for t in range(T):
                kgr = res[t].get(timeout=self.pool_timeout).real
                for d in range(self.dim):
                    prod[ll] += self.dt * numpy.dot(g1[:,d,t], kgr[:,d])
        return prod

    def endOfIteration(self):
        self.optimize_iteration += 1
#        #if (self.optimize_iteration % self.write_iter == 0):
#        #    self.writeData("iter%d" % (self.optimize_iteration))
    # ***********************************************************************
    # end of non-linear cg callbacks
    # ***********************************************************************

    def computeMatching(self):
        (rg,N,T) = self.getSimData()
        for iter in range(100):
            self.optimizeN()
            conjugateGradient.cg(self, True, maxIter = 25, TestGradient=False,\
                               epsInit=self.cg_init_eps)
            for t in range(T):
                rg.create_vtk_sg()
                rg.add_vtk_point_data(self.n[:,t], "n")
                rg.add_vtk_point_data(self.v[...,t], "v")
                rg.vtk_write(t, "test", output_dir=self.output_dir)
            import pdb
            pdb.set_trace()
        return self

if __name__ == "__main__":
    # set permanent options
    output_directory_base = metamorphosisConfig.compute_output_dir
    #output_directory_base = "./"
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
    loggingUtils.setup_default_logging(output_dir, metamorphosisConfig)
    logging.info(options)
    sim = Metamorphosis(output_dir, options.config_name)
    sim.computeMatching()
    sim.writeData("final")
