#! /usr/bin/python

import logging
import os
import shutil
import numpy
import optparse

import regularGrid
import simulation
import problemSetup

import scipy.linalg.basic
import scipy.sparse.linalg
import scipy.optimize

import pdb

class MeasureMeta(object):

    def __init__(self, output_dir):
        # simulation parameters *******
        self.dim = 2
        self.sigma = .1
        self.num_points = (128,128)
        self.domain_max = (10.,10.)
        self.dx = None
        self.num_times = 21
        self.min_steps = 500
        self.beta = .4
        self.alpha = .05
        self.mf = .001
        self.cg_max_iter = 25
        self.ncg_max_iter = 25
        self.iter_output_dir = None
        self.output_dir = output_dir
        # ******************************
        self.rg = regularGrid.RegularGrid(self.dim, self.num_points, \
                            self.domain_max, self.dx, "meta")
        self.time_min = 0.
        self.time_max = 1.
        self.times = numpy.linspace(self.time_min, self.time_max, \
                            self.num_times)
        self.dt = (self.time_max - self.time_min) / (self.num_times - 1)
        self.template_diracs = None
        self.target_diracs = None
        self.input_state = None

        self.kernelv = self.get_kernelv()
        self.n0 = None
        self.n1 = None
        self.n = None
        logging.info("sigma: %f" % self.sigma)
        logging.info("num_points: %s" % str(self.rg.num_points))
        logging.info("domain_max: %s" % str(self.rg.domain_max))
        logging.info("num_times: %d" % self.num_times)
        logging.info("dx: %s" % str(self.rg.dx))
        logging.info("dt: %f" % self.dt)
        logging.info("beta: %f" % self.beta)
        logging.info("alpha: %f" % self.alpha)
        logging.info("mf: %f" % self.mf)
        logging.info("cg_max_iter: %d" % self.cg_max_iter)
        logging.info("ncg_max_iter: %d" % self.ncg_max_iter)

    def get_sim_data(self):
        return [self.rg, self.num_points, self.num_times]

    def delta(self, loc, mf, beta):
        rg, N, T = self.get_sim_data()
        i = complex(0,1)
        outer_cut = rg.raised_cosine(mf, beta)
        d1 = numpy.ones(rg.dims).astype(complex)
        loc_mod = numpy.exp(-1.0*i*(loc[0] * self.rg.xsi_1 + \
                             loc[1] * self.rg.xsi_2))
        d1 *= loc_mod
        #a = beta
        #r_sqr_xsi = numpy.power(rg.xsi_1,2) + numpy.power(rg.xsi_2,2)
        #cut2 = numpy.exp(-a/2. * r_sqr_xsi)
        d1 *= outer_cut
        #d1 *= cut2
        v_sym = numpy.fft.fftshift(d1)
        tvs = numpy.fft.ifft2(v_sym) * 1./rg.element_volumes[0]
        tvs = numpy.fft.fftshift(tvs)
        tvs = numpy.reshape(tvs, (rg.num_nodes))
        return tvs

    def initialize_diracs(self):
        rg, N, T = self.get_sim_data()
        self.template_diracs = []
        self.target_diracs = []
        sh = problemSetup.SetupHelper()
        sh.init_one(1, self.template_diracs, self.target_diracs)
        #sh.init_three_line()
        ##ds = sh.init_circle(self.template_diracs, 5., 3 * 12)
        #sh.init_circle(self.target_diracs, 6, 0, ds)
        #sh.init_open_square(self.template_diracs, 6, 6)
        ##sh.init_square(self.target_diracs, 8., 4 * 4)
        #sh.init_square(self.template_diracs, 6., 6)
        #sh.init_img_desc(self.template_diracs, self.target_diracs,
        #                    "../image-landmarks/eight1.dat",
        #                    "../image-landmarks/eight2.dat", self.domain_max)
        #sh.init_muct_face(self.template_diracs, "../muct-landmarks/face1.dat")
        #sh.init_muct_face(self.target_diracs, "../muct-landmarks/face2.dat")
        logging.info("%d in template, %d in target"
                    % (len(self.template_diracs), len(self.target_diracs)))
        logging.info("Setup boundary conditions.")
        self.n0 = numpy.zeros(rg.num_nodes).astype(complex)
        self.n1 = numpy.zeros(rg.num_nodes).astype(complex)
        for d in self.template_diracs:
          self.n0 += self.delta(d[0], self.mf, self.beta)
        for d in self.target_diracs:
          self.n1 += self.delta(d[0], self.mf, self.beta)
        self.n = numpy.zeros(rg.num_nodes * T).astype(complex)

    def initialize_v(self):
        rg, N, T = self.get_sim_data()
        # initialze the identity maps
        logging.info("Setting up initial v field.")
        v0 = numpy.zeros((len(rg.nodes), 3, T))
        #v1 = 2.0 * numpy.ones((len(rg.nodes), 3, T))
        v1 = numpy.zeros((len(rg.nodes), 3, T))
        v = numpy.zeros((len(rg.nodes), 3, T))
        v2 = numpy.zeros((len(rg.nodes), 3, T))
        #set1 = numpy.array(r>=2).astype(int)
        #set2 = numpy.array(r<5).astype(int)
        #set3 = ((set1-set2)==0)
        for t in range(T):
          loc = -2 + 4*self.dt * t
          x_sqr = numpy.power(rg.nodes[:,0]-loc, 2)
          y_sqr = numpy.power(rg.nodes[:,1], 2)
          #r = numpy.sqrt(r_sqr)
          v[:,0,t] = 4 * numpy.exp(-.5*(.1*x_sqr + .1*y_sqr))
          v[:,1,t] = 1e-30 * numpy.exp(-.5*(.1*x_sqr + .1*y_sqr))
          v1[:,0,t] = 4
          #v2[set3,0,t] = -(2./3.)*(r[set3]-5)
        self.v = v0

    def initialize_rho(self):
        rg, N, T = self.get_sim_data()
        rho = numpy.zeros((len(rg.nodes), 3, T)).astype(complex)
        rho_zero = rho.copy()
        for t in range(T):
          loc = 1.25*self.dt * t + -2
          vx = 4*self.delta((loc,0.), self.mf, self.beta)
          rho[:,0,t] = vx
        self.rho = rho_zero.copy()
        #self.rho = rho

    def get_kernelv(self):
        rg, N, T = self.get_sim_data()
        #a = .4
        a = .2
        b = 1./a
        r_sqr_xsi = (numpy.power(rg.xsi_1,2) + numpy.power(rg.xsi_2,2))
        #Kv = 2.*numpy.pi * b * numpy.exp(-b/2. * r_sqr_xsi)
        Kv = 1.0 / numpy.power(1 + 4 * self.alpha*(numpy.power(rg.xsi_1,2) + \
                            numpy.power(rg.xsi_2,2)),7)
        return Kv

    def apply_kernelv(self, right):
        rg, N, T = self.get_sim_data()
        krho = numpy.zeros((rg.num_nodes, 3))
        for j in range(2):
            rr = right[:,j].copy().astype(complex)
            fr = numpy.reshape(rr, rg.dims)
            fr = numpy.fft.fftshift(fr)
            fr = numpy.fft.fft2(fr * rg.element_volumes[0])
            #Kv = self.get_kernelv()
            Kv = self.kernelv
            Kv = numpy.fft.fftshift(Kv)
            fr = fr * Kv
            out = numpy.fft.ifft2(fr) * 1./rg.element_volumes[0]
            out = numpy.fft.fftshift(out)
            krho[:,j] = out.real.ravel()
        return krho

    def kernel_normv(self, right):
        rg, N, T = self.get_sim_data()
        kright = self.apply_kernelv(right)
        kn = 0.
        kn += numpy.dot(right[:,0], rg.integrate_dual(kright[:,0]))
        kn += numpy.dot(right[:,1], rg.integrate_dual(kright[:,1]))
        kn += numpy.dot(right[:,2], rg.integrate_dual(kright[:,2]))
        return kn

    def krho(self):
        rg, N, T = self.get_sim_data()
        for t in range(T):
          self.v[:,:,t] = self.apply_kernelv(self.rho[:,:,t])
        self.v = self.v.real

    def minimize_eta(self):
        rg, N, T = self.get_sim_data()
        logging.info("Applying kernel to momenta.")
        self.krho()
        self.input_state = numpy.zeros(rg.num_nodes * T).astype(complex)
        logging.info("Minimize eta.")
        self.input_state[0:rg.num_nodes] = self.n0.copy()
        self.input_state[(T-1)*rg.num_nodes:(T)*rg.num_nodes] = self.n1.copy()
        sh = simulation.SolveHelper(mm)
        logging.info("Setting up right hand side.")
        initial_mult = sh.kernel_mult(self.input_state, full=False, ic=True)
        rhs = -1.0 * initial_mult
        input_state2 = self.n[1*rg.num_nodes:(T-1)*rg.num_nodes].copy()
        logging.info("Solving system with cg.")
        linop = scipy.sparse.linalg.LinearOperator((len(input_state2), \
                            len(input_state2)), sh.solve_mult, dtype=complex)
        input_rhs = rhs[1*rg.num_nodes:(T-1)*rg.num_nodes]
        self.irhs = input_rhs.copy()
        sol = scipy.sparse.linalg.cg(linop, input_rhs, x0=input_state2, \
                            maxiter=self.cg_max_iter, \
                            callback=sh.solve_callback)[0]
        self.n = self.input_state.copy()
        self.n[1*rg.num_nodes:(T-1)*rg.num_nodes] = sol.copy()

    def minimize_rho(self):
        rg, N, T = self.get_sim_data()
        min_rho = numpy.zeros((rg.num_nodes, 3, T)).astype(complex)
        total_energy = 0.
        total_rho_energy = 0.
        for t in range(0,T-1):
          sh = simulation.SolveHelper(mm)
          sh.cgt = t
          logging.info("Nonlinear cg -- time: %d." % (sh.cgt))
          iv = numpy.reshape(self.rho[:,:,t], (3*rg.num_nodes))
          res, energy_t = self.nonlinear_cg(sh.ev, iv, fprime=sh.grad_ev)
          logging.info("V optimize; energy: %f." % (energy_t))
          min_rho[:,:,t] = numpy.reshape(res, (rg.num_nodes, 3))
          total_energy += energy_t
          total_rho_energy += self.dt * self.kernel_normv(min_rho[:,:,t])
        self.rho = min_rho.copy()
        total_edot_energy = numpy.power(self.sigma, 2) * \
                            (total_energy - total_rho_energy)
        return (total_energy, total_rho_energy, total_edot_energy)

    def nonlinear_cg(self, f, x0, fprime):
        rg, N, T = self.get_sim_data()
        k = 0
        resets = 0
        xk = x0.copy()
        old_fval = f(xk)
        start_energy = old_fval.copy()

        gfk = None
        gfk_old = None
        pk = None
        pk_old = None
        cont = True
        search_ok = True
        while(cont and (k < self.ncg_max_iter)):
          gfk = -1.0 * fprime(xk)
          if gfk_old == None:
            beta_k = 0.
          else:
            gfk1 = numpy.reshape(gfk, (rg.num_nodes, 3))
            gfk_old1 = numpy.reshape(gfk_old, (rg.num_nodes, 3))
            k1 = self.apply_kernelv(gfk1-gfk_old1)
            yk = numpy.dot(gfk1[:,0], rg.integrate_dual(k1[:,0]))
            yk += numpy.dot(gfk1[:,1], rg.integrate_dual(k1[:,1]))
            yk += numpy.dot(gfk1[:,2], rg.integrate_dual(k1[:,2]))
            k2 = self.apply_kernelv(gfk_old1)
            deltak = numpy.dot(gfk_old1[:,0], rg.integrate_dual(k2[:,0]))
            deltak += numpy.dot(gfk_old1[:,1], rg.integrate_dual(k2[:,1]))
            deltak += numpy.dot(gfk_old1[:,2], rg.integrate_dual(k2[:,2]))
            if deltak < 1e-15:
              beta_k = 0.
            else:
              beta_k = numpy.maximum(0, yk/deltak)
              if beta_k == 0:
                resets += 1
          if pk_old != None:
            pk = gfk + beta_k * pk_old
          else:
            pk = gfk.copy()
          old_old_fval = old_fval
          alpha_k, old_fval, search_ok = self.simple_line_search(f, xk, pk)
          if alpha_k == None:
            alpha_k = 0.
          search_ok =(alpha_k != None) and (alpha_k > 0.)
          if search_ok:
            xk = xk + alpha_k * pk
            gfk_old = gfk.copy()
            pk_old = pk.copy()
            k += 1
          else:
            logging.info("Nonlinear cg stopped; iter: %d; resets: %d" % \
                                 (k, resets))
            logging.info("energy change: %f" % (start_energy - old_fval))
            return (xk, old_old_fval)
        logging.info("Nonlinear cg reached max iterations; resets: %d." % \
                             (resets))
        logging.info("energy change: %f" % (start_energy - old_fval))
        return (xk, old_fval)

    def simple_line_search(self, f, x, pk):
        step = 10.
        energy0 = f(x)
        energy = 2*energy0
        while(( (energy - energy0) > -1e-10 ) and (step > 1e-10)):
          step *= .5
          x1 = x + step * pk
          energy = f(x1)
        if ((energy - energy0) > -1e-10):
          return (0, energy0, False)
        else:
          return (step, energy, True)

    def optimize(self):
        rg, N, T = self.get_sim_data()
        total_energy = -1.
        for min_step in range(0, self.min_steps):
          self.iter_output_dir = "%s/iter_%d" % (self.output_dir, min_step)
          os.mkdir(self.iter_output_dir)
          self.minimize_eta()
          old_total_energy = total_energy
          total_energy, total_rho_energy, total_edot_energy = \
                                  self.minimize_rho()
          logging.info("Saving final vtk data for outer iteration %d." % \
                              (min_step))
          for t in range(self.num_times):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.n[t*rg.num_nodes:(t+1)*rg.num_nodes].real.ravel(), "n_t")
            rg.add_vtk_point_data(self.v[:,:,t], "v")
            rg.add_vtk_point_data(self.rho[:,:,t].real, "rho")
            rg.vtk_write(t, output_dir=self.iter_output_dir)
          logging.info("*****************************************")
          logging.info("Iteration %d complete." % (min_step))
          logging.info("total energy: %f ; old total energy: %f." % (total_energy, old_total_energy))
          logging.info("total rho energy: %f." % (total_rho_energy))
          logging.info("total edot energy: %f." % (total_edot_energy))
          logging.info("*****************************************")

def setup_default_logging(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    fh = logging.FileHandler("%s/mm.log" % (output_dir))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if __name__ == "__main__":

    # set permanent options
    output_directory_base = "/cis/home/clr/compute/mm/"
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
    mm = MeasureMeta(output_dir)
    mm.initialize_diracs()
    mm.initialize_rho()
    mm.initialize_v()
    mm.optimize()



