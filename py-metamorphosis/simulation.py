
import pdb
import numpy
import logging
import numpy.fft

class SolveHelper(object):

    def __init__(self, mm):
        self.mm = mm
        self.solve_iter = 0
        self.energy = 0.
        self.old_energy = 0.

    def solve_callback(self, sol_k):
        rg, N, T = self.mm.get_sim_data()
        full_state = self.mm.input_state.copy()
        full_state[1*rg.num_nodes:(T-1)*rg.num_nodes] = sol_k
        mult = self.kernel_mult(full_state, full=True)
        self.energy = 1./numpy.power(self.mm.sigma,2) * 1./self.mm.dt * numpy.dot(full_state, mult)
        logging.info("solve iteration %d:  energy %f" % (self.solve_iter, self.energy.real))
        self.solve_iter += 1

        if (self.solve_iter % self.mm.cg_max_iter == 0) or (self.solve_iter == 1):
          for t in range(T):
            n_t = full_state[t*rg.num_nodes:(t+1)*rg.num_nodes]
            dtv = rg.nodes - 1.0*self.mm.dt*self.mm.v[:,:,t]
            divv = rg.divergence(self.mm.v[:,:,t])
            if t<T-1:
              n_t1 = full_state[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes]
              in_t1 = rg.grid_interpolate(n_t1, dtv)
            else:
              n_t1 = n_t.copy()
              in_t1 = n_t.copy()
            rg.create_vtk_sg()
            kn = self.apply_kernel(n_t)
            rg.add_vtk_point_data(n_t.real.ravel(), "n_t")
            rg.add_vtk_point_data(kn.real.ravel(), "kn_t")
            rg.add_vtk_point_data(in_t1.real.ravel(), "in_t1")
            rg.add_vtk_point_data(n_t1.real.ravel(), "n_t1")
            rg.add_vtk_point_data((in_t1-n_t).real.ravel(), "diffn")
            rg.add_vtk_point_data((divv).real.ravel(), "divv")
            rg.add_vtk_point_data(self.mm.v[:,:,t], "v")
            rg.add_vtk_point_data(self.mm.rho[:,:,t].real, "rho")
            rg.vtk_write(t, "cg_space%d" % (self.solve_iter), output_dir=self.mm.iter_output_dir)
            fr = numpy.reshape(n_t, rg.dims)
            fr = numpy.fft.fftshift(fr)
            fnt = numpy.fft.fft2(fr * rg.element_volumes[0])
            fnt = numpy.fft.fftshift(fnt)
            rg.create_vtk_fourier_sg()
            rg.add_vtk_point_data(fnt.real.ravel(), "fnt")
            K = 1.0 / numpy.power(1 + (numpy.power(rg.xsi_1,2) + numpy.power(rg.xsi_2,2)),2)
            outer_cut = rg.raised_cosine(self.mm.mf, self.mm.beta)
            rg.add_vtk_point_data(K.real.ravel(), "K")
            rg.add_vtk_point_data(outer_cut.real.ravel(), "outer_cut")
            rg.vtk_write(t, "cg_fourier%d" % (self.solve_iter), output_dir=self.mm.iter_output_dir)

    def nonlinear_cg_callback(self, rho):
        rg, N, T = self.mm.get_sim_data()
        self.solve_iter += 1
        srho = numpy.reshape(rho, (rg.num_nodes, 3))
        sv = self.mm.apply_kernelv(srho)

    def apply_kernel(self, right, t=0):
        rg, N, T = self.mm.get_sim_data()
        i = numpy.complex(0,1)
        K = 1.0 / numpy.power(1 + self.mm.alpha*(numpy.power(rg.xsi_1,2) + numpy.power(rg.xsi_2,2)),2)
        K = numpy.fft.fftshift(K)
        rr = right.copy()
        fr = numpy.reshape(rr, rg.dims)
        fr = numpy.fft.fftshift(fr)
        fr = numpy.fft.fft2(fr * rg.element_volumes[0])
        if t>0 and True:
          rg.create_vtk_fourier_sg()
          rg.add_vtk_point_data(K.real.ravel(), "K")
          rg.add_vtk_point_data(fr.real.ravel(), "fr")
          rg.vtk_write(t, "k_test",  output_dir=self.mm.iter_output_dir)
          pdb.set_trace()
        fr = fr * K
        out = numpy.fft.ifft2(fr) * 1./rg.element_volumes[0]
        out = numpy.fft.fftshift(out)
        out = numpy.reshape(out, (rg.num_nodes))
        return out.copy()

    def kernel_mult(self, in_vec, full=True, ic=False):
        rg, N, T = self.mm.get_sim_data()
        i = numpy.complex(0,1)
        vec = numpy.zeros_like(in_vec).astype(complex)
        for t in range(0,T-1):
          n_t = in_vec[t*rg.num_nodes:(t+1)*rg.num_nodes].copy()
          n_t_1 = in_vec[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes].copy()
          dtv = rg.nodes - 1.0*self.mm.dt*self.mm.v[:,:,t]
          interp_n_1 = rg.grid_interpolate_dual_2d(n_t_1, dtv)
          divv = rg.divergence(self.mm.v[:,:,t])
          if (full or t>0):
            right = interp_n_1 - n_t
          else:
            right = interp_n_1
          if ic:
            right = -2.0*n_t
          right = self.apply_kernel(right)
          right = rg.integrate_dual(right)
          right = rg.grid_interpolate(right, dtv)
          vec[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes] += right
          if (full or t<T-2):
            right = -interp_n_1 + n_t
          else:
            right = n_t
          if ic:
            right = -2.0*interp_n_1
          right = self.apply_kernel(right)
          right = rg.integrate_dual(right)
          vec[t*rg.num_nodes:(t+1)*rg.num_nodes] += right
        return vec

    def solve_mult(self, v):
        rg, N, T = self.mm.get_sim_data()
        full_state = self.mm.input_state.copy()
        full_state[1*rg.num_nodes:(T-1)*rg.num_nodes] = v.copy()
        mult = self.kernel_mult(full_state, full=False)
        return 2.0*mult[1*rg.num_nodes:(T-1)*rg.num_nodes]

    def ev(self, rho):
        t = self.cgt
        rg, N, T = self.mm.get_sim_data()
        srho = numpy.reshape(rho, (rg.num_nodes, 3))
        sv = self.mm.apply_kernelv(srho)
        n_t = self.mm.n[t*rg.num_nodes:(t+1)*rg.num_nodes].copy()
        n_t_1 = self.mm.n[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes].copy()
        dtv = rg.nodes - 1.0*self.mm.dt*sv
        interp_n_1 = rg.grid_interpolate_dual_2d(n_t_1, dtv)
        It1 = n_t - interp_n_1
        norm_v = self.mm.kernel_normv(srho)
        energy2 = numpy.dot(It1, rg.integrate_dual(self.apply_kernel(It1)))
        energy =  self.mm.dt * (norm_v + 1./numpy.power(self.mm.sigma,2) * 1./numpy.power(self.mm.dt, 2)*energy2)
        return energy

    def grad_ev(self, rho):
        t = self.cgt
        rg, N, T = self.mm.get_sim_data()
        srho = numpy.reshape(rho, (rg.num_nodes, 3))
        sv = self.mm.apply_kernelv(srho)
        n_t = self.mm.n[t*rg.num_nodes:(t+1)*rg.num_nodes].copy()
        n_t_1 = self.mm.n[(t+1)*rg.num_nodes:(t+2)*rg.num_nodes].copy()
        dtv = rg.nodes - 1.0*self.mm.dt*sv
        interp_n_1 = rg.grid_interpolate_dual_2d(n_t_1, dtv)
        It1c = n_t - interp_n_1
        It1c *= 1./self.mm.dt
        It1 = self.apply_kernel(It1c)
        drn = rg.grid_interpolate_gradient_2d(It1, dtv)
        grad_energy = numpy.zeros((rg.num_nodes,3)).astype(complex)
        gft = rg.gradient(n_t * It1)
        grad_energy[:,0] = srho[:,0] + 1./numpy.power(self.mm.sigma,2) * (drn[:,0] * n_t_1)
        grad_energy[:,1] = srho[:,1] + 1./numpy.power(self.mm.sigma,2) * (drn[:,1] * n_t_1)
        grad_energy[:,2] = srho[:,2] + 1./numpy.power(self.mm.sigma,2) * (drn[:,2] * n_t_1)
        grad_energy *= self.mm.dt * 2.0
        grad_energy = numpy.reshape(grad_energy, (3*rg.num_nodes))
        return grad_energy












