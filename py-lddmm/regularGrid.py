
import logging
import numpy
import numpy.fft
import scipy.integrate
import scipy.linalg
import scipy.sparse
import scipy.interpolate
from tvtk.api import tvtk


def meshgrid2(arrs):
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = numpy.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans[::-1])


class RegularGrid(object):

    def __init__(self, dim, num_points=None, domain_max=None, dx=None, \
                        mesh_name="test", mesh_only=False):
        assert dim==2 or dim==3, "Dimension must be 2 or 3."
        self.dim = dim
        self.num_points = numpy.empty(self.dim).astype(int)
        self.dx = numpy.empty(self.dim)
        self.domain_max = numpy.empty(self.dim)
        self.dk = numpy.empty(self.dim)
        for d in range(dim):
            if dx == None:
                self.num_points[d] = num_points[d]
                self.domain_max[d] = domain_max[d]
                self.dx[d] = (2.0*self.domain_max[d])/(self.num_points[d])
                self.dk[d] = (2.0*numpy.pi)/((self.num_points[d]) * self.dx[d])
            elif domain_max == None:
                self.num_points[d] = num_points[d]
                self.dx[d] = dx[d]
                self.domain_max[d] = self.dx[d] * self.num_points[d] / 2.0
                self.dk[d] = (2.0*numpy.pi)/((self.num_points[d]) * self.dx[d])
            elif num_points == None:
                self.domain_max[d] = domain_max[d]
                self.dx[d] = dx[d]
                self.num_points[d] = (2.0*self.domain_max[d])/self.dx[d]
                self.dk[d] = (2.0*numpy.pi)/((self.num_points[d]) * self.dx[d])
        self.nodes = None
        self.vtk_dim = None
        self.x = None
        self.y = None
        self.z = None
        self.xsi_1 = None
        self.xsi_2 = None
        self.xsi_3 = None
        self.fourier_nodes = None
        self.mesh_name = mesh_name
        self.sg = None
        self.elements = None
        self.element_volumes = None
        self.fourier_element_areas = None
        self.boundary_nodes = None
        self.edge_nodes = None
        self.corner_nodes = None
        self.interior_nodes = None
        self.elements_of_node = None
        self.create(mesh_only)

    def __getGridZeros(self):
      return numpy.zeros((self.num_points, self.num_points))
    grid_zeros = property(__getGridZeros)

    def __associate(self, el_index, node_list, el_list):
        self.elements[:,el_index] = node_list
        self.elements_of_node[node_list,0] += 1
        self.elements_of_node[node_list, \
                            self.elements_of_node[node_list,0]] = el_list

    def create(self, mesh_only):
        if self.dim == 2:
            self.resol = self.dx
            self.f_resol = self.dk
        elif self.dim == 3:
            self.resol = self.dx
            self.f_resol = self.dk

        xsi_copies = []
        x_copies = []

        for d in range(self.dim):
            xsi = numpy.zeros(self.num_points[d])
            x = numpy.zeros(self.num_points[d])
            for j in range(self.num_points[d]):
              if j <= self.num_points[d]/2:
                xsi[j] = j * self.dk[d]
                x[j] = j * self.dx[d]
              else:
                xsi[j] = -1.*xsi[self.num_points[d]-j]
                x[j] = -1.*x[self.num_points[d]-j]
            xsi_s = numpy.fft.fftshift(xsi)
            x_s = numpy.fft.fftshift(x)
            xsi_s[0] *= -1.0
            x_s[0] *= -1.0
            xsi_copies.append(xsi_s)
            x_copies.append(x_s)
        temp_xsi = meshgrid2(xsi_copies)
        temp_x = meshgrid2(x_copies)
        self.dims = temp_x[0].shape

        self.nodes = numpy.zeros(temp_x[0].shape + (3,), dtype=float)
        self.fourier_nodes = numpy.zeros(temp_xsi[0].shape + (3,), \
                            dtype=float)

        for d in range(self.dim):
            self.nodes[...,d] = temp_x[d]
            self.fourier_nodes[...,d] = temp_xsi[d]
        self.nodes.shape = self.nodes.size/3, 3
        self.fourier_nodes.shape = self.fourier_nodes.size/3, 3

        # these are convenient accessor to node and fourier data
        self.x = temp_x[0].copy()
        self.y = temp_x[1].copy()
        self.xsi_1 = temp_xsi[0]
        self.xsi_2 = temp_xsi[1]
        if (self.dim==3):
            self.z = temp_x[2].copy()
            self.xsi_3 = temp_xsi[2].copy()
            self.vtk_dim = self.num_points.copy()
        else:
            self.z = numpy.zeros_like(temp_x[0])
            self.xsi_3 = numpy.zeros_like(temp_xsi[0])
            self.vtk_dim = (self.num_points[0], self.num_points[1], 1)

        self.num_nodes = len(self.nodes)
        node_list = numpy.array(range(len(self.nodes)))
        self.elements_of_node = -1 * numpy.ones((self.num_nodes, 9)).astype(int)
        self.elements_of_node[:,0] = 0
        if not mesh_only:
            self.num_elements = 1
            for d in range(self.dim):
                self.num_elements *= self.num_points[d]-1
            temp = numpy.arange(self.num_nodes).astype(int)
            temp2 = temp.reshape(self.dims)
            el_list = numpy.arange(self.num_elements).astype(int)
            if self.dim==2:
                self.elements = -1 * numpy.ones((self.num_elements, 4)).astype(int)
                temp3 = temp2.copy()
                temp2 = temp2[0:self.num_points[1]-1, 0:self.num_points[0]-1].ravel()
                self.__associate(0, temp2, el_list)
                self.__associate(1, temp2+1, el_list)
                self.__associate(2, temp2+self.num_points[0]+1, el_list)
                self.__associate(3, temp2+self.num_points[0], el_list)
            elif self.dim==3:
                self.elements = numpy.zeros((self.num_elements, 8)).astype(int)
                temp3 = temp2.copy()
                temp2 = temp2[0:self.num_points[2]-1, 0:self.num_points[1]-1, \
                                    0:self.num_points[0]-1].ravel()
                nsqr = self.num_points[0]*self.num_points[1]
                self.__associate(0, temp2, el_list)
                self.__associate(1, temp2+1, el_list)
                self.__associate(2, temp2+self.num_points[0]+1, el_list)
                self.__associate(3, temp2+self.num_points[0], el_list)
                self.__associate(4, temp2+nsqr, el_list)
                self.__associate(5, temp2+1+nsqr, el_list)
                self.__associate(6, temp2+self.num_points[0]+1+nsqr, el_list)
                self.__associate(7, temp2+self.num_points[0]+nsqr, el_list)
            if self.dim==2:
                max_els = 4
            elif self.dim==3:
                max_els = 8
            self.interior_nodes = numpy.where( \
                                self.elements_of_node[:,0]==max_els)[0]
            self.edge_nodes = numpy.setdiff1d(node_list, self.interior_nodes)
            self.element_volumes = numpy.ones(len(self.elements))
            for d in range(self.dim):
                self.element_volumes *= self.dx[d]
        logging.info("Mesh %s created." % (self.mesh_name))
        logging.info("dx: %s." % (self.dx))


    def create_vtk_sg(self):
        self.sg = tvtk.StructuredGrid(dimensions=self.vtk_dim, \
                                points=self.nodes)

    def create_vtk_fourier_sg(self):
        self.sg = tvtk.StructuredGrid(dimensions=self.vtk_dim, \
                                points=self.fourier_nodes)

    def add_vtk_point_data(self, var, name, as_scalars=False, as_vectors=False):
        if as_scalars:
            self.sg.point_data.scalars = var
            self.sg.point_data.scalars.name = name
            return
        if as_vectors:
            self.sg.point_data.vectors = var
            self.sg.point_data.vectors.name = name
            return
        arr_id = self.sg.point_data.add_array(var)
        self.sg.point_data.get_array(arr_id).name = name

    def vtk_write(self, iter, mesh_name="", output_dir="."):
      if mesh_name == "":
        mesh_name = self.mesh_name
      w = tvtk.XMLStructuredGridWriter(input=self.sg, file_name="%s/%s_mesh%d_%d.vts" % (output_dir, mesh_name, self.num_points[0], iter))
      w.write()
      self.sg = None

    def conv(self, f,s):
        return s * self.num_points + f

    def neighbors(self, node):
        assert(not node in self.boundary_nodes)
        c = node
        w = node - 1
        e = node + 1
        s = node - self.num_points
        n = node + self.num_points
        if self.dim == 2:
            return numpy.array([c,w,e,s,n,c,c]).astype(int)
        elif self.dim==3:
            u = node + self.num_points**2
            d = node - self.num_points**2
            return numpy.array([c,w,e,s,n,d,u]).astype(int)

    def grid_neighbors(self):
        # return neighbor lists at interior nodes
        interior = self.interior_nodes
        w = numpy.zeros(len(self.nodes)).astype(int)
        e = numpy.zeros(len(self.nodes)).astype(int)
        s = numpy.zeros(len(self.nodes)).astype(int)
        n = numpy.zeros(len(self.nodes)).astype(int)
        d = numpy.zeros(len(self.nodes)).astype(int)
        u = numpy.zeros(len(self.nodes)).astype(int)
        w[interior] = interior - 1
        e[interior] = interior + 1
        s[interior] = interior - self.num_points[0]
        n[interior] = interior + self.num_points[0]
        c = numpy.arange(self.num_nodes)
        if self.dim==2:
            return [w, e, s, n, c, c]
        elif self.dim==3:
            nsqr = self.num_points[0] * self.num_points[1]
            d[interior] = interior - nsqr
            u[interior] = interior + nsqr
            return [w, e, s, n, d, u]

    def barycentric_coordinates(self, point):
        x = point[0]
        y = point[1]
        eps = 1e-8
        test1 = (self.nodes[self.elements[:,0],0] - eps <= x).astype(int)
        test2 = (self.nodes[self.elements[:,1],0] + eps >= x).astype(int)
        test3 = (self.nodes[self.elements[:,2],1] + eps >= y).astype(int)
        test4 = (self.nodes[self.elements[:,0],1] - eps <= y).astype(int)
        test = test1 + test2 + test3 + test4
        el = numpy.where(test==4)[0][0]
        el_nodes = self.nodes[self.elements[el]]
        bary = numpy.zeros(2)
        bary[0] = (x - el_nodes[0,0])/self.dx
        bary[1] = (y - el_nodes[0,1])/self.dx
        return [el, bary]

    def raised_cosine(self, mf, beta):
      f_x = 1./(2.0*numpy.pi) * self.xsi_1
      f_y = 1./(2.0*numpy.pi) * self.xsi_2
      if mf == -1:
        f_max = numpy.max(f_x)
        T = (1+beta)/(2*f_max)
      else:
        f_cut = 1./(2.0*numpy.pi) * numpy.sqrt(numpy.sqrt(1./mf) - 1)
        T = 1./(2*f_cut)
        f_max = numpy.max(f_x)
        roll_max = (1+beta)/(2*T)
        if roll_max >= f_max:
          logging.info("warning: rollover exceeds max frequency.")
          suggest_beta = f_max * 2*T - 1.
          logging.info("suggest beta: %f." % (suggest_beta))

      outer_cut_x = self.grid_zeros
      outer_cut_y = self.grid_zeros
      inner = numpy.abs(f_x) <= (1-beta)/ (2.*T)
      mid1 = numpy.abs(f_x) > (1-beta) / (2.*T)
      mid2 = numpy.abs(f_x) < (1+beta) / (2.*T)
      mid = (mid1.astype(int) + mid2.astype(int)) == 2
      outer_cut_x[inner] = T
      outer_cut_x[mid] = (T/2.) * (1 + numpy.cos((numpy.pi*T)/beta*(numpy.abs(f_x) - (1-beta)/(2*T))))[mid]
      inner = numpy.abs(f_y) <= (1-beta)/ (2.*T)
      mid1 = numpy.abs(f_y) > (1-beta) / (2.*T)
      mid2 = numpy.abs(f_y) < (1+beta) / (2.*T)
      mid = (mid1.astype(int) + mid2.astype(int)) == 2
      outer_cut_y[inner] = T
      outer_cut_y[mid] = (T/2.) * (1 + numpy.cos((numpy.pi*T)/beta*(numpy.abs(f_y) - (1-beta)/(2*T))))[mid]
      outer_cut =  (1./T) * outer_cut_x * (1./T) * outer_cut_y
      return outer_cut

    def interpolate(self, f_list, x, y):
        interp_list = []
        for f in f_list:
          interp_list.append(numpy.zeros_like(f))
        for j in range(len(self.nodes)):
          el, bary = self.barycentric_coordinates(numpy.array([x[j], y[j]]))
          el_nodes = self.elements[el,:]
          for k in range(len(interp_list)):
            f = f_list[k]
            interp_list[k][j] = f[el_nodes[0]]*bary[0]*bary[1] + f[el_nodes[1]]*(1-bary[0])*bary[1] + f[el_nodes[2]]*(1-bary[0])*(1-bary[1]) + f[el_nodes[3]]*bary[0]*(1-bary[1])
        return interp_list

    def interpolate_points(self, f, points):
        f_interp = numpy.zeros(len(points))
        for j in range(len(points)):
            el, bary = self.barycentric_coordinates(points[j,:])
            el_nodes = self.elements[el, :]
            bary[0] = 1.-bary[0]
            bary[1] = 1.-bary[1]
            f_interp[j] = f[el_nodes[0]]*bary[0]*bary[1] + \
                            f[el_nodes[1]]*(1-bary[0])*bary[1] + \
                            f[el_nodes[2]]*(1-bary[0])*(1-bary[1]) + \
                            f[el_nodes[3]]*bary[0]*(1-bary[1])
        return f_interp

    def divergence(self, v):
      w, e, s, n = self.grid_neighbors()
      dxv1 = (v[e,0] - v[w,0])/(2.0*self.dx)
      dyv2 = (v[n,1] - v[s,1])/(2.0*self.dx)
      div = dxv1 + dyv2
      return div

    def gradient(self, f):
      w, e, s, n, d, u = self.grid_neighbors()
      gf = numpy.zeros((len(f), 3)).astype(complex)
      gf[:,0] = (f[e] - f[w])/(2.0*self.dx[0])
      gf[:,1] = (f[n] - f[s])/(2.0*self.dx[1])
      if self.dim==3:
          gf[:,2] = (f[u] - f[d])/(2.0*self.dx[2])
      return gf

    def grid_interpolate2(self, f, pts):
        interp = scipy.interpolate.griddata(self.nodes, f, pts, \
                             method='linear')
        return interp

    def grid_interpolate(self, f, pts):
        if self.dim==2:
            return self.grid_interpolate2d(f, pts)
        elif self.dim==3:
            return self.grid_interpolate3d(f, pts)

    def grid_interpolate2d(self, f, pts):
        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]

        N = self.num_points
        (indexx, indexy) = meshgrid2((range(N[0]), range(N[1])))
        indexx = indexx.astype(int)
        indexy = indexy.astype(int)

        F = numpy.reshape(f.copy(), self.dims).astype(complex)
        X = numpy.reshape(x - self.nodes[:,0], self.dims)
        Y = numpy.reshape(y - self.nodes[:,1], self.dims)

        stepsx = X / self.dx[0]
        stepsy = Y / self.dx[1]

        px = numpy.floor(stepsx)
        py = numpy.floor(stepsy)

        ax = stepsx - px
        ay = stepsy - py

        pxindex = (indexx + px).astype(int)
        pyindex = (indexy + py).astype(int)
        pxindex_x = (pxindex + 1).astype(int)
        pyindex_y = (pyindex + 1).astype(int)

        pxindex[(pxindex<0)] = 0
        pyindex[(pyindex<0)] = 0
        pxindex_x[(pxindex_x<0)] = 0
        pyindex_y[(pyindex_y<0)] = 0

        pxindex[(pxindex>N[0]-1)] = N[0]-1
        pyindex[(pyindex>N[1]-1)] = N[1]-1
        pxindex_x[(pxindex_x>N[0]-1)] = N[0]-1
        pyindex_y[(pyindex_y>N[1]-1)] = N[1]-1

        pindex = (pxindex + (N[0])*(pyindex)).astype(int)
        pindex_x = (pxindex_x + (N[0])*(pyindex)).astype(int)
        pindex_y = (pxindex + (N[0])*(pyindex_y)).astype(int)
        pindex_xy = (pxindex_x + (N[0])*(pyindex_y)).astype(int)

        F[:,:] = f[pindex]*(1-ax)*(1-ay) + f[pindex_x]*(ax)*(1-ay) + f[pindex_y]*(1-ax)*(ay) + f[pindex_xy]*(ax)*(ay)
        return numpy.reshape(F[:,:], N[0]*N[1])

    def grid_interpolate3d(self, f, pts):
        diffpts = pts - self.nodes
        x = diffpts[:,0]
        y = diffpts[:,1]
        z = diffpts[:,2]
        N = self.num_points

        (indexx, indexy, indexz) = meshgrid2( \
                            (range(N[0]), range(N[1]), range(N[2])))
        indexx = indexx.astype(int)
        indexy = indexy.astype(int)
        indexz = indexz.astype(int)

        F = numpy.reshape(f.copy(), self.dims)
        X = numpy.reshape(x, self.dims)
        Y = numpy.reshape(y, self.dims)
        Z = numpy.reshape(z, self.dims)

        stepsx = X / self.dx[0]
        stepsy = Y / self.dx[1]
        stepsz = Z / self.dx[2]

        px = numpy.floor(stepsx)
        py = numpy.floor(stepsy)
        pz = numpy.floor(stepsz)

        ax = stepsx - px
        ay = stepsy - py
        az = stepsz - pz

        pxindex = (indexx + px).astype(int)
        pyindex = (indexy + py).astype(int)
        pzindex = (indexz + pz).astype(int)
        pxindex_x = (pxindex + 1).astype(int)
        pyindex_y = (pyindex + 1).astype(int)
        pzindex_z = (pzindex + 1).astype(int)

        pxindex[(pxindex<0)] = 0
        pyindex[(pyindex<0)] = 0
        pxindex_x[(pxindex_x<0)] = 0
        pyindex_y[(pyindex_y<0)] = 0
        pzindex[(pzindex<0)] = 0
        pzindex_z[(pzindex_z<0)] = 0

        pxindex[(pxindex>N[0]-1)] = N[0]-1
        pyindex[(pyindex>N[1]-1)] = N[1]-1
        pxindex_x[(pxindex_x>N[0]-1)] = N[0]-1
        pyindex_y[(pyindex_y>N[1]-1)] = N[1]-1
        pzindex[(pzindex>N[2]-1)] = N[2]-1
        pzindex_z[(pzindex_z>N[2]-1)] = N[2]-1

        nsqr = N[0] * N[1]
        pindex = (pxindex + (N[0])*(pyindex) + nsqr*pzindex).astype(int)
        pindex_x = (pxindex_x + (N[0])*(pyindex) + nsqr*pzindex).astype(int)
        pindex_y = (pxindex + (N[0])*(pyindex_y) + nsqr*pzindex).astype(int)
        pindex_xy = (pxindex_x + (N[0])*(pyindex_y) + nsqr*pzindex).astype(int)

        pindex_z = (pxindex + (N[0])*(pyindex) + nsqr*pzindex_z).astype(int)
        pindex_z_x = (pxindex_x + (N[0])*(pyindex) + nsqr*pzindex_z).astype(int)
        pindex_z_y = (pxindex + (N[0])*(pyindex_y) + nsqr*pzindex_z).astype(int)
        pindex_z_xy = (pxindex_x + (N[0])*(pyindex_y) + nsqr*pzindex_z).astype(int)

        F[:,:] = f[pindex]*(1-ax)*(1-ay)*(1-az) + \
                         f[pindex_x]*(ax)*(1-ay)*(1-az) + \
                         f[pindex_y]*(1-ax)*(ay)*(1-az) + \
                         f[pindex_xy]*(ax)*(ay)*(1-az)
        F[:,:] += f[pindex_z]*(1-ax)*(1-ay)*(az) + \
                         f[pindex_z_x]*(ax)*(1-ay)*(az) + \
                         f[pindex_z_y]*(1-ax)*(ay)*(az) + \
                         f[pindex_z_xy]*(ax)*(ay)*(az)
        return numpy.reshape(F[:,:], N[0]*N[1]*N[2])

    def inner_prod(self, f, g):
      f_c = .25 * (f[self.elements[:,0]] + f[self.elements[:,1]] + f[self.elements[:,2]] + f[self.elements[:,3]])
      g_c = .25 * (g[self.elements[:,0]] + g[self.elements[:,1]] + g[self.elements[:,2]] + g[self.elements[:,3]])
      integral = numpy.sum(f_c * g_c * self.element_volumes)
      return integral

    def integrate_dual(self, b):
        int_b = numpy.zeros_like(b)
        if self.dim==2:
            int_b[self.elements[:,0]] += .25 * b[self.elements[:,0]] * self.element_volumes
            int_b[self.elements[:,1]] += .25 * b[self.elements[:,1]] * self.element_volumes
            int_b[self.elements[:,2]] += .25 * b[self.elements[:,2]] * self.element_volumes
            int_b[self.elements[:,3]] += .25 * b[self.elements[:,3]] * self.element_volumes
        elif self.dim==3:
            for j in range(8):
                int_b[self.elements[:,j]] += numpy.power(.5,3) * \
                                    b[self.elements[:,j]] * self.element_volumes
        return int_b


