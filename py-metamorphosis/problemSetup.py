import pickle
import numpy

class SetupHelper(object):
    def init_one(self, a, temp_list, targ_list):
        temp_list.append([numpy.array([-a,0.0,0]), 1.0])
        targ_list.append([numpy.array([a,0.0,0]), 1.0])

    def init_circle(self, dlist, r, num_points, ds=None):
        if ds == None:
          dtheta = (2*numpy.pi)/(num_points - 1)
          np = num_points
        else:
          dtheta = ds / r
          np = (2*numpy.pi)/dtheta + 1.
        ds = dtheta * r
        symbol_x = numpy.linspace(0, (2*numpy.pi-dtheta), np)
        ptsx = numpy.cos(symbol_x)*r
        ptsy = numpy.sin(symbol_x)*r
        for j in range(len(ptsx)):
          loc = numpy.array([ptsx[j], ptsy[j], 0.])
          dlist.append([loc, 1.0])
        return ds

    def init_img_desc(self, dlist1, dlist2, filename1, filename2, domain_max):
        f = open(filename1, 'r')
        obj = pickle.load(f)
        f2 = open(filename2, 'r')
        obj2 = pickle.load(f2)
        mx = .75 * domain_max
        m1 = numpy.max([numpy.max(obj[:,1]),numpy.max(obj[:,0])])
        m0 = numpy.min([numpy.min(obj[:,1]),numpy.min(obj[:,0])])
        m3 = numpy.max([numpy.max(obj2[:,1]),numpy.max(obj2[:,0])])
        m4 = numpy.min([numpy.min(obj2[:,1]),numpy.min(obj2[:,0])])
        m1 = numpy.max([m1, m3])
        m0 = numpy.min([m0, m4])
        obj *= 2.0 * mx / (m1-m0)
        obj2 *= 2.0 * mx / (m1-m0)
        m1 = numpy.max([numpy.max(obj[:,1]),numpy.max(obj[:,0])])
        m0 = numpy.min([numpy.min(obj[:,1]),numpy.min(obj[:,0])])
        m3 = numpy.max([numpy.max(obj2[:,1]),numpy.max(obj2[:,0])])
        m4 = numpy.min([numpy.min(obj2[:,1]),numpy.min(obj2[:,0])])
        m1 = numpy.max([m1, m3])
        m0 = numpy.min([m0, m4])
        obj -= (m1 - mx)
        obj2 -= (m1 - mx)
        for j in range(len(obj)):
          pt = obj[j,:]
          dlist1.append([pt, 1.0])
        for j in range(len(obj2)):
          pt = obj2[j,:]
          dlist2.append([pt, 1.0])
        return obj

    def init_muct_face(self, dlist, filename, domain_max):
        f = open(filename, 'r')
        obj = pickle.load(f)
        for j in range(len(obj)):
          pt = obj[j,:]
          if pt[0] != 0. and pt[1] != 0.:
            m = numpy.max(numpy.abs(obj))
            pt *=  .8 * domain_max / (m)
            pt[0] -= domain_max / 6.
            pt[1] += domain_max / 4.
            dlist.append([pt, 1.0])
        return obj

    def init_three_line(self, temp_list, targ_list):
        a = 2
        temp_list.append([numpy.array([-a,-3,0]), 1.0])
        temp_list.append([numpy.array([-a,0.0,0]), 1.0])
        temp_list.append([numpy.array([-a,3,0]), 1.0])
        targ_list.append([numpy.array([a,-3,0]), 1.0])
        targ_list.append([numpy.array([a,0,0]), 1.0])
        targ_list.append([numpy.array([a,3,0]), 1.0])

    def init_square(self, dlist, a, num_points):
        eps = 0.
        dlist.append([eps + numpy.array([-a/2.,-a/2,0]), 1.0])
        dlist.append([eps + numpy.array([-a/2.,a/2.,0]), 1.0])
        dlist.append([eps + numpy.array([a/2.,a/2.,0]), 1.0])
        dlist.append([eps + numpy.array([a/2.,-a/2.,0]), 1.0])
        x = numpy.linspace(-a/2, a/2, num_points)
        for j in range(1,len(x)-1):
          p = x[j]
          dlist.append([eps + numpy.array([a/2.,p,0]), 1.0])
          dlist.append([eps + numpy.array([-a/2.,p,0]), 1.0])
          dlist.append([eps + numpy.array([p,a/2,0]), 1.0])
          dlist.append([eps + numpy.array([p,-a/2,0]), 1.0])

    def init_open_square(self, dlist, a, num_points):
        dlist.append([numpy.array([-a/2.,-a/2,0]), 1.0])
        dlist.append([numpy.array([-a/2.,a/2.,0]), 1.0])
        dlist.append([numpy.array([a/2.,a/2.,0]), 1.0])
        dlist.append([numpy.array([a/2.,-a/2.,0]), 1.0])
        x = numpy.linspace(-a/2, a/2, num_points)
        for j in range(1,len(x)-1):
          p = x[j]
          dlist.append([numpy.array([-a/2.,p,0]), 1.0])
          dlist.append([numpy.array([p,a/2,0]), 1.0])
          dlist.append([numpy.array([p,-a/2,0]), 1.0])
