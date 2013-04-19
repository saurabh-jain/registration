import os
import sys
import struct
import numpy as np
from vtk import *
import array
from PIL import Image
import vtk.util.numpy_support as v2n

## Functions for images and diffeomorphisms

# Useful functions for multidimensianal arrays
class gridScalars:
   # initializes either form a previous array (data) or from a file 
   def __init__(self, data=None, fileName = None, dim = 3, resol = [1., 1., 1.], force_axun=False, withBug=False):
      if not (data == None):
         self.data = np.copy(data)
         self.resol = np.copy(resol)
      elif not (fileName==None):
         if (dim == 1):
            self.resol = 1.
            with open(filename, 'r') as ff:
               ln0 = ff.readline()
               while (len(ln0) == 0) | (ln0=='\n'):
                  ln0 = ff.readline()
               ln = ln0.split()
               nb = int(ln[0])
               self.data = zeros(nb)
               j = 0 
               while j < nb:
                  ln = ln0.readline.split()
                  for u in ln:
                     self.data[j] = u
                     j += 1
         elif (dim==2):
            self.resol  = [1., 1.]
            # u = vtkImageReader2()
            # u.setFileName(fileName)
            # u.Update()
            # v = u.GetOutput()
            img = Image.open(fileName)
            self.data = np.array(img.convert("L").getdata())
            self.data.resize(img.size)
         elif (dim == 3):
            (nm, ext) = os.path.splitext(fileName)
            if ext=='.hdr':
               self.loadAnalyze(fileName, force_axun= force_axun, withBug=withBug)
            elif ext =='.vtk':
               self.readVTK(fileName)
         else:
            print "get_image: unsupported input dimensions"
            return

   # Reads from vtk file
   def readVTK(self, filename):
      u = vtkStructuredPointsReader()
      u.SetFileName(filename)
      u.Update()
      v = u.GetOutput()
      dim = np.zeros(3)
      dim = v.GetDimensions()
      self.resol = v.GetSpacing()
      self.data = np.ndarray(shape=dim, order='F', buffer = v.GetPointData().GetScalars())

   # Saves in vtk file
   def saveVTK(self, filename, scalarName='scalars_', title='lddmm data'):
      with open(filename, 'w') as ff:
         ff.write('# vtk DataFile Version 2.0\n'+title+'\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {0: d} {1: d} {2: d}\n'.format(self.data.shape[0], self.data.shape[1], self.data.shape[2]))
         nbVox = np.array(self.data.shape).prod()
         ff.write('ORIGIN 0 0 0\nSPACING {0: f} {1: f} {2: f}\nPOINT_DATA {3: d}\n'.format(self.resol[0], self.resol[1], self.resol[2], nbVox))
         ff.write('SCALARS '+scalarName+' double 1\nLOOKUP_TABLE default\n')
         if sys.byteorder[0] == 'l':
            tmp = self.data.byteswap()
            tmp.T.tofile(ff)
         else:
            self.data.T.tofile(ff, order='F')

   # Reads from analyze file
   def saveAnalyze(self, filename):
      [nm, ext] = os.path.splitext(filename)
      with open(nm+'.hdr', 'w') as ff:
         x = 348
         ff.write(struct.pack('i', x))
         self.header[33] = self.data.shape[0]
         self.header[34] = self.data.shape[1]
         self.header[35] = self.data.shape[2]
         self.header[53] = 16
         self.header[57:60] = self.resol
         self.header[178] = 1
         frmt = 28*'B'+'i'+'h'+2*'B'+8*'h'+12*'B'+4*'h'+16*'f'+2*'i'+168*'B'+8*'i'
         ff.write(struct.pack(frmt, *self.header.tolist()))
      with open(nm+'.img', 'w') as ff:
         print self.data.max()
         #array.array('f', self.data[::-1,::-1,::-1].T.flatten()).tofile(ff)
         array.array('f', self.data.T.flatten()).tofile(ff)
         #uu = self.data[::-1,::-1,::-1].flatten()
         #print uu.max()
         #uu.tofile(ff)

   # Saves in analyze file
   def loadAnalyze(self, filename, force_axun=False, withBug=False):
      [nm, ext] = os.path.splitext(filename)
      with open(nm+'.hdr', 'r') as ff:
         frmt = 28*'B'+'i'+'h'+2*'B'+8*'h'+12*'B'+4*'h'+16*'f'+2*'i'+168*'B'+8*'i'
         lend = '<'
         ls = struct.unpack(lend+'i', ff.read(4))
         x = int(ls[0])
         #print 'x=',x
         if not (x == 348):
            lend = '>'
         ls = struct.unpack(lend+frmt, ff.read())
         self.header = np.array(ls)
         #print ls

         sz = ls[33:36]
         #print sz
         datatype = ls[53]
         #print  "Datatype: ", datatype
         self.resol = ls[57:60]
         self.hist_orient = ls[178]
         if force_axun:
               self.hist_orient = 0
         if withBug:
               self.hist_orient = 0
         print "Orientation: ", int(self.hist_orient)


      with open(nm+'.img', 'r') as ff:
         nbVox = sz[0]*sz[1]*sz[2]
         s = ff.read()
         #print s[0:30]
         if datatype == 2:
            ls2 = struct.unpack(lend+nbVox*'B', s)
         elif datatype == 4:
            ls2 = struct.unpack(lend+nbVox*'h', s)
         elif datatype == 8:
            ls2 = struct.unpack(lend+nbVox*'i', s)
         elif datatype == 16:
            ls2 = struct.unpack(lend+nbVox*'f', s)
         elif datatype == 32:
            ls2 = struct.unpack(lend+2*nbVox*'f', s)
            print 'Warning: complex input not handled'
         elif datatype == 64:
            ls2 = struct.unpack(lend+nbVox*'d', s)
         else:
            print 'Unknown datatype'
            return

         #ls = np.array(ls)
         #print ls
      #print 'size:', sz
      self.data = np.float_(ls2)
      self.data.resize(sz[::-1])
      #self.data.resize(sz)
      #print 'size:', self.data.shape
      self.data = self.data.T
      #self.data = self.data[::-1,::-1,::-1]
      #print 'size:', self.data.shape
      #print self.resol, ls[57]
      if self.hist_orient == 1:
            # self.resol = [ls[57],ls[58], ls[59]]
            self.resol = [ls[58],ls[57], ls[59]]
            #self.data = self.data.swapaxes(1,2)
            #self.data = self.data[::-1,::-1,::-1].swapaxes(1,2)
            #print self.resol
            #print self.data.shape
      elif self.hist_orient == 2:
            self.resol = [ls[58],ls[59], ls[57]]
            self.data = self.data[::-1,::-1,::-1].swapaxes(0,1).swapaxes(1,2)
      elif self.hist_orient == 3:
            self.resol = [ls[57],ls[58], ls[59]]
            self.data  = self.data[:, ::-1, :]
      elif self.hist_orient == 4:
            self.resol = [ls[58],ls[57], ls[59]]
            self.data = self.data[:,  ::-1, :].swapaxes(0,1)
      elif self.hist_orient == 5:
            self.resol = [ls[58],ls[59], ls[57]]
            self.data = self.data[:,::-1,:].swapaxes(0,1).swapaxes(1,2)
      else:
            self.resol = [ls[57],ls[58], ls[59]]
            if withBug:
               self.data  = self.data[::-1,::-1,::-1]
            #self.saveAnalyze('Data/foo.hdr')

   def zeroPad(self, h):
      d = np.copy(self.data)
      self.data = np.zeros([d.shape[0] + 2, d.shape[1]+2, d.shape[2]+2])
      self.data[1:d.shape[0]+1, 1:d.shape[1]+1, 1:d.shape[2]+1] = d 




class Diffeomorphism:
   def __init__(self, filename = None):
      self.readVTK(filename)
   def readVTK(self, filename):
      u = vtkStructuredPointsReader()
      u.SetFileName(filename)
      u.Update()
      v = u.GetOutput()
      self.resol = v.GetSpacing()
      dim = np.zeros(4)
      dim[1:4] = v.GetDimensions()
      dim[0] = 3
      v= v.GetPointData().GetVectors()
      self.data = np.ndarray(shape=dim, order='F', buffer = v)



# multilinear interpolation
def multilinInterp(img, diffeo):
   if img.ndim > 3:
      print 'interpolate only in dimension 1 to 3'
      return
   for k in range(img.ndim, 3):
      np.expand_dims(img, k)
      np.expand_dims(diffeo, k)
   tooLarge = diffeo.min() < 0
   for k in range(img.ndim):
      if (diffeo[k, :,:,:].max(axis=k) > img.shape[k]-1):
         tooLarge = True
         if tooLarge:
            dfo = np.copy(diffeo)
            dfo = max(df0, 0)
            for k in range(img.ndim):
               dfo[k, :, :, :] = min(dfo[k, :,:,:], img.shape[k]-1)
      else:
         dfo = diffeo

   res = np.copy(img)
   if img.shape[0] > 1:
      i0  = np.floor(dfo[0,:,:,:])
      i1 = min(i0+1, img.shape[0]-1)
      r0 = dfo[0,:,:,:] - i0
      res = np.multiply(img[i0, :, :], 1-r0) + np.multiply(img[i1, :, :], r0)
   if img.shape[1] > 1:
      i0  = np.floor(dfo[1,:,:,:])
      i1 = min(i0+1, img.shape[1]-1)
      r0 = dfo[1,:,:,:] - i0
      res = np.multiply(res[:, i0, :], 1-r0) + np.multiply(res[:, i1, :], r0)
   if img.shape[2] > 1:
      i0  = np.floor(dfo[2,:,:,:])
      i1 = min(i0+1, img.shape[2]-1)
      r0 = dfo[2,:,:,:] - i0
      res = np.multiply(res[:, :, i0], 1-r0) + np.multiply(res[:, :, i1], r0)

   img = np.squeeze(img)
   diffeo = np.squeeze(diffeo)
   return np.squeeze(res)


# Computes gradient
def gradient(img, resol=None):
   if img.ndim > 3:
      print 'gradient only in dimension 1 to 3'
      return
   for k in range(img.ndim, 3):
      np.expand_dims(img, k)
      np.expand_dims(diffeo, k)

   if img.ndim == 3:
      if resol == None:
         resol = [1.,1.,1.]
      res = np.zeros([3,img.shape[0], img.shape[1], img.shape[2]])
      res[0,1:img.shape[0]-1, :, :] = (img[2:img.shape[0], :, :] - img[0:img.shape[0]-2, :, :])/(2*resol[0])
      res[0,0, :, :] = (img[1, :, :] - img[0, :, :])/(resol[0])
      res[0,img.shape[0]-1, :, :] = (img[img.shape[0]-1, :, :] - img[img.shape[0]-2, :, :])/(resol[0])
      res[1,:, 1:img.shape[1]-1, :] = (img[:, 2:img.shape[1], :] - img[:, 0:img.shape[1]-2, :])/(2*resol[1])
      res[1,:, 0, :] = (img[:, 1, :] - img[:, 0, :])/(resol[1])
      res[1,:, img.shape[1]-1, :] = (img[:, img.shape[1]-1, :] - img[:, img.shape[1]-2, :])/(resol[1])
      res[2,:, :, 1:img.shape[2]-1] = (img[:, :, 2:img.shape[2]] - img[:, :, 0:img.shape[2]-2])/(2*resol[2])
      res[2,:, :, 0] = (img[:, :, 1] - img[:, :, 0])/(resol[2])
      res[2,:, :, img.shape[2]-1] = (img[:, :, img.shape[2]-1] - img[:, :, img.shape[2]-2])/(resol[2])
   elif img.ndim ==2:
      if resol == None:
         resol = [1.,1.]
      res = np.zeros([2,img.shape[0], img.shape[1]])
      res[0,1:img.shape[0]-1, :] = (img[2:img.shape[0], :] - img[0:img.shape[0]-2, :])/(2*resol[0])
      res[0,0, :] = (img[1, :] - img[0, :])/(resol[0])
      res[0,img.shape[0]-1, :] = (img[img.shape[0]-1, :] - img[img.shape[0]-2, :])/(resol[0])
      res[1,:, 0] = (img[:, 1] - img[:, 0])/(resol[1])
      res[1,:, img.shape[1]-1] = (img[:, img.shape[1]-1] - img[:, img.shape[1]-2])/(resol[1])
      res[1,:, 1:img.shape[1]-1] = (img[:, 2:img.shape[1]] - img[:, 0:img.shape[1]-2])/(2*resol[1])
   elif img.ndim ==1:
      if resol == None:
         resol = 1
      res = np.zeros(img.shape[0])
      res[1:img.shape[0]-1] = (img[2:img.shape[0]] - img[0:img.shape[0]-2])/(2*resol)
      res[0] = (img[1] - img[0])/(resol)
      res[img.shape[0]-1] = (img[img.shape[0]-1] - img[img.shape[0]-2])/(resol)
   return res

# Computes Jacobian determinant
def jacobianDeterminant(diffeo, resol=[1.,1.,1.], periodic=False):
   if diffeo.ndim > 4:
      print 'No jacobian in dimension larget than 3'
      return

   if diffeo.ndim == 4:
      if periodic == True:
         w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2], 0:diffeo.shape[3]]
         dw = diffeo-w
         for k in range(3):
            diffeo[k,:,:,:] -= np.rint(dw[k,:,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
      grad = np.zeros([3,3,diffeo.shape[1], diffeo.shape[2], diffeo.shape[3]])
      grad[0,:,:,:,:] = gradient(np.squeeze(diffeo[0,:,:,:]), resol=resol)
      grad[1,:,:,:,:] = gradient(np.squeeze(diffeo[1,:,:,:]), resol=resol)
      grad[2,:,:,:,:] = gradient(np.squeeze(diffeo[2,:,:,:]), resol=resol)
      res = np.zeros([diffeo.shape[0], diffeo.shape[1], diffeo.shape[2]])
      res = np.fabs(grad[0,0,:,:,:] * grad[1,1,:,:,:] * grad[2,2,:,:,:] 
                    - grad[0,0,:,:,:] * grad[1,2,:,:,:] * grad[2,1,:,:,:]
                    - grad[0,1,:,:,:] * grad[1,0,:,:,:] * grad[2,2,:,:,:]
                    - grad[0,2,:,:,:] * grad[1,1,:,:,:] * grad[2,0,:,:,:]
                    + grad[0,1,:,:,:] * grad[1,2,:,:,:] * grad[2,0,:,:,:] 
                    + grad[0,2,:,:,:] * grad[1,0,:,:,:] * grad[2,1,:,:,:])
   elif diffeo.ndim == 3:
      if periodic == True:
         w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2]]
         dw = diffeo-w
         for k in range(2):
            diffeo[k,:,:] -= np.rint(dw[k,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
      grad = np.zeros([2,2,diffeo.shape[1], diffeo.shape[2]])
      grad[0,:,:,:] = gradient(np.squeeze(diffeo[0,:,:]), resol=resol)
      grad[1,:,:,:] = gradient(np.squeeze(diffeo[1,:,:]), resol=resol)
      res = np.zeros([diffeo.shape[0], diffeo.shape[1]], resol=resol)
      res = np.fabs(grad[0,0,:,:] * grad[1,1,:,:] - grad[0,1,:,:] * grad[1,0,:,:])
   else:
      if periodic == True:
         w = np.mgrid[0:diffeo.shape[0]]
         dw = diffeo-w
         diffeo -= np.rint(dw/diffeo.shape[0])*diffeo.shape[0]
      res =  np.fabs(gradient(np.squeeze(diffeo)), resol=resol)
   return res

# Computes differential
def jacobianMatrix(diffeo, resol=[1.,1.,1.], periodic=False):
   if diffeo.ndim > 4:
      print 'No jacobian in dimension larget than 3'
      return

   if diffeo.ndim == 4:
      if periodic == True:
         w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2], 0:diffeo.shape[3]]
         dw = diffeo-w
         for k in range(3):
            diffeo[k,:,:,:] -= np.rint(dw[k,:,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
      grad = np.zeros([3,3,diffeo.shape[1], diffeo.shape[2], diffeo.shape[3]])
      grad[0,:,:,:,:] = gradient(np.squeeze(diffeo[0,:,:,:]), resol=resol)
      grad[1,:,:,:,:] = gradient(np.squeeze(diffeo[1,:,:,:]), resol=resol)
      grad[2,:,:,:,:] = gradient(np.squeeze(diffeo[2,:,:,:]), resol=resol)
   elif diffeo.ndim == 3:
      if periodic == True:
         w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2]]
         dw = diffeo-w
         for k in range(2):
            diffeo[k,:,:] -= np.rint(dw[k,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
      grad = np.zeros([2,2,diffeo.shape[1], diffeo.shape[2]])
      grad[0,:,:,:] = gradient(np.squeeze(diffeo[0,:,:]), resol=resol)
      grad[1,:,:,:] = gradient(np.squeeze(diffeo[1,:,:]), resol=resol)
   else:
      if periodic == True:
         w = np.mgrid[0:diffeo.shape[0]]
         dw = diffeo-w
         diffeo -= np.rint(dw/diffeo.shape[0])*diffeo.shape[0]
      grad =  np.fabs(gradient(np.squeeze(diffeo)), resol=resol)
   return grad


