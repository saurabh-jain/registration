import numpy as np
from vtk import *
import os

# General surface class
class Grid:
    def __init__(self, gridPoints=None):
        if gridPoints == None:
            self.vertices = []
            self.faces = []
        else:
            x = gridPoints[0]
            y = gridPoints[1]
            #print x.shape, y.shape, x.size
            self.vertices = np.zeros([x.size, 2])
            self.vertices[:, 0] = x.flatten()
            self.vertices[:, 1] = y.flatten()
            n = x.shape[0]
            m = x.shape[1]
            self.faces = np.zeros([n*(m-1)+m*(n-1), 2])
            j = 0 
            for k in range(n):
                for l in range(m-1):
                    self.faces[j,:] = (k*m+l,k*m+l+1)
                    j += 1
            for k in range(n-1):
                for l in range(m):
                    self.faces[j,:] = (k*m+l,(k+1)*m+l)
                    j += 1
            self.faces = np.int_(self.faces)

    def copy(self, src):
        self.vertices = np.copy(src.vertices)
        self.faces = np.copy(src.faces)
        
    # Saves in .vtk format
    def saveVTK(self, fileName):
        F = self.faces
        V = self.vertices

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], 0))
            fvtkout.write('\nLINES {0:d} {1:d}'.format(F.shape[0], 3*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(F[ll,0], F[ll,1]))
            fvtkout.write('\n')

    def restrict(self, keepVert):
        V = np.copy(self.vertices)
        F = np.copy(self.faces)
        newInd = -np.ones(V.shape[0])
        j=0
        for k,kv in enumerate(keepVert):
            if kv:
                self.vertices[j,:] = V[k, :]
                newInd[k] = j
                j+=1
        self.vertices = self.vertices[0:j,:]
        j=0
        for k in range(F.shape[0]):
            if keepVert[F[k,0]] & keepVert[F[k,1]]:
                self.faces[j,0] = newInd[F[k,0]]
                self.faces[j,1] = newInd[F[k,1]]
                j+=1 
        self.faces = self.faces[0:j,:]

    def inPolygon(self, fv):
        nvert = self.vertices.shape[0]
        K = np.zeros(nvert)
        for k in range(nvert-1):
            pv0 = fv.vertices[fv.faces[:,0], :] - self.vertices[k,:]
            pv1 = fv.vertices[fv.faces[:,1], :] - self.vertices[k,:]
            c = np.multiply(pv1[:,1], pv0[:,0]) - np.multiply(pv1[:,0], pv0[:,1]) 
            #print c.shape
            c0 = np.sqrt((pv0**2).sum(axis=1))
            c1 = np.sqrt((pv1**2).sum(axis=1))
            c = np.divide(c, np.multiply(c0,c1)+1e-10)
            c = np.arcsin(c)
            #print c
            w = c.sum()/np.pi
            #print w /np.pi
            if abs(w) > .001:
                K[k] = 1
        return np.int_(K)

    def distPolygon(self, fv):
        nvert = self.vertices.shape[0]
        D = np.zeros(nvert)
        for k in range(nvert-1):
            D[k] = np.min(np.sqrt((fv.vertices[:, 0] - self.vertices[k,0])**2 +
                                  (fv.vertices[:, 1] - self.vertices[k,1])**2))
        return D

    def signedDistPolygon(self, fv):
        D = self.distPolygon(fv)
        K = self.inPolygon(fv)
        D *= 1-2*K 
        return D
