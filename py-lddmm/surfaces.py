import numpy as np
import scipy as sp
import os
import glob
from vtk import *
import kernelFunctions as kfun

# General surface class
class Surface:
    def __init__(self, surf=None, filename=None, FV = None):
        if surf == None:
            if FV == None:
                if filename == None:
                    self.vertices = np.empty(0)
                    self.centers = np.empty(0)
                    self.faces = np.empty(0)
                    self.surfel = np.empty(0)
                else:
                    (mainPart, ext) = os.path.splitext(filename)
                    if ext == '.byu':
                        self.readbyu(filename)
                    elif ext=='.vtk':
                        self.readVTK(filename)
                    else:
                        print 'Unknown Surface Extension:', ext
                        self.vertices = np.empty(0)
                        self.centers = np.empty(0)
                        self.faces = np.empty(0)
                        self.surfel = np.empty(0)
            else:
                self.vertices = np.copy(FV[1])
                self.faces = np.int_(FV[0])
                self.computeCentersAreas()
        else:
            self.vertices = np.copy(surf.vertices)
            self.surfel = np.copy(surf.surfel)
            self.faces = np.copy(surf.faces)
            self.centers = np.copy(surf.centers)

    # face centers and area weighted normal
    def computeCentersAreas(self):
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0) 
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)

    def computeVertexArea(self):
        a = np.zeros(self.vertices.shape[0])
        n = np.zeros(self.vertices.shape[0])
        af = np.sqrt((self.surfel**2).sum(axis=1))
        for jj in range(3):
            I = self.faces[:,jj]
            for k in range(I.size):
                a[I[k]] += af[k]
                n[I[k]] += 1
        a = np.divide(a,n)
        return a

         

    # Computes edges from vertices/faces
    def getEdges(self):
        self.edges = []
        for k in range(self.faces.shape[0]):
            u = [self.faces[k, 0], self.faces[k, 1]]
            if (u not in self.edges) & (u.reverse() not in self.edges):
                self.edges.append(u)
            u = [self.faces[k, 1], self.faces[k, 2]]
            if (u not in self.edges) & (u.reverse() not in self.edges):
                self.edges.append(u)
            u = [self.faces[k, 2], self.faces[k, 0]]
            if (u not in self.edges) & (u.reverse() not in self.edges):
                self.edges.append(u)

    # computes the signed distance function in a small neighborhood of a shape 
    def LocalSignedDistance(self, data, value):
        d2 = 2*np.array(data >= value) - 1
        c2 = np.cumsum(d2, axis=0)
        for j in range(2):
            c2 = np.cumsum(c2, axis=j+1)
        (n0, n1, n2) = c2.shape

        rad = 3
        diam = 2*rad+1
        (x,y,z) = np.mgrid[-rad:rad+1, -rad:rad+1, -rad:rad+1]
        cube = (x**2+y**2+z**2)
        maxval = (diam)**3
        s = 3.0*rad**2
        res = d2*s
        u = maxval*np.ones(c2.shape)
        u[rad+1:n0-rad, rad+1:n1-rad, rad+1:n2-rad] = (c2[diam:n0, diam:n1, diam:n2]
                                                 - c2[0:n0-diam, diam:n1, diam:n2] - c2[diam:n0, 0:n1-diam, diam:n2] - c2[diam:n0, diam:n1, 0:n2-diam] 
                                                 + c2[0:n0-diam, 0:n1-diam, diam:n2] + c2[diam:n0, 0:n1-diam, 0:n2-diam] + c2[0:n0-diam, diam:n1, 0:n2-diam]
                                                 - c2[0:n0-diam, 0:n1-diam, 0:n2-diam])

        I = np.nonzero(np.fabs(u) < maxval)
        #print len(I[0])

        for k in range(len(I[0])):
            p = np.array((I[0][k], I[1][k], I[2][k]))
            bmin = p-rad
            bmax = p+rad + 1
            #print p, bmin, bmax
            if (d2[p[0],p[1], p[2]] > 0):
                #print u[p[0],p[1], p[2]]
                #print d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]].sum()
                res[p[0],p[1], p[2]] = min(cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] < 0)])-.25
            else:
                res[p[0],p[1], p[2]] =- min(cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] > 0)])-.25
                
        return res                
            
    # Computes isosurfaces using vtk               
    def Isosurface(self, data, value=0.5, target=1000.0, scales = [1., 1., 1.], smooth = -1, fill_holes = 1.):
        #data = self.LocalSignedDistance(data0, value)
        img = vtkImageData()
        img.SetDimensions(data.shape)
        img.SetNumberOfScalarComponents(1)
        img.SetOrigin(0,0,0)
        v = vtkDoubleArray()
        v.SetNumberOfValues(data.size)
        v.SetNumberOfComponents(1)
        for ii,tmp in enumerate(np.ravel(data, order='F')):
            v.SetValue(ii,tmp)
        img.GetPointData().SetScalars(v)
        cf = vtkContourFilter()
        cf.SetInput(img)
        cf.SetValue(0,value)
        cf.SetNumberOfContours(1)
        cf.Update()
        #print cf
        connectivity = vtkPolyDataConnectivityFilter()
        connectivity.ScalarConnectivityOff()
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.SetInput(cf.GetOutput())
        connectivity.Update()
        g = connectivity.GetOutput()

            


        if smooth > 0:
            smoother= vtkWindowedSincPolyDataFilter()
            smoother.SetInput(g)
            #     else:
            # smoother.SetInputConnection(contour.GetOutputPort())    
            smoother.SetNumberOfIterations(30)
            #this has little effect on the error!
            #smoother.BoundarySmoothingOff()
            #smoother.FeatureEdgeSmoothingOff()
            #smoother.SetFeatureAngle(120.0)
            smoother.SetPassBand(smooth)        #this increases the error a lot!
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.GenerateErrorScalarsOn() 
            #smoother.GenerateErrorVectorsOn()
            smoother.Update()
            g = smoother.GetOutput()

        dc = vtkDecimatePro()
        red = 1 - min(np.float(target)/g.GetNumberOfPoints(), 1)
            #print 'Reduction: ', red
        dc.SetTargetReduction(red)
        dc.PreserveTopologyOn()
        #dc.SetSplitting(0)
        dc.SetInput(g)
        #print dc
        dc.Update()
        g = dc.GetOutput()
        #print 'points:', g.GetNumberOfPoints()
        cp = vtkCleanPolyData()
        cp.SetInput(dc.GetOutput())
        #cp.SetPointMerging(1)
        cp.ConvertPolysToLinesOn()
        cp.SetAbsoluteTolerance(1e-5)
        cp.Update()
        g = cp.GetOutput()
        #print g
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfPolys())
        print 'Dimensions:', npoints, nfaces, g.GetNumberOfCells()
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk))
            #print kk, V[kk]
            #print kk, np.array(g.GetPoint(kk))
        F = np.zeros([nfaces, 3])
        gf = 0
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            if(c.GetNumberOfPoints() == 3):
                for ll in range(3):
                    F[gf,ll] = c.GetPointId(ll)
                    #print kk, gf, F[gf]
                gf += 1

                #self.vertices = np.multiply(data.shape-V-1, scales)
        self.vertices = np.multiply(V, scales)
        self.faces = np.int_(F[0:gf, :])
        self.computeCentersAreas()

    # Ensures that orientation is correct
    def edgeRecover(self):
        v = self.vertices
        f = self.faces
        nv = v.shape[0]
        nf = f.shape[0]
        # faces containing each oriented edge
        edg0 = np.int_(np.zeros((nv, nv)))
        # number of edges between each vertex
        edg = np.int_(np.zeros((nv, nv)))
        # contiguous faces
        edgF = np.int_(np.zeros((nf, nf)))
        for (kf, c) in enumerate(f):
            if (edg0[c[0],c[1]] > 0):
                edg0[c[1],c[0]] = kf+1  
            else:
                edg0[c[0],c[1]] = kf+1
                
            if (edg0[c[1],c[2]] > 0):
                edg0[c[2],c[1]] = kf+1  
            else:
                edg0[c[1],c[2]] = kf+1  

            if (edg0[c[2],c[0]] > 0):
                edg0[c[0],c[2]] = kf+1  
            else:
                edg0[c[2],c[0]] = kf+1  

            edg[c[0],c[1]] += 1
            edg[c[1],c[2]] += 1
            edg[c[2],c[0]] += 1

        for kv in range(nv):
            I2 = np.nonzero(edg0[kv,:])
            for kkv in I2[0].tolist():
                edgF[edg0[kkv,kv]-1,edg0[kv,kkv]-1] = kv+1

        isOriented = np.int_(np.zeros(f.shape[0]))
        isActive = np.int_(np.zeros(f.shape[0]))
        I = np.nonzero(np.squeeze(edgF[0,:]))
        # list of faces to be oriented
        activeList = [0]+I[0].tolist()
        lastOriented = 0
        isOriented[0] = True
        for k in activeList:
            isActive[k] = True 

        while lastOriented < len(activeList)-1:
            i = activeList[lastOriented]
            j = activeList[lastOriented +1]
            I = np.nonzero(edgF[j,:])
            foundOne = False
            for kk in I[0].tolist():
                if (foundOne==False)  & (isOriented[kk]):
                    foundOne = True
                    u1 = edgF[j,kk] -1
                    u2 = edgF[kk,j] - 1
                    if not ((edg[u1,u2] == 1) & (edg[u2,u1] == 1)): 
                        # reorient face j
                        edg[f[j,0],f[j,1]] -= 1
                        edg[f[j,1],f[j,2]] -= 1
                        edg[f[j,2],f[j,0]] -= 1
                        a = f[j,1]
                        f[j,1] = f[j,2]
                        f[j,2] = a
                        edg[f[j,0],f[j,1]] += 1
                        edg[f[j,1],f[j,2]] += 1
                        edg[f[j,2],f[j,0]] += 1
                elif (not isActive[kk]):
                    activeList.append(kk)
                    isActive[kk] = True
            if foundOne:
                lastOriented = lastOriented+1
                isOriented[j] = True
                #print 'oriented face', j, lastOriented,  'out of',  nf,  ';  total active', len(activeList) 
            else:
                print 'Unable to orient face', j 
                return
        self.vertices = v ;
        self.faces = f ;

        z= self.surfVolume()
        if (z > 0):
            self.faces = f[:, [0,2,1]]

    # Computes surface volume
    def surfVolume(self):
        f = self.faces
        v = self.vertices
        z = 0
        for c in f:
            z += np.linalg.det(v[c[:], :])/6
        return z

    # Reads from .byu file
    def readbyu(self, byufile):
        with open(byufile,'r') as fbyu:
            ln0 = fbyu.readline()
            ln = ln0.split()
            # read header
            ncomponents = int(ln[0])	# number of components
            npoints = int(ln[1])  # number of vertices
            nfaces = int(ln[2]) # number of faces
                        #fscanf(fbyu,'%d',1);		% number of edges
                        #%ntest = fscanf(fbyu,'%d',1);		% number of edges
            for k in range(ncomponents):
                fbyu.readline() # components (ignored)
            # read data
            self.vertices = np.empty([npoints, 3]) ;
            k=-1
            while k < npoints-1:
                ln = fbyu.readline().split()
                k=k+1 ;
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                self.vertices[k, 2] = float(ln[2])
                if len(ln) > 3:
                    k=k+1 ;
                    self.vertices[k, 0] = float(ln[3])
                    self.vertices[k, 1] = float(ln[4]) 
                    self.vertices[k, 2] = float(ln[5])

            self.faces = np.empty([nfaces, 3])
            ln = fbyu.readline().split()
            kf = 0
            j = 0
            while ln:
		if kf >= nfaces:
		    break
		#print nfaces, kf, ln
                for s in ln:
                    self.faces[kf,j] = int(sp.fabs(int(s)))
                    j = j+1
                    if j == 3:
                        kf=kf+1
                        j=0
                ln = fbyu.readline().split()
        self.faces = np.int_(self.faces) - 1
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)

    #Saves in .byu format
    def savebyu(self, byufile):
        #FV = readbyu(byufile)
        #reads from a .byu file into matlab's face vertex structure FV

        with open(byufile,'w') as fbyu:
            # copy header
            ncomponents = 1	    # number of components
            npoints = self.vertices.shape[0] # number of vertices
            nfaces = self.faces.shape[0]		# number of faces
            nedges = 3*nfaces		# number of edges

            str = '{0: d} {1: d} {2: d} {3: d} 0\n'.format(ncomponents, npoints, nfaces,nedges)
            fbyu.write(str) 
            str = '1 {0: d}\n'.format(nfaces)
            fbyu.write(str) 


            k=-1
            while k < (npoints-1):
                k=k+1 
                str = '{0: f} {1: f} {2: f} '.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                fbyu.write(str) 
                if k < (npoints-1):
                    k=k+1
                    str = '{0: f} {1: f} {2: f}\n'.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                    fbyu.write(str) 
                else:
                    fbyu.write('\n')

            j = 0 
            for k in range(nfaces):
                for kk in (0,1):
                    fbyu.write('{0: d} '.format(self.faces[k,kk]+1))
                    j=j+1
                    if j==16:
                        fbyu.write('\n')
                        j=0

                fbyu.write('{0: d} '.format(-self.faces[k,2]-1))
                j=j+1
                if j==16:
                    fbyu.write('\n')
                    j=0

    # Saves in .vtk format
    def saveVTK(self, fileName, scalars = None, normals = None, scal_name='scalars'):
        F = self.faces ;
        V = self.vertices ;

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], V[ll,2]))
            fvtkout.write('\nPOLYGONS {0:d} {1:d}'.format(F.shape[0], 4*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n3 {0: d} {1: d} {2: d}'.format(F[ll,0], F[ll,1], F[ll,2]))
            if (not (scalars == None)) | (not (normals==None)):
                fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            if not (scalars == None):
                fvtkout.write('\nSCALARS '+scal_name+' float 1\nLOOKUP_TABLE default')
                for ll in range(V.shape[0]):
                    fvtkout.write('\n {0: .5f}'.format(scalars[ll]))
            if not (normals == None):
                fvtkout.write('\nNORMALS normals float')
                for ll in range(V.shape[0]):
                    fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(normals[ll, 0], normals[ll, 1], normals[ll, 2]))
            fvtkout.write('\n')


    # Reads .vtk file
    def readVTK(self, fileName):
        u = vtkPolyDataReader()
        u.SetFileName(fileName)
        u.Update()
        v = u.GetOutput()
        npoints = int(v.GetNumberOfPoints())
        nfaces = int(v.GetNumberOfPolys())
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(v.GetPoint(kk))

        F = np.zeros([nfaces, 3])
        for kk in range(nfaces):
            c = v.GetCell(kk)
            for ll in range(3):
                F[kk,ll] = c.GetPointId(ll)
        
        self.vertices = V
        self.faces = np.int_(F)
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)

# Reads several .byu files
def readMultipleByu(regexp, Nmax = 0):
    files = glob.glob(regexp)
    if Nmax > 0:
        nm = min(Nmax, len(files))
    else:
        nm = len(files)
    fv1 = []
    for k in range(nm):
        fv1.append(Surface(files[k]))
    return fv1

# saves time dependent surfaces (fixed topology)
def saveEvolution(fileName, fv0, xt):
    fv = Surface(fv0)
    for k in range(xt.shape[0]):
        fv.vertices = np.squeeze(xt[k, :, :])
        fv.savebyu(fileName+'{0: 02d}'.format(k)+'.byu')



# Current norm of fv1
def currentNorm0(fv1, KparDist):
    c2 = fv1.centers
    cr2 = np.mat(fv1.surfel)
    g11 = kfun.kernelMatrix(KparDist, c2)
    return np.multiply((cr2*cr2.T), g11).sum()
        

# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct 
def currentNormDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 = np.mat(fvDef.surfel)
    c2 = fv1.centers
    cr2 = np.mat(fv1.surfel)
    g11 = kfun.kernelMatrix(KparDist, c1)
    g12 = kfun.kernelMatrix(KparDist, c1, c2)
    obj = (np.multiply(cr1*cr1.T, g11).sum() - 2*np.multiply(cr1*(cr2.T), g12).sum())
    return obj

# Returns |fvDef - fv1|^2 for current norm
def currentNorm(fvDef, fv1, KparDist):
    return currentNormDef(fvDef, fv1, KparDist) + currentNorm0(fv1, KparDist) 

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentNormGradient(fvDef, fv1, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = np.mat(fvDef.surfel)
    c2 = fv1.centers
    cr2 = np.mat(fv1.surfel)
    dim = c1.shape[1]

    g11 = kfun.kernelMatrix(KparDist, c1)
    dg11 = kfun.kernelMatrix(KparDist, c1, diff=True)
    
    g12 = kfun.kernelMatrix(KparDist, c1, c2)
    dg12 = kfun.kernelMatrix(KparDist, c1, c2, diff=True)


    z1 = g11*cr1 - g12 * cr2
    dg11 = np.multiply(dg11 ,(cr1*(cr1.T)))
    dg12 = np.multiply(dg12 , (cr1*(cr2.T)))

    dz1 = (2./3.) * (np.multiply(np.tile(dg11.sum(axis=1), (1,dim)), c1) - dg11*c1 - np.multiply(np.tile(dg12.sum(axis=1), (1, dim)), c1) + dg12*c2)

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:,0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    return 2*px

# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    c2 = fv1.centers
    cr2 = fv1.surfel
    cr2 = np.mat(np.sqrt((cr2**2).sum(axis=1)))
    g11 = kfun.kernelMatrix(KparDist, c2)
    #print cr2.shape, g11.shape
    return (cr2 * (g11*cr2.T)).sum()
        
    
# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct 
def measureNormDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 = fvDef.surfel
    cr1 = np.mat(np.sqrt((cr1**2).sum(axis=1)+1e-10))
    c2 = fv1.centers
    cr2 = fv1.surfel
    cr2 = np.mat(np.sqrt((cr2**2).sum(axis=1)+1e-10))
    g11 = kfun.kernelMatrix(KparDist, c1)
    g12 = kfun.kernelMatrix(KparDist, c1, c2)
    #obj = (np.multiply(cr1*cr1.T, g11).sum() - 2*np.multiply(cr1*(cr2.T), g12).sum())
    obj = (cr1 * g11 * cr1.T).sum() - 2* (cr1 * g12 *cr2.T).sum()
    return obj

# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist) 


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.surfel
    c2 = fv1.centers
    cr2 = fv1.surfel
    dim = c1.shape[1]
    a1 = np.mat(np.sqrt((cr1**2).sum(axis=1)+1e-10))
    a2 = np.mat(np.sqrt((cr2**2).sum(axis=1)+1e-10))
    cr1 = np.divide(cr1, a1.T)
    cr2 = np.divide(cr2, a2.T)

    g11 = kfun.kernelMatrix(KparDist, c1)
    dg11 = kfun.kernelMatrix(KparDist, c1, diff=True)
    
    g12 = kfun.kernelMatrix(KparDist, c1, c2)
    dg12 = kfun.kernelMatrix(KparDist, c1, c2, diff=True)


    z1 = g11*a1.T - g12 * a2.T
    z1 = np.multiply(z1, cr1)
    dg1 = np.multiply(dg11, a1.T)
    dg11 = np.multiply(dg1, a1)
    dg1 = np.multiply(dg12, a1.T)
    dg12 = np.multiply(dg1, a2)

    dz1 = (2./3.) * (np.multiply(dg11.sum(axis=1), c1) - dg11*c1 - np.multiply(dg12.sum(axis=1), c1) + dg12*c2)

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:,0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    I = fvDef.faces[:,2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :]+dz1[k, :] -  crs[k, :]

    return 2*px

# class MultiSurface:
#     def __init__(self, pattern):
#         self.surf = []
#         files = glob.glob(pattern)
#         for f in files:
#             self.surf.append(Surface(filename=f))
