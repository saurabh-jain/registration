import numpy as np
import scipy as sp
import os
import glob
from vtk import *
import kernelFunctions as kfun

# General surface class
class Curve:
    def __init__(self, curve=None, filename=None, FV = None, pointSet=None, isOpen=False):
        if curve == None:
            if FV == None:
                if filename == None:
                    if pointSet == None:
                        self.vertices = np.empty(0)
                        self.centers = np.empty(0)
                        self.faces = np.empty(0)
                        self.lenel = np.empty(0)
                    else:
                        self.vertices = np.copy(pointSet)
                        self.faces = np.int_(np.zeros([pointSet.shape[0], 2]))
                        for k in range(pointSet.shape[0]-1):
                            self.faces[k,:] = (k, k+1)
                        if isOpen == False:
                            self.faces[pointSet.shape[0]-1, :] = (pointSet.shape[0]-1, 0) ;
                        self.computeCentersLengths()
                else:
                    (mainPart, ext) = os.path.splitext(filename)
                    if ext == '.dat':
                        self.readCurve(filename)
                    elif ext=='.vtk':
                        self.readVTK(filename)
                    else:
                        print 'Unknown Surface Extension:', ext
                        self.vertices = np.empty(0)
                        self.centers = np.empty(0)
                        self.faces = np.empty(0)
                        self.lenel = np.empty(0)
            else:
                self.vertices = np.copy(FV[1])
                self.faces = np.int_(FV[0])
                self.computeCentersLengths()
        else:
            self.vertices = np.copy(curve.vertices)
            self.lenel = np.copy(curve.lenel)
            self.faces = np.copy(curve.faces)
            self.centers = np.copy(curve.centers)

    # face centers and area weighted normal
    def computeCentersLengths(self):
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        self.centers = (xDef1 + xDef2) / 2
        #self.lenel = np.zeros([self.faces.shape[0], self.vertices.shape[1]]) ;
        self.lenel = xDef2 - xDef1 ; 
        #self.lenel[:,1] = xDef2[:,1] - xDef1[:,1] ; 

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0) 
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        self.centers = (xDef1 + xDef2) / 2
        #self.lenel = np.zeros([self.faces.shape[0], self.vertices.shape[1]]) ;
        self.lenel = xDef2 - xDef1 ; 
        #self.lenel[:,1] = xDef2[:,1] - xDef1[:,1] ; 

    def computeVertexLength(self):
        a = np.zeros(self.vertices.shape[0])
        n = np.zeros(self.vertices.shape[0])
        af = np.sqrt((self.lenel**2).sum(axis=1))
        for jj in range(3):
            I = self.faces[:,jj]
            for k in range(I.size):
                a[I[k]] += af[k]
                n[I[k]] += 1
        a = np.divide(a,n)
        return a


            
    # Computes isocontours using vtk               
    def Isocontour(self, data, value=0.5, target=100.0, scales = [1., 1.], smooth = 30, fill_holes = 1., singleComponent = True):
        #data = self.LocalSignedDistance(data0, value)
        img = vtkImageData()
        img.SetDimensions(data.shape[0], data.shape[1], 1)
        img.SetNumberOfScalarComponents(1)
        img.SetOrigin(0,0, 0)
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
        # return cf
        # #print cf
        if singleComponent:
            connectivity = vtkPolyDataConnectivityFilter()
            connectivity.ScalarConnectivityOff()
            connectivity.SetExtractionModeToLargestRegion()
            connectivity.SetInput(cf.GetOutput())
            connectivity.Update()
            g = connectivity.GetOutput()
        else:
            g = cf.GetOutput()
            
        # if smooth > 0:
        #     smoother= vtkWindowedSincPolyDataFilter()
        #     smoother.SetInput(g)
        #     smoother.SetNumberOfIterations(smooth)
        #     smoother.NonManifoldSmoothingOn()
        #     smoother.NormalizeCoordinatesOn()
        #     smoother.GenerateErrorScalarsOn() 
        #     smoother.Update()
        #     g = smoother.GetOutput()

        # dc = vtkDecimatePro()
        # red = 1 - min(np.float(target)/g.GetNumberOfPoints(), 1)
        # dc.SetTargetReduction(red)
        # dc.PreserveTopologyOn()
        # dc.SetInput(g)
        # dc.Update()
        # g = dc.GetOutput()
        # #print 'points:', g.GetNumberOfPoints()

        # cp = vtkCleanPolyData()
        # cp.SetInput(dc.GetOutput())
        # cp.ConvertPolysToLinesOn()
        # cp.SetAbsoluteTolerance(1e-5)
        # cp.Update()
        # g = cp.GetOutput()

        #g = cf.GetOutput() ;
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfLines())
        print 'Dimensions:', npoints, nfaces, g.GetNumberOfCells()
        V = np.zeros([npoints, 2])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk)[0:2])
            #print kk, V[kk]
            #print kk, np.array(g.GetPoint(kk))
        F = np.zeros([nfaces, 2])
        gf = 0
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            if(c.GetNumberOfPoints() == 2):
                for ll in range(2):
                    F[gf,ll] = c.GetPointId(ll)
                    #print kk, gf, F[gf]
                gf += 1

                #self.vertices = np.multiply(data.shape-V-1, scales)
        self.vertices = np.multiply(V, scales)
        self.faces = np.int_(F[0:gf, :])
        self.computeCentersLengths()
        #self.checkEdges()
        #print self.faces.shape
        self.orientEdges()
        #print self.faces.shape
        #self.checkEdges()


    def orientEdges(self):
        isInFace = - np.ones([self.vertices.shape[0], 2])
        is0 = np.zeros(self.vertices.shape[0])
        is1 = np.zeros(self.vertices.shape[0])
        for k in range(self.faces.shape[0]):
            if isInFace[self.faces[k,0], 0] == -1:
                isInFace[self.faces[k,0], 0] = k
            else:
                isInFace[self.faces[k,0], 1] = k

            if isInFace[self.faces[k,1], 0] == -1:
                isInFace[self.faces[k,1], 0] = k
            else:
                isInFace[self.faces[k,1], 1] = k
            is0[self.faces[k,0]] += 1
            is1[self.faces[k,1]] += 1
        isInFace = np.int_(isInFace)
        is0 = np.int_(is0)
        is1 = np.int_(is1)

        if ((is0+is1).max() !=2) | ((is0+is1).min()!=2):
            print 'Problems with curve: wrong topology'
            return

        count = np.zeros(self.vertices.shape[0])
        usedFace = np.zeros(self.faces.shape[0])
        F = np.int_(np.zeros(self.faces.shape))
        F[0, :] = self.faces[0,:]
        usedFace[0] = 1
        count[F[0,0]] = 1
        count[F[0,1]] = 1
        k0 = F[0,0]
        kcur = F[0,1]
        j=1
        while j < self.faces.shape[0]:
            #print j
            if usedFace[isInFace[kcur,0]]>0.5:
                kf = isInFace[kcur,1]
            else:
                kf = isInFace[kcur,0]
                #print kf
            usedFace[kf] = 1
            F[j, 0] = kcur
            if self.faces[kf,0] == kcur:
                F[j,1] = self.faces[kf,1]
            else:
                F[j,1] = self.faces[kf,0]
                #print kcur, self.faces[kf,:], F[j,:]
            if count[F[j,1]] > 0.5:
                j += 1
                if (j < self.faces.shape[0]):
                    print 'Early loop in curve:', j, self.faces.shape[0]
                break
            count[F[j,1]]=1
            kcur = F[j,1]
            j += 1
            #print j
            #print j, self.faces.shape[0]
        self.faces = np.int_(F[0:j, :])
            

    def checkEdges(self):
        is0 = np.zeros(self.vertices.shape[0])
        is1 = np.zeros(self.vertices.shape[0])
        is0 = np.int_(is0)
        is1 = np.int_(is1)
        for k in range(self.faces.shape[0]):
            is0[self.faces[k,0]] += 1
            is1[self.faces[k,1]] += 1
        #print is0 + is1
        if ((is0.max() !=1) | (is0.min() !=1) | (is1.max() != 1) | (is1.min() != 1)):
            print 'Problem in Curve'
            #print is0+is1
            return 1
        else:
            return 0

    # Computes enclosed area
    def enclosedArea(self):
        f = self.faces
        v = self.vertices
        z = 0
        for c in f:
            z += np.linalg.det(v[c[:], :])/2
        return z

    # Reads from .byu file
    def readCurve(self, infile):
        with open(infile,'r') as fbyu:
            ln0 = fbyu.readline()
            ln = ln0.split()
            # read header
            ncomponents = int(ln[0])	# number of components
            npoints = int(ln[1])  # number of vertices
            nfaces = int(ln[2]) # number of faces
            for k in range(ncomponents):
                fbyu.readline() # components (ignored)
            # read data
            self.vertices = np.empty([npoints, 2]) ;
            k=-1
            while k < npoints-1:
                ln = fbyu.readline().split()
                k=k+1 ;
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                if len(ln) > 2:
                    k=k+1 ;
                    self.vertices[k, 0] = float(ln[2])
                    self.vertices[k, 1] = float(ln[3]) 

            self.faces = np.empty([nfaces, 2])
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
                    if j == 2:
                        kf=kf+1
                        j=0
                ln = fbyu.readline().split()
        self.faces = np.int_(self.faces) - 1
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        self.centers = (xDef1 + xDef2) / 2
        #self.lenel = np.zeros(self.faces.shape[0], self.vertices.shape[1]) ;
        self.lenel = xDef2 - xDef1 ; 
        #self.lenel[:,1] = xDef2[:,1] - xDef1[:,1] ; 

    #Saves in .byu format
    def saveCurve(self, outfile):
        #FV = readbyu(byufile)
        #reads from a .byu file into matlab's face vertex structure FV

        with open(outfile,'w') as fbyu:
            # copy header
            ncomponents = 1	    # number of components
            npoints = self.vertices.shape[0] # number of vertices
            nfaces = self.faces.shape[0]		# number of faces
            nedges = 2*nfaces		# number of edges

            str = '{0: d} {1: d} {2: d} {3: d} 0\n'.format(ncomponents, npoints, nfaces,nedges)
            fbyu.write(str) 
            str = '1 {0: d}\n'.format(nfaces)
            fbyu.write(str) 


            k=-1
            while k < (npoints-1):
                k=k+1 
                str = '{0: f} {1: f} '.format(self.vertices[k, 0], self.vertices[k, 1])
                fbyu.write(str) 
                if k < (npoints-1):
                    k=k+1
                    str = '{0: f} {1: f}\n'.format(self.vertices[k, 0], self.vertices[k, 1])
                    fbyu.write(str) 
                else:
                    fbyu.write('\n')

            j = 0 
            for k in range(nfaces):
                fbyu.write('{0: d} '.format(self.faces[k,0]+1))
                j=j+1
                if j==16:
                    fbyu.write('\n')
                    j=0

                fbyu.write('{0: d} '.format(-self.faces[k,1]-1))
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
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], 0))
            fvtkout.write('\nLINES {0:d} {1:d}'.format(F.shape[0], 3*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(F[ll,0], F[ll,1]))
            if (not (scalars == None)) | (not (normals==None)):
                fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
            if not (scalars == None):
                fvtkout.write('\nSCALARS '+scal_name+' float 1\nLOOKUP_TABLE default')
                for ll in range(V.shape[0]):
                    fvtkout.write('\n {0: .5f}'.format(scalars[ll]))
            if not (normals == None):
                fvtkout.write('\nNORMALS normals float')
                for ll in range(V.shape[0]):
                    fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(normals[ll, 0], normals[ll, 1], 0))
            fvtkout.write('\n')


    # Reads .vtk file TO DO
    def readVTK(self, fileName):
        u = vtkPolyDataReader()
        u.SetFileName(fileName)
        u.Update()
        v = u.GetOutput()
        npoints = int(v.GetNumberOfPoints())
        nfaces = int(v.GetNumberOfLines())
        V = np.zeros([npoints, 2])
        for kk in range(npoints):
            V[kk, :] = np.array(v.GetPoint(kk)[0:2])

        F = np.zeros([nfaces, 2])
        for kk in range(nfaces):
            c = v.GetCell(kk)
            for ll in range(2):
                F[kk,ll] = c.GetPointId(ll)
        
        self.vertices = V
        self.faces = np.int_(F)
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.computeCentersLengths()



def mergeCurves(curves, tol=0.01):
    N = 0
    M = 0
    dim = curves[0].vertices.shape[1]
    for c in curves:
        N += c.vertices.shape[0]
        M += c.faces.shape[0]

    vertices = np.zeros([N,dim])
    faces = np.zeros([M,dim], dtype=int)
    N = 0
    M = 0
    for c in curves:
        N1 = c.vertices.shape[0]
        M1 = c.faces.shape[0]
        vertices[N:N+N1,:] = c.vertices
        faces[M:M+M1, :] = c.faces + N
        N += N1
        M += M1
        #print N,M
    dist = np.sqrt(((vertices[:, np.newaxis, :]-vertices[np.newaxis,:,:])**2).sum(axis=2))
    j=0
    openV = np.ones(N)
    refIndex = -np.ones(N)
    for k in range(N):
        if openV[k]:
            #vertices[j,:] = np.copy(vertices[k,:])
            J = np.nonzero((dist[k,:] < tol) * openV==1)
            J = J[0]
            openV[J] = 0
            refIndex[J] = j
            j=j+1
    vert2 = np.zeros([j, dim])
    for k in range(j):
        J = np.nonzero(refIndex==k)
        J = J[0]
        #print vertices[J]
        vert2[k,:] = vertices[J].sum(axis=0)/len(J)
        #print J, len(J), J.shape
    #vertices = vertices[0:j, :]
    #print faces
    faces = refIndex[faces]
    faces2 = np.copy(faces)
    j = 0
    for k in range(faces.shape[0]):
        if faces[k,1] != faces[k,0]:
            faces2[j,:] = faces[k,:]
            j += 1
            #print k,j
    faces2 = faces2[range(j)]
    return Curve(FV=(faces2,vert2))

# Reads several .byu files
def readMultipleCurves(regexp, Nmax = 0):
    files = glob.glob(regexp)
    if Nmax > 0:
        nm = min(Nmax, len(files))
    else:
        nm = len(files)
    fv1 = []
    for k in range(nm):
        fv1.append(Curve(files[k]))
    return fv1

# saves time dependent curves (fixed topology)
def saveEvolution(fileName, fv0, xt):
    fv = Curve(fv0)
    for k in range(xt.shape[0]):
        fv.vertices = np.squeeze(xt[k, :, :])
        fv.saveCurve(fileName+'{0: 02d}'.format(k)+'.byu')



# Current norm of fv1
def currentNorm0(fv1, KparDist):
    c2 = fv1.centers
    cr2 = fv1.lenel
    g11 = kfun.kernelMatrix(KparDist, c2)
    return np.multiply(np.dot(cr2, cr2.T), g11).sum()
        

# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct 
def currentNormDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 =fvDef.lenel
    c2 = fv1.centers
    cr2 = fv1.lenel
    g11 = kfun.kernelMatrix(KparDist, c1)
    g12 = kfun.kernelMatrix(KparDist, c2, c1)
    obj = (np.multiply(np.dot(cr1,cr1.T), g11).sum() - 2*np.multiply(np.dot(cr1, cr2.T), g12).sum())
    return obj

# Returns |fvDef - fv1|^2 for current norm
def currentNorm(fvDef, fv1, KparDist):
    return currentNormDef(fvDef, fv1, KparDist) + currentNorm0(fv1, KparDist) 

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentNormGradient(fvDef, fv1, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.lenel
    c2 = fv1.centers
    cr2 = fv1.lenel
    dim = c1.shape[1]
    #print cr2

    g11 = kfun.kernelMatrix(KparDist, c1)
    dg11 = kfun.kernelMatrix(KparDist, c1, diff=True)
    
    g12 = kfun.kernelMatrix(KparDist, c2, c1)
    dg12 = kfun.kernelMatrix(KparDist, c2, c1, diff=True)


    z1 = np.dot(g11, cr1) - np.dot(g12, cr2)
    dg11 = np.multiply(dg11 , np.dot(cr1, cr1.T))
    dg12 = np.multiply(dg12 , np.dot(cr1, cr2.T))

    dz1 = (np.multiply(dg11.sum(axis=1), c1.T).T - np.dot(dg11,c1)
           - np.multiply(dg12.sum(axis=1), c1.T).T + np.dot(dg12,c2))

    # xDef1 = xDef[fvDef.faces[:, 0], :]
    # xDef2 = xDef[fvDef.faces[:, 1], :]

    px = np.zeros([xDef.shape[0], dim])
    # ###########

    # crs = np.zeros(z1.shape)
    # #print crs, z1[:,1]
    # crs[:, 0] = np.squeeze(z1[:,1]) ;
    # crs[:, 1] = - np.squeeze(z1[:,0]) ;

    I = fvDef.faces[:,0]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - z1[k, :]

    I = fvDef.faces[:,1]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[k, :]


    return 2*px






# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    c2 = fv1.centers
    cr2 = fv1.lenel
    cr2 = np.sqrt((cr2**2).sum(axis=1))
    g11 = kfun.kernelMatrix(KparDist, c2)
    #print cr2.shape, g11.shape
    return np.dot(np.dot(cr2, g11), cr2.T).sum()
        
    
# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct 
def measureNormDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 = fvDef.lenel
    cr1 = np.sqrt((cr1**2).sum(axis=1)+1e-10)
    c2 = fv1.centers
    cr2 = fv1.lenel
    cr2 = np.sqrt((cr2**2).sum(axis=1)+1e-10)
    g11 = kfun.kernelMatrix(KparDist, c1)
    g12 = kfun.kernelMatrix(KparDist, c2, c1)
    #obj = (np.multiply(cr1*cr1.T, g11).sum() - 2*np.multiply(cr1*(cr2.T), g12).sum())
    obj = np.dot(np.dot(cr1, g11), cr1.T).sum() - 2* np.dot(np.dot(cr1, g12), cr2.T).sum()
    return obj

# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist) 


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.lenel
    c2 = fv1.centers
    cr2 = fv1.lenel
    dim = c1.shape[1]
    a1 = np.reshape(np.sqrt((cr1**2).sum(axis=1)+1e-10), (cr1.shape[0], 1))
    a2 = np.reshape(np.sqrt((cr2**2).sum(axis=1)+1e-10), (cr2.shape[0],1)) 
    cr1 = np.divide(cr1, a1)
    cr2 = np.divide(cr2, a2)

    g11 = kfun.kernelMatrix(KparDist, c1)
    dg11 = kfun.kernelMatrix(KparDist, c1, diff=True)
    
    g12 = kfun.kernelMatrix(KparDist, c2, c1)
    dg12 = kfun.kernelMatrix(KparDist, c2, c1, diff=True)


    z1 = np.dot(g11, a1) - np.dot(g12, a2)
    z1 = np.multiply(z1, cr1)
    dg1 = np.multiply(dg11, a1)
    dg11 = np.multiply(dg1, a1.T)
    dg1 = np.multiply(dg12, a1)
    dg12 = np.multiply(dg1, a2.T)

    dz1 = (np.multiply(dg11.sum(axis=1).reshape((-1,1)), c1) - np.dot(dg11,c1) - np.multiply(dg12.sum(axis=1).reshape((-1,1)), c1) + np.dot(dg12,c2))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]

    px = np.zeros([xDef.shape[0], dim])
    ###########

    I = fvDef.faces[:,0]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - z1[k, :]

    I = fvDef.faces[:,1]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[k, :]

    return 2*px

# class MultiSurface:
#     def __init__(self, pattern):
#         self.surf = []
#         files = glob.glob(pattern)
#         for f in files:
#             self.surf.append(Surface(filename=f))
