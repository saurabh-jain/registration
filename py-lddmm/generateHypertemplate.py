import os
from os import path
import glob
import argparse
import numpy as np
import diffeo
import surfaces
from vtk import *


def main():
    parser = argparse.ArgumentParser(description='Computes hypertemplate: chooses surface from a directory with average volume and retriangulates it as a smoother volume')
    parser.add_argument('dirIn', metavar='dirIn', type = str, help='input directory')
    parser.add_argument('fileout', metavar='fileout', type = str, help='output file') 
    parser.add_argument('--pattern', metavar = 'pattern', type = str, dest='pattern', default='*.byu', help='Regular expression for files to process (default: *.byu)')
    parser.add_argument('--targetSize', metavar = 'targetSize', type = int, dest = 'targetSize', default = 1000, help='targeted number of vertices')
    parser.add_argument('--imageDisc', metavar = 'imageDisc', type = float, dest = 'disc', default = 100, help='discretization step for triangulation')

    args = parser.parse_args()

    sf = surfaces.Surface()
    files = glob.glob(args.dirIn+'/'+args.pattern)
    z = np.zeros(len(files))
    k=0
    for name in files:
        fv = surfaces.Surface(filename = name)
        z[k] = np.fabs(fv.surfVolume())
        print name, z[k]
        k+=1

    mean = z.sum() / z.shape[0]
    print mean
    k0 = np.argmin(np.fabs(z-mean))
    fv = surfaces.Surface(filename = files[k0])
    minx = fv.vertices[:,0].min() 
    maxx = fv.vertices[:,0].max() 
    miny = fv.vertices[:,1].min() 
    maxy = fv.vertices[:,1].max() 
    minz = fv.vertices[:,2].min() 
    maxz = fv.vertices[:,2].max()

    dx = (maxx-minx)/ args.disc ;
    dy = (maxy-miny)/ args.disc ;
    dz = (maxz-minz)/ args.disc ;

    g = fv.toPolyData() ;
    h = vtkSelectEnclosedPoints() ;
    #h.SetSurface(g) ;
    h.Initialize(g) ;

    grd = np.mgrid[(minx-10*dx):(maxx+10*dx):dx, miny-10*dy:maxy+10*dy:dy, minz-10*dz:maxz+10*dz:dz]
    img = np.zeros([grd.shape[1], grd.shape[2], grd.shape[3]])
    for k1 in range(img.shape[0]):
        for k2 in range(img.shape[1]):
            for k3 in range(img.shape[2]):
                img[k1,k2,k3] = h.IsInsideSurface(grd[:,k1,k2,k3]) ;

    fv.Isosurface(img, (img.max() + img.min())/2, target = args.targetSize, smooth=1) ;
    fv.vertices = (np.array([minx, miny, minz]) - 10*np.array([dx, dy, dz])) + np.array([dx, dy, dz]) * fv.vertices ;

    fv.savebyu(args.fileout) 


if __name__=="__main__":
    main()
