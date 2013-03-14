#!/opt/local/bin/python2.7
import os
from os import path
import argparse
import numpy as np
import diffeo
import pointSets
import surfaces
import surfaceMatching
from kernelFunctions import *
from affineRegistration import *
from surfaceMatching import *


def main():
    parser = argparse.ArgumentParser(description='runs surface matching registration over directories (relative to the template)')
    parser.add_argument('target', metavar='target', type = str, help='target')
    parser.add_argument('highfield', metavar='highfield', type = str, help='highfield (global)')
    parser.add_argument('highfield_parts', metavar='highfield_parts', nargs='+', type = str, help='highfield segments (list)')
    parser.add_argument('--sigmaKernel', metavar='sigmaKernel', type=float, dest='sigmaKernel', default = 6.5, help='kernel width') 
    parser.add_argument('--sigmaDist', metavar='sigmaDist', type=float, dest='sigmaDist', default = 2.5, help='kernel width (error term); (default = 2.5)') 
    parser.add_argument('--sigmaError', metavar='sigmaError', type=float, dest='sigmaError', default = 1.0, help='std error; (default = 1.0)') 
    parser.add_argument('--typeError', metavar='typeError', type=str, dest='typeError', default = 'measure', help='type error term (default: measure)') 
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--tmpOut', metavar = 'tmpOut', type = str, dest = 'tmpOut', default = '', help='info files directory')
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = '.'

    if args.tmpOut == '':
        args.tmpOut = args.dirOut + '/tmp'
    if not os.path.exists(args.dirOut):
        os.makedirs(args.dirOut)
    if not os.path.exists(args.tmpOut):
        os.makedirs(args.tmpOut)


    targ = surfaces.Surface(filename=args.target)
    hf = surfaces.Surface(filename=args.highfield)
    hfSeg = []
    for name in args.highfield_parts:
        hfSeg.append(surfaces.Surface(filename=name))
    nsub = len(hfSeg)
    nvSeg = np.zeros(nsub)
    
    for k in range(nsub):
        nvSeg[k] = np.int_(hfSeg[k].vertices.shape[0]) ;
    


    # Find Labels
    nv = hf.vertices.shape[0] 
    print 'vertices', nv, nvSeg
    dist = np.zeros([nv,nsub])
    for k in range(nsub):
        dist[:,k] = ((hf.vertices.reshape(nv, 1, 3) - hfSeg[k].vertices.reshape(1, nvSeg[k], 3))**2).sum(axis=2).min(axis=1)
    hfLabel = 1+dist.argmin(axis=1) ;
    K1 = Kernel(name='gauss', sigma = args.sigmaKernel)
    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=args.sigmaDist, sigmaError=args.sigmaError, errorType=args.typeError)

    R0, T0 = rigidRegistration(surfaces = (hf.vertices, targ.vertices),  verb=False, temperature=10., annealing=True)
    hf.vertices = np.dot(hf.vertices, R0.T) + T0
    u = path.split(args.highfield)
    [nm,ext] = path.splitext(u[1])
    print hfLabel
    hf.saveVTK(args.dirOut+'/'+nm+'.vtk', scalars=hfLabel, scal_name='Labels')

    print 'Starting Matching'
    f = SurfaceMatching(Template=hf, Target=targ, outputDir=args.tmpOut,param=sm, testGradient=False,
                        maxIter=1000, affine= 'none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.optimizeMatching()
    u = path.split(args.target)
    [nm,ext] = path.splitext(u[1])
    f.fvDef.savebyu(args.dirOut+'/'+nm+'Def.byu')
    nvTarg = targ.vertices.shape[0]
    closest = np.int_(((f.fvDef.vertices.reshape(nv, 1, 3) - targ.vertices.reshape(1, nvTarg, 3))**2).sum(axis=2).argmin(axis=0))
    targLabel = np.zeros(nvTarg) ;
    for k in range(nvTarg):
        targLabel[k] = hfLabel[closest[k]]
    targ.saveVTK(args.dirOut+'/'+nm+'.vtk', scalars=targLabel, scal_name='Labels')
        

if __name__=="__main__":
    main()
