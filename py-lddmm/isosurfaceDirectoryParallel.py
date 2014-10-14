#!/opt/local/bin/python2.7
import os
from os import path
import glob
import argparse
import diffeo
import surfaces
import multiprocessing as mp


def process_file(name, args):
    sf = surfaces.Surface()
    print 'Processing ', name
    u = path.split(name)
    [nm,ext] = path.splitext(u[1])
    v = diffeo.gridScalars(fileName=name, force_axun = args.axun, withBug=args.withbug)
    if args.zeroPad:
        v.zeroPad(1)
    #print v.resol
    #if args.smooth:
    #    sf.Isosurface(v.data, value=0.5, target = args.targetSize, scales = v.resol, smooth=.5)
    #else:
    #    sf.Isosurface(v.data, value=0.5, target = args.targetSize, scales = v.resol, smooth =-1)
    t =  0.5 * (v.data.max() + v.data.min())
    #print v.resol
    if args.smooth:
        sf.Isosurface(v.data, value=t, target = args.targetSize, scales = v.resol, smooth=.75)
    else:
        sf.Isosurface(v.data, value=t, target = args.targetSize, scales = v.resol, smooth =-1)

    sf.edgeRecover()
    #print sf.surfVolume()
    sf.savebyu(args.dirOut+'/'+nm+'.byu')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Computes isosurfaces over directories')
    parser.add_argument('dirIn', metavar='dirIn', type = str, help='input directory')
    parser.add_argument('--pattern', metavar = 'pattern', type = str, dest='pattern', default='*.hdr', help='Regular expression for files to process (default: *.hdr)')
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--targetSize', metavar = 'targetSize', type = int, dest = 'targetSize', default = 1000, help='targeted number of vertices')
    parser.add_argument('--force-axial-unflipped', action = 'store_true', dest='axun', default=False,help='force reading .img files using axial unflipped order')
    parser.add_argument('--with-bug', action = 'store_true', dest='withbug', default=False,help='for back compatibility')
    parser.add_argument('--smooth', action = 'store_true', dest='smooth', default=False,help='adds smoothing step to triangulation')
    parser.add_argument('--zeroPad', action = 'store_true', dest='zeroPad', default=False,help='inserts a layer of zeros around the image before triangulation')
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = args.dirIn

    if path.exists(args.dirOut)==False:
        os.mkdir(args.dirOut)
    
    pool = mp.Pool()
    for name in  glob.glob(args.dirIn+'/'+args.pattern):
        pool.apply_async(process_file, args=(name, args))
    pool.close()
    pool.join()




