#!/opt/local/bin/python2.7
from os import path
import glob
import argparse
import diffeo
import surfaces


def main():
    parser = argparse.ArgumentParser(description='Computes isosurfaces over directories')
    parser.add_argument('dirIn', metavar='dirIn', type = str, help='input directory')
    parser.add_argument('--pattern', metavar = 'pattern', type = str, dest='pattern', default='*.hdr', help='Regular expression for files to process (default: *.hdr)')
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--targetSize', metavar = 'targetSize', type = int, dest = 'targetSize', default = 1000, help='targeted number of vertices')
    parser.add_argument('--force-axial-unflipped', action = 'store_true', dest='axun', default=False,help='force reading .img files using axial unflipped order')
    parser.add_argument('--smooth', action = 'store_true', dest='smooth', default=False,help='adds smoothing step to triangulation')
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = args.dirIn

    sf = surfaces.Surface()
    for name in glob.glob(args.dirIn+'/'+args.pattern):
        print 'Processing ', name
        u = path.split(name)
        [nm,ext] = path.splitext(u[1])
        v = diffeo.gridScalars(fileName=name, force_axun = args.axun)
        #print v.resol
        if args.smooth:
            sf.Isosurface(v.data, value=0.5, target = args.targetSize, scales = v.resol, smooth=.75)
        else:
            sf.Isosurface(v.data, value=0.5, target = args.targetSize, scales = v.resol, smooth =-1)
        sf.edgeRecover()
        #print sf.surfVolume()
        sf.savebyu(args.dirOut+'/'+nm+'.byu')

if __name__=="__main__":
    main()
