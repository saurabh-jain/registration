import numpy as np
import scipy as sp
import os



def loadlmk(filename, dim=3):
# [x, label] = loadlmk(filename, dim)
# Loads 3D landmarks from filename in .lmk format.
# Determines format version from first line in file
#   if version number indicates scaling and centering, transform coordinates...
# the optional parameter s in a 3D scaling factor

    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            versionNum = 1
            versionStrs = ln0.split("-")
            if len(versionStrs) == 2:
                try:
                    versionNum = int(float(versionStrs[1]))
                except:
                    pass

            #print fn
            ln = fn.readline().split()
            #print ln0, ln
            N = int(ln[0])
            #print 'reading ', filename, ':', N, ' landmarks'
            x = np.zeros([N, dim])
            label = []

            for i in range(N):
                ln = fn.readline()
                label.append(ln) 
                ln0 = fn.readline().split()
                #print ln0
                for k in range(dim):
                    x[i,k] = float(ln0[k])
            if versionNum >= 6:
                lastLine = ''
                nextToLastLine = ''
                # read the rest of the file
                # the last two lines contain the center and the scale variables
                while 1:
                    thisLine = fn.readline()
                    if not thisLine:
                        break
                    nextToLastLine = lastLine
                    lastLine = thisLine
                    
                centers = nextToLastLine.rstrip('\r\n').split(',')
                scales = lastLine.rstrip('\r\n').split(',')
                if len(scales) == dim and len(centers) == dim:
                    if scales[0].isdigit and scales[1].isdigit and scales[2].isdigit and centers[0].isdigit and centers[1].isdigit and centers[2].isdigit:
                        x[:, 0] = x[:, 0] * float(scales[0]) + float(centers[0])
                        x[:, 1] = x[:, 1] * float(scales[1]) + float(centers[1])
                        x[:, 2] = x[:, 2] * float(scales[2]) + float(centers[2])
                
    except IOError:
        print 'cannot open ', filename
        raise
    return x, label




def  savelmk(x, filename):
# savelmk(x, filename)
# save landmarks in .lmk format.

    with open(filename, 'w') as fn:
        str = 'Landmarks-1.0\n {0: d}\n'.format(x.shape[0])
        fn.write(str)
        for i in range(x.shape[0]):
            str = '"L-{0:d}"\n'.format(i)
            fn.write(str)
            str = ''
            for k in range(x.shape[1]):
                str = str + '{0: f} '.format(x[i,k])
            str = str + '\n'
            fn.write(str)
        fn.write('1 1 \n')
