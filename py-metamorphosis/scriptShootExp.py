import numpy
import smoothImageNoJ
import smoothImageConfig
import optparse
import shutil
import os
import logging
import loggingUtils
#from tvtk.api import tvtk

output_directory_base = smoothImageConfig.compute_output_dir
parser = optparse.OptionParser()
parser.add_option("-o", "--output_dir", dest="output_dir")
parser.add_option("-c", "--config_name", dest="config_name")
(options, args) = parser.parse_args()
output_dir = output_directory_base + options.output_dir
# remove any old results in the output directory
if os.access(output_dir, os.F_OK):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
loggingUtils.setup_default_logging(output_dir, smoothImageConfig)
logging.info(options)
letter_match = (11,18,0)
sim = smoothImageNoJ.SmoothImageMeta(output_dir, options.config_name, letter_match)
sim.computeMatching()

# digList = numpy.array([0,1,25])
# #digList = numpy.arange(39)
# digWeight = (1./len(digList)) * numpy.ones_like(digList)
# print digWeight
# mat = numpy.zeros((39, 24**2))
# for d in range(len(digList)):
#     file1="/cis/home/clr/compute/output/smoothImage_meta/letterB_11_18_%s/final_mesh24_%s.vts"
#     r = tvtk.XMLStructuredGridReader(file_name=file1 % (digList[d],0))
#     r.update()
#     alpha1 = numpy.array(r.output.point_data.get_array("alpha")).astype(float)
#     sim.alpha += digWeight[d] * alpha1
#     mat[d,:] = alpha1
# #import pdb
# #pdb.set_trace()
# sim.shoot()
# sim.writeData("shoot")
