import logging
import numpy
from PIL import Image
import sys
import diffeomorphisms
import regularGrid

log_file_name = "metamorphosis.log"
compute_output_dir = "/cis/home/clr/compute/metamorphosis/"
image_dir = "/cis/home/clr/compute/meta_images/test_images/"
brain_image_dir = "/cis/home/clr/compute/meta_images/child/"
inho_image_dir = "/cis/home/clr/compute/meta_images/inho/"
phantoms_image_dir = "/cis/home/clr/compute/meta_images/phantoms/"
multiproc_pool_size = 16
multiproc_pool_timeout = 5000
file_write_iter = 5

def configure(sim, config_name):
    sim.config_name = config_name
    modname = globals()['__name__']
    module = sys.modules[modname]
    method = getattr(module, config_name)
    method(sim)

def eight(sim):
    sim.dim = 2
    sim.sigma = .05
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (40,40)
    #sim.domain_max = (1., 1.)
    sim.dx = (1., 1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-4
    sim.write_iter = file_write_iter

    sim.rg = regularGrid.RegularGrid(sim.dim, sim.num_points, \
                             sim.domain_max, sim.dx, "meta")
    sim.times = numpy.linspace(sim.time_min, sim.time_max, \
                            sim.num_times)
    sim.dt = (sim.time_max - sim.time_min) / (sim.num_times - 1)

    sim.cg_max_iter = 25
    sim.nonlinear_cg_max_iter = 25
    sim.pool_size = multiproc_pool_size
    sim.pool_timeout =  multiproc_pool_timeout

    sim.alpha = 10.
    sim.gamma = 1.
    sim.Lpower = 2.

    logging.info("eight image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))
    logging.info("kernel params- alpha: %f, gamma: %f, Lpower: %f" % \
                        (sim.alpha, sim.gamma, sim.Lpower))

    size = sim.num_points
    im1 = Image.open(image_dir + "eight_1c.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "eight_2c.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) / 255.
            tr[j,k] = ims[1].getpixel((j,k)) / 255.
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def inho(sim):
    sim.dim = 2
    sim.sigma = 1.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (256,124)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-5
    sim.write_iter = file_write_iter

    sim.rg = regularGrid.RegularGrid(sim.dim, sim.num_points, \
                             sim.domain_max, sim.dx, "meta")
    sim.times = numpy.linspace(sim.time_min, sim.time_max, \
                            sim.num_times)
    sim.dt = (sim.time_max - sim.time_min) / (sim.num_times - 1)

    sim.cg_max_iter = 5
    sim.nonlinear_cg_max_iter = 5
    sim.pool_size = multiproc_pool_size
    sim.pool_timeout =  multiproc_pool_timeout

    sim.alpha = 1.
    sim.gamma = 1.
    sim.Lpower = 2.

    logging.info("inho image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))
    logging.info("kernel params- alpha: %f, gamma: %f, Lpower: %f" % \
                        (sim.alpha, sim.gamma, sim.Lpower))

    size = sim.num_points
    im1 = Image.open(inho_image_dir + "TemplateSliceA.png").rotate(-90).resize(size)
    im2 = Image.open(inho_image_dir + "Target1SliceA.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k))
            tr[j,k] = ims[1].getpixel((j,k))
    sim.template_in = tp.ravel(order='F')
    sim.target_in = tr.ravel(order='F')
    scaleFactor = numpy.max(sim.template_in)
    #sim.template_in /= scaleFactor
    #sim.target_in /= scaleFactor

def phantoms(sim):
    sim.dim = 2
    sim.sigma = 1.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (100,100)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3
    sim.write_iter = file_write_iter

    sim.rg = regularGrid.RegularGrid(sim.dim, sim.num_points, \
                             sim.domain_max, sim.dx, "meta")
    sim.times = numpy.linspace(sim.time_min, sim.time_max, \
                            sim.num_times)
    sim.dt = (sim.time_max - sim.time_min) / (sim.num_times - 1)

    sim.cg_max_iter = 5
    sim.nonlinear_cg_max_iter = 5
    sim.pool_size = multiproc_pool_size
    sim.pool_timeout =  multiproc_pool_timeout

    sim.alpha = 1.
    sim.gamma = 1.
    sim.Lpower = 2.

    logging.info("phantoms image parameters: ")
    logging.info("dimension: %d" % (sim.dim))
    logging.info("num_points: %s" % (str(sim.rg.num_points)))
    logging.info("domain_max: %s" % (str(sim.rg.domain_max)))
    logging.info("dx: %s" % (str(sim.rg.dx)))
    logging.info("sigma: %f" % (sim.sigma))
    logging.info("dt: %f" % (sim.dt))
    logging.info("kernel params- alpha: %f, gamma: %f, Lpower: %f" % \
                        (sim.alpha, sim.gamma, sim.Lpower))

    size = sim.num_points
    im1 = Image.open(phantoms_image_dir + "Phantom.png").rotate(-90).resize(size)
    #im2 = Image.open(phantoms_image_dir + "BiasedPhantom.png").rotate(-90).resize(size)
    #im2 = Image.open(phantoms_image_dir + "TranslatedPhantom.png").rotate(-90).resize(size)
    im2 = Image.open(phantoms_image_dir + "TranslatedBiasedPhantom.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k))
            tr[j,k] = ims[1].getpixel((j,k))
    sim.template_in = tp.ravel(order='F')
    sim.target_in = tr.ravel(order='F')
