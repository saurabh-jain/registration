import logging
import numpy
from PIL import Image
import sys

log_file_name = "metamorphosis.log"
compute_output_dir = "/cis/home/clr/compute/smoothImage_meta/"
image_dir = "/cis/home/clr/compute/meta_images/test_images/"

def configure(sim, config_name):
    sim.config_name = config_name
    modname = globals()['__name__']
    module = sys.modules[modname]
    method = getattr(module, config_name)
    method(sim)

def d72(sim):
    sim.dim = 2
    sim.sigma = 10.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (72,72)
    sim.domain_max = (1., 1.)
    sim.dx = None
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = .07
    sim.khs = .015 / 2.0
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
    size = sim.num_points
    im1 = Image.open(image_dir + "d72_1.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "d72_2.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) / 255.
            tr[j,k] = ims[1].getpixel((j,k)) / 255.
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def leaf200(sim):
    sim.dim = 2
    sim.sigma = 15.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (200,200)
    sim.domain_max = (1., 1.)
    sim.dx = None
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = .03
    sim.khs = .015 / 3.0 / 2.0
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
    size = sim.num_points
    im1 = Image.open(image_dir + "leaf200_1_reg.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "leaf200_2.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) / 255.
            tr[j,k] = ims[1].getpixel((j,k)) / 255.
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def eight(sim):
    sim.dim = 2
    sim.sigma = 15.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (40,40)
    sim.domain_max = (1., 1.)
    sim.dx = None
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = .07
    sim.khs = .015
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
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
