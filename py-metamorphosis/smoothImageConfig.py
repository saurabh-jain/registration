import logging
import numpy
from PIL import Image
import sys
import diffeomorphisms

log_file_name = "metamorphosis.log"
compute_output_dir = "/cis/home/clr/compute/smoothImage_meta/"
image_dir = "/cis/home/clr/compute/meta_images/test_images/"
brain_image_dir = "/cis/home/clr/compute/meta_images/child/"
inho_image_dir = "/cis/home/clr/compute/meta_images/inho/"

def configure(sim, config_name):
    sim.config_name = config_name
    modname = globals()['__name__']
    module = sys.modules[modname]
    method = getattr(module, config_name)
    method(sim)

def d72(sim):
    sim.dim = 2
    sim.sigma = 5.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (72,72)
    #sim.domain_max = (1., 1.)
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-6

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 3.
    sim.khs = .15
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
            tp[j,k] = ims[0].getpixel((j,k))
            tr[j,k] = ims[1].getpixel((j,k))
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def d72_OLD(sim):
    sim.dim = 2
    sim.sigma = 20.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (72,72)
    sim.domain_max = (1., 1.)
    sim.dx = None
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = .07
    sim.khs = .015 / 1.5 #/ 2.0
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
    sim.sigma = 5.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (200,200)
    #sim.domain_max = (1., 1.)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-5

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    #sim.kvs = .03 / 2.0
    #sim.khs = .015 / 3.0 / 1.5
    sim.kvs = 7. # 2.
    sim.khs = .4
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
    size = sim.num_points
    #im1 = Image.open(image_dir + "leaf200_1_reg.png").rotate(-90).resize(size)
    #im2 = Image.open(image_dir + "leaf200_2.png").rotate(-90).resize(size)
    im1 = Image.open(image_dir + "leaf200_1.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "leaf200_3.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) #/ 255.
            tr[j,k] = ims[1].getpixel((j,k)) #/ 255.
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def eight(sim):
    sim.dim = 2
    sim.sigma = 10. #15.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (40,40)
    sim.domain_max = (1., 1.)
    sim.dx = None
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3

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


def brains(sim):
    sim.dim = 2
    sim.sigma = 3.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (90,110)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 1.5
    sim.khs = .2
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))

    num_nodes = 90*110
    sim.sc = diffeomorphisms.gridScalars()
    sim.sc.loadAnalyze(brain_image_dir + "child_2d.hdr")
    #sim.template_in = sim.sc.data[45,...].reshape(num_nodes)
    sim.template_in = sim.sc.data.reshape(num_nodes)
    sim.sc = diffeomorphisms.gridScalars()
    sim.sc.loadAnalyze(brain_image_dir + "adult_2d.hdr")
    #sim.target_in = sim.sc.data[45,...].reshape(num_nodes)
    sim.target_in = sim.sc.data.reshape(num_nodes)

def inho(sim):
    sim.dim = 2
    sim.sigma = 12.
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (256,124)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 10.
    sim.khs = .3
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))

    size = sim.num_points
    im1 = Image.open(inho_image_dir + "TemplateSliceA.png").rotate(-90).resize(size)
    im2 = Image.open(inho_image_dir + "Target1SliceA.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) / 255.
            tr[j,k] = ims[1].getpixel((j,k)) / 255.
    sim.template_in = tp.ravel(order='F')
    sim.target_in = tr.ravel(order='F')
