import logging
import numpy
from PIL import Image
import sys
import os
import diffeomorphisms

where = os.environ["HOME"] 
if where.startswith('/Users'):
    compute_path = os.environ["HOME"] + '/Development/Results/py-meta'
    image_dir =  os.environ["HOME"] + "/Development/Data/meta_images/test_images/"
    inho_image_dir =  compute_path + "/Development/Data/meta_images/inho/"
else:
    compute_path = os.environ["HOME"] + '/MorphingData/py-meta'
    image_dir =  os.environ["HOME"] + "/IMAGES/meta_images/test_images/"
    inho_image_dir =  compute_path + "/IMAGES/meta_images/inho/"

log_file_name = "metamorphosis.log"
compute_output_dir = compute_path + "/"
phantoms_image_dir = os.environ["HOME"] + "/IMAGES/meta_images/phantoms/"
file_write_iter = 1

def configure(sim, config_name):
    sim.config_name = config_name
    modname = globals()['__name__']
    module = sys.modules[modname]
    print module
    method = getattr(module, config_name)
    method(sim)

def letter(sim):
    sim.dim = 2
    sim.sigma = .5
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (24,24)
    #sim.domain_max = (36., 36.)
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-6
    sim.write_iter = file_write_iter
    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 1.5
    sim.khs = .5
    sim.kvo = 4
    sim.kho = 1
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))

    import scipy.io
    data = scipy.io.loadmat(image_dir + "binaryalphadigs.mat")["dat"]
    img1 =  data[sim.letter_match[0],sim.letter_match[1]]
    img2 = data[sim.letter_match[0],sim.letter_match[2]]
    tp = numpy.zeros(sim.num_points)
    tr = numpy.zeros(sim.num_points)
    for k in range(16):
        for j in range(20):
            tp[2+j,4+k] = img1[19-j,k]
            tr[2+j,4+k] = img2[19-j,k]
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def d72(sim):
    sim.dim = 2
    sim.sigma = 3.0
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    down_factor = 3
    assert 72 % down_factor == 0, "down_factor does not divide 72."
    sim.num_points = (72 / down_factor, 72/down_factor)
    sim.domain_max = (36., 36.)
    #sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-6
    sim.write_iter = file_write_iter
    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 3.
    sim.khs = .6
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
    size = (72,72)
    im1 = Image.open(image_dir + "d72_1.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "d72_2.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k))
            tr[j,k] = ims[1].getpixel((j,k))

    #tp_new = numpy.zeros(sim.num_points)
    #tr_new = numpy.zeros(sim.num_points)
    tp_new = tp[0:size[0]:down_factor,0:size[0]:down_factor]
    tr_new = tr[0:size[0]:down_factor,0:size[0]:down_factor]
    sim.template_in = tp_new.ravel()
    sim.target_in = tr_new.ravel()

def d72_unit_cube(sim):
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
    sim.write_iter = file_write_iter

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

def eight(sim):
    sim.dim = 2
    sim.sigma = .25
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (40,40)
    #sim.domain_max = (1., 1.)
    sim.dx = (1., 1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-4
    sim.write_iter = file_write_iter

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 2.0
    sim.khs = .5
    sim.kvo = 4
    sim.kho = 2
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

def eight_unit_cube(sim):
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
    sim.write_iter = file_write_iter

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

def leaf200(sim):
    sim.dim = 2
    sim.sigma = .2
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (200,200)
    #sim.domain_max = (1., 1.)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-5
    sim.write_iter = file_write_iter

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    #sim.kvs = .03 / 2.0
    #sim.khs = .015 / 3.0 / 1.5
    sim.kvs = 2. # 2.
    sim.khs = .2
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
    size = sim.num_points
    im1 = Image.open(image_dir + "leaf200_1_reg.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "leaf200_2.png").rotate(-90).resize(size)
    #im1 = Image.open(image_dir + "leaf200_1.png").rotate(-90).resize(size)
    #im2 = Image.open(image_dir + "leaf200_3.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) #/ 255.
            tr[j,k] = ims[1].getpixel((j,k)) #/ 255.
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def leaf100(sim):
    sim.dim = 2
    sim.sigma = 0.1
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (100,100)
    #sim.domain_max = (1., 1.)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-4
    sim.write_iter = file_write_iter

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 1.5
    sim.khs = .5
    sim.kvo = 4
    sim.kho = 2
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))
    size = sim.num_points
    im1 = Image.open(image_dir + "leaf100_1_reg.png").rotate(-90).resize(size)
    im2 = Image.open(image_dir + "leaf100_2.png").rotate(-90).resize(size)
    #im1 = Image.open(image_dir + "leaf100_1.png").rotate(-90).resize(size)
    #im2 = Image.open(image_dir + "leaf100_3.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k)) #/ 255.
            tr[j,k] = ims[1].getpixel((j,k)) #/ 255.
    sim.template_in = tp.ravel()
    sim.target_in = tr.ravel()

def brains(sim):
    sim.dim = 2
    sim.sigma = .5
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (90,110)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3
    sim.write_iter = file_write_iter
    brain_image_dir =  os.environ["HOME"] +  "/IMAGES/meta_images/child/"
    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 3.
    sim.khs = .1
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
    sim.target_in = sim.sc.data.reshape(num_nodes)
    sim.sc = diffeomorphisms.gridScalars()
    sim.sc.loadAnalyze(brain_image_dir + "adult_2d.hdr")
    #sim.target_in = sim.sc.data[45,...].reshape(num_nodes)
    sim.template_in = sim.sc.data.reshape(num_nodes)

def phantoms(sim):
    sim.dim = 2
    sim.sigma = .25
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (100,100)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3
    sim.write_iter = file_write_iter

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 3.
    sim.khs = .2
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))

    size = sim.num_points
    im1 = Image.open(phantoms_image_dir + "Phantom.png").rotate(-90).resize(size)
    #im2 = Image.open(phantoms_image_dir + "BiasedPhantom.png").rotate(-90).resize(size)
    im2 = Image.open(phantoms_image_dir + "TranslatedPhantom.png").rotate(-90).resize(size)
    #im2 = Image.open(phantoms_image_dir + "TranslatedBiasedPhantom.png").rotate(-90).resize(size)
    ims = [im1, im2]
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tp[j,k] = ims[0].getpixel((j,k))
            tr[j,k] = ims[1].getpixel((j,k))
    sim.template_in = tp.ravel(order='F')
    sim.target_in = tr.ravel(order='F')

def cell(sim):
    sim.dim = 2
    sim.sigma = .25
    sim.sfactor = 1./numpy.power(sim.sigma, 2)
    sim.num_points = (78,108)
    sim.domain_max = None
    sim.dx = (1.,1.)
    sim.num_times = 11
    sim.time_min = 0.
    sim.time_max = 1.
    sim.cg_init_eps = 1e-3
    sim.write_iter = file_write_iter

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 3.
    sim.khs = .2
    sim.kvo = 4
    sim.kho = 4
    logging.info("KV params: name=%s, sigma=%f, order=%f" \
                        % (sim.kvn,sim.kvs,sim.kvo))
    logging.info("KH params: name=%s, sigma=%f, order=%f" \
                        % (sim.khn,sim.khs,sim.kho))

    size = sim.num_points
    im2 = Image.open(image_dir + "helasmall_rescaled.png").rotate(-90).resize(size)
    #im2 = Image.open(phantoms_image_dir + "TranslatedBiasedPhantom.png").rotate(-90).resize(size)
    tp = numpy.zeros(size)
    tr = numpy.zeros(size)
    for j in range(size[0]):
        for k in range(size[1]):
            tr[j,k] = im2.getpixel((j,k))
    sim.template_in = tp.ravel(order='F')
    sim.target_in = tr.ravel(order='F')


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
    sim.write_iter = file_write_iter

    sim.kvn = 'laplacian'
    sim.khn = 'laplacian'
    sim.kvs = 10.
    sim.khs = .3
    sim.kvo = 4
    sim.kho = 2
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
