import logging

def setup_default_logging(output_dir=None, config=None, fileName=None, stdOutput=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(asctime)s-%(levelname)s: %(message)s")
    formatter = logging.Formatter("%(message)s")
    if config == None:
        log_file_name = fileName
    else:
        log_file_name = config.log_file_name
    if output_dir == None:
        output_dir = ""
    if fileName != None:
        print 'creating file handler for ', "%s%s" % (output_dir, log_file_name)
        fh = logging.FileHandler("%s/%s" % (output_dir, log_file_name), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if stdOutput:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
