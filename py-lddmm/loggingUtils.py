
import logging

def setup_default_logging(output_dir=None, config=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    if config == None:
        log_file_name = "log.txt"
    else:
        log_file_name = config.log_file_name
    if output_dir == None:
        output_dir = "./"
    fh = logging.FileHandler("%s/%s" % (output_dir, log_file_name))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
