#!/usr/bin/python
# -*- encoding: utf-8 -*-
""" Basic Logger setup file
"""


import os
import logging
import time


def get_logger(log_dir = "train_log",time_str = ""):
    """Create an logging directory,with filename "train-{time_str}", return an logging object 
    
    Args:
        log_dir (str, optional): Directory to where the log are
        time_str (str, optional): Time in string format to generate distinctive file
    
    Returns:
        logging.Logger Object: logging object
    """
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logfile = f'train-{time_str}.log'
    logfile = os.path.join(log_dir, logfile)
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    return logger