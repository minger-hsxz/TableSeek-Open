import os
import datetime
import logging

def mklog(filename=''):
    now = datetime.datetime.now()
    file_path = 'log/' + now.strftime('%Y-%m-%d_%H-%M-%S') + '-' + filename
    filename = file_path + '/log.txt'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(pathname)s-%(lineno)d|: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return file_path