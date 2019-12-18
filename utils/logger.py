import logging
import time
import os
from logging import handlers
import datetime
import tensorflow as tf
debug=True

def _p(tensor,msg):
    if (debug):
        dt = datetime.datetime.now().strftime('TF_DEBUG: %m-%d %H:%M:%S: ')
        msg = dt +  msg
        return tf.Print(tensor, [tensor], msg,summarize= 100)
    else:
        return tensor


def _p_shape(tensor,msg):
    if (debug):
        dt = datetime.datetime.now().strftime('TF_DEBUG: %m-%d %H:%M:%S: ')
        msg = dt +  msg
        return tf.Print(tensor, [tf.shape(tensor)], msg,summarize= 100)
    else:
        return tensor


def init(level=logging.DEBUG,when="D",backup=7,_format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d行 %(message)s"):
    train_start_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    filename = 'logs/ocr-attention-'+train_start_time + '.log'
    _dir = os.path.dirname(filename)
    if not os.path.isdir(_dir):os.makedirs(_dir)

    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter(_format)
        logger.setLevel(level)

        handler = handlers.TimedRotatingFileHandler(filename, when=when, backupCount=backup,encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
