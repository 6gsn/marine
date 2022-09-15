from __future__ import absolute_import, print_function, with_statement

import logging
import os
from os.path import dirname

format = "[%(asctime)s][%(name)s:%(module)s][%(levelname)s] - %(message)s"


def getLogger(verbose=0, filename=None, name="marine"):
    handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format))
    handlers.append(stream_handler)

    if filename is not None:
        os.makedirs(dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(format))
        handlers.append(file_handler)

    logging.basicConfig(handlers=handlers)

    logger = logging.getLogger(name)
    if verbose >= 100:
        logger.setLevel(logging.DEBUG)
    elif verbose > 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    return logger
