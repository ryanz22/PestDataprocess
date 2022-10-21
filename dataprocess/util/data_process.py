import numpy as np

import logging
logger = logging.getLogger(__name__)


def replace_zeroes(data):
    min_nonzero = np.min(np.abs(data[np.nonzero(data)]))
    logger.debug(f'min_nonzero: {min_nonzero}')
    data[data == 0] = min_nonzero
    # data[data == 0] = 0.00001
    return data
