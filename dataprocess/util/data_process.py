import numpy as np
from numpy.typing import NDArray
import logging
import librosa

logger = logging.getLogger(__name__)


def replace_zeroes(data):
    min_nonzero = np.min(np.abs(data[np.nonzero(data)]))
    logger.debug(f"min_nonzero: {min_nonzero}")
    data[data == 0] = min_nonzero
    # data[data == 0] = 0.00001
    return data


def fill_fix_len(y: NDArray, sr: int, dura: float):
    tlen = librosa.get_duration(y=y, sr=sr)
    if tlen < dura:
        fill_len = int((dura - tlen) * sr)
        a = np.empty(fill_len)
        return np.append(y, a)
    else:
        return y
