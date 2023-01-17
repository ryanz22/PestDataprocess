import numpy as np
from numpy.typing import NDArray
from typing import Tuple
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


def read_snd_file(
    fname,
    sr: int = 22050,
    offset: float = 0.0,
    duration: int = 60,
    mono: bool = True,
    scale: bool = True,
) -> Tuple[NDArray, int, float]:
    """_summary_
    By default, read_snd_file will load sound as sr=22050, mono=True,
    offset = 0.0, duration = 60s, scale = True
    Args:
        fname (_type_): _description_
        sr (int, optional): _description_. Defaults to 22050.
        offset (float, optional): _description_. Defaults to 0.0.
        duration (int, optional): _description_. Defaults to 60.
        mono (bool, optional): _description_. Defaults to True.
        scale (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[NDArray, int, float]: _description_
    """
    print(
        f"load {fname} sr: {sr} offset: {offset} duration: {duration} mono: {mono} scale: {scale}"
    )
    data, sr = librosa.load(
        fname, sr=sr, mono=mono, offset=offset, duration=duration, dtype=np.float32
    )
    logger.debug(f"data type: {data.dtype}")
    # logger.debug(data[2000:2020])
    from sklearn import preprocessing

    if scale:
        mean = data.mean()
        data = preprocessing.minmax_scale(data - mean, feature_range=(-1, 1))
    # logger.debug(data[2000:2020])
    duration = librosa.get_duration(y=data, sr=sr)
    logger.debug(f"data size should be {duration * sr * 4}")
    logger.debug(f"size of read_snd_file data: {data.size * data.itemsize}")

    return data, sr, duration
    # return data.astype(np.float16)
