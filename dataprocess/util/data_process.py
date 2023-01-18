import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Callable
import logging
import librosa
import pathlib
from functools import partial
import functional

from dataprocess.util.file import change_ext, check_create_folder, append_suffix

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


def process_all(result: List[str], in_dir: str, out_dir: str, func: Callable[[str, str], str],
                out_ext: str, processes: int = 8, partition: int = 100):
    def inner_fn(fn, idir) -> str:
        if fn.startswith(idir):
            nfn = fn[len(idir) :]
            if nfn.startswith("/"):
                nfn = nfn[1:]
            return nfn
        else:
            return f"error[inner_fn -> fn is not in idir]: fn {fn} idir {idir}"

    def output_fn(fn: str, outdir: str) -> str:
        p = pathlib.Path(outdir)
        nfn = str(pathlib.Path.joinpath(p, fn))
        nfn = change_ext(nfn, f".{out_ext}")
        if not nfn.startswith(outdir):
            return f"error[output_fn -> fn is not in outdir]: fn {fn} outdir {outdir}"

        return nfn

    # inner = inner_fn(result[0], in_dir)
    # print(f"inner: {inner}")
    # t2 = output_fn(inner, out_dir)
    # print(f"out fn: {t2}")
    # resize(result[0], t2, 512, 0)
    p_inner = partial(inner_fn, idir=in_dir)
    p_output_fn = partial(output_fn, outdir=out_dir)
    inner_fn_list = functional.seq(result).map(p_inner).list()
    err = functional.seq(inner_fn_list).filter(lambda s: s.startswith("error"))
    if err.non_empty():
        err.for_each(print)
        return

    out_fn_list = functional.seq(inner_fn_list).map(p_output_fn).list()
    err2 = functional.seq(out_fn_list).filter(lambda s: s.startswith("error"))
    if err2.non_empty():
        err2.for_each(print)
        return

    fn_pair_list = functional.seq(result).zip(functional.seq(out_fn_list)).list()
    print(fn_pair_list)
    # processes=None, partition_size=None
    # The following operations are run in parallel with more to be implemented
    # in a future release:
    #       map/select
    #       filter/filter_not/where
    #       flat_map
    functional.pseq(fn_pair_list, processes=processes, partition_size=partition).map(
        lambda t: func(t[0], t[1])
    ).for_each(print)