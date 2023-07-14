import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Callable
import logging
import librosa
import pathlib
from functools import partial
import functional as pyf
import yaml

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
        return librosa.util.fix_length(y, size=int(sr * dura))
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
    logger.debug(
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


def process_all(
    result: List[str],
    in_dir: str,
    out_dir: str,
    func: Callable[[str, str], str],
    out_ext: str,
    processes: int = 8,
    partition: int = 100,
):
    def inner_fn(fn, idir) -> str:
        """given idir is /test and fn is  /test/sub_1/file_1
        inner_fn will return sub_1/file_1
        """
        if fn.startswith(idir):
            nfn = fn[len(idir) :]
            if nfn.startswith("/"):
                nfn = nfn[1:]
            return nfn
        else:
            return f"error[inner_fn -> fn is not in idir]: fn {fn} idir {idir}"

    def output_fn(fn: str, outdir: str, ext: str) -> str:
        p = pathlib.Path(outdir)
        nfn = pathlib.Path.joinpath(p, fn)
        match ext:
            case "dir":  # like peaks, single input to multiple output in a folder
                nfn = nfn.parent
            case "auto":  # no need todo anything
                pass
            case _:  # like wav
                nfn = change_ext(str(nfn), f".{ext}")

        if not str(nfn).startswith(outdir):
            return f"error[output_fn -> fn is not in outdir]: fn {fn} outdir {outdir}"

        return str(nfn)

    # inner = inner_fn(result[0], in_dir)
    # print(f"inner: {inner}")
    # t2 = output_fn(inner, out_dir)
    # print(f"out fn: {t2}")
    # resize(result[0], t2, 512, 0)
    p_inner = partial(inner_fn, idir=in_dir)
    p_output_fn = partial(output_fn, outdir=out_dir, ext=out_ext)
    inner_fn_list = pyf.seq(result).map(p_inner).list()
    err = pyf.seq(inner_fn_list).filter(lambda s: s.startswith("error"))
    if err.non_empty():
        err.for_each(print)
        return

    out_fn_list = pyf.seq(inner_fn_list).map(p_output_fn).list()
    err2 = pyf.seq(out_fn_list).filter(lambda s: s.startswith("error"))
    if err2.non_empty():
        err2.for_each(print)
        return

    fn_pair_list = pyf.seq(result).zip(pyf.seq(out_fn_list)).list()
    print(fn_pair_list)
    # processes=None, partition_size=None
    # The following operations are run in parallel with more to be implemented
    # in a future release:
    #       map/select
    #       filter/filter_not/where
    #       flat_map
    pyf.pseq(fn_pair_list, processes=processes, partition_size=partition).map(
        lambda t: func(t[0], t[1])
    ).for_each(print)


def xeno_canto_meta(in_dir: str) -> List[Tuple[pathlib.Path, str]] | str:
    meta_files = [f for f in pathlib.Path(in_dir).glob(f"**/meta.yaml")]
    if not meta_files:
        return f"failed to find meta.yaml in folder {in_dir}"
    logger.debug(f"meta files:\n{meta_files}")

    def get_ds_name(fn: str) -> str:
        with open(fn, "r") as f:
            yml = yaml.safe_load(f)
            return yml["dataset name"]

    ds_names = pyf.seq(meta_files).map(get_ds_name).list()
    print(f"dataset name:\n{ds_names}")
    dup_ds_names = pyf.seq(ds_names).distinct().list()
    if len(dup_ds_names) != len(ds_names):
        return f"there are duplicate dataset names:\n{ds_names}\n{dup_ds_names}"

    meta_dsn_list = pyf.seq(meta_files).zip(pyf.seq(ds_names)).list()

    return meta_dsn_list


def split_list(my_list: list, factors: tuple[float, float, float]) -> list[list]:
    import math

    total_length = len(my_list)
    split_points = [math.ceil(total_length * factor) for factor in factors]

    s_list = []
    start = 0

    for split_point in split_points:
        end = start + split_point
        s_list.append(my_list[start:end])
        start = end

    return s_list
