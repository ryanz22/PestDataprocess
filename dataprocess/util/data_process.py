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
    duration: float | None = None,
    mono: bool = True,
    scale: bool = True,
) -> Tuple[NDArray, int, float]:
    """_summary_
    By default, read_snd_file will load the WHOLE file as sr=22050, mono=True,
    offset = 0.0, scale = True
    Args:
        fname (_type_): _description_
        sr (int, optional): _description_. Defaults to 22050.
        offset (float, optional): _description_. Defaults to 0.0.
        duration (float | None, optional): seconds to load; None loads whole file.
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
    if scale:
        data = data - data.mean()
        span = data.max() - data.min()
        span = span if span > 0 else 1.0
        data = 2.0 * (data - data.min()) / span - 1.0
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
        if not fn.startswith(idir):
            raise ValueError(f"fn is not in idir: fn {fn} idir {idir}")
        return fn[len(idir) :].lstrip("/")

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
            raise ValueError(f"fn is not in outdir: fn {fn} outdir {outdir}")

        return str(nfn)

    p_inner = partial(inner_fn, idir=in_dir)
    p_output_fn = partial(output_fn, outdir=out_dir, ext=out_ext)
    inner_fn_list = pyf.seq(result).map(p_inner).list()
    out_fn_list = pyf.seq(inner_fn_list).map(p_output_fn).list()

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


def xeno_canto_meta(in_dir: str) -> List[Tuple[pathlib.Path, str]]:
    meta_files = [f for f in pathlib.Path(in_dir).glob(f"**/meta.yaml")]
    if not meta_files:
        raise ValueError(f"failed to find meta.yaml in folder {in_dir}")
    logger.debug(f"meta files:\n{meta_files}")

    def get_ds_name(fn: str) -> str:
        with open(fn, "r") as f:
            yml = yaml.safe_load(f)
            return yml["dataset name"]

    ds_names = pyf.seq(meta_files).map(get_ds_name).list()
    print(f"dataset name:\n{ds_names}")
    dup_ds_names = pyf.seq(ds_names).distinct().list()
    if len(dup_ds_names) != len(ds_names):
        raise ValueError(f"there are duplicate dataset names:\n{ds_names}\n{dup_ds_names}")

    meta_dsn_list = pyf.seq(meta_files).zip(pyf.seq(ds_names)).list()

    return meta_dsn_list


def split_list(my_list: list, factors: tuple[float, float, float]) -> list[list]:
    import math

    total_length = len(my_list)
    split_points = [math.floor(total_length * factor) for factor in factors]
    left_over = total_length - sum(split_points)
    if left_over != 0:
        # remainder goes to the last split to keep total == len(my_list)
        split_points[-1] += left_over

    s_list = []
    start = 0

    for split_point in split_points:
        end = start + split_point
        s_list.append(my_list[start:end])
        start = end

    return s_list
