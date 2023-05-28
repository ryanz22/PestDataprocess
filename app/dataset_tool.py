import os
import sys
import pathlib
import numpy as np
import pprint
from typing import Tuple, List
import click
import logging
import functional as pyf
import shutil
import soundfile as sf
from functools import partial

from dataprocess.util.file import (
    copy_dir_only,
    check_create_folder,
    change_ext,
    append_suffix,
)
from dataprocess.util.data_process import process_all, xeno_canto_meta
from dataprocess.cwt.scalogram import (
    plot_spectro,
    plot_scalo,
)
from dataprocess.util.data_process import read_snd_file

from dataprocess.sound.preprocess import snd_peaks, normalize

from dataprocess.sound.sep_data import create_sep_dataset


@click.group()
def cli():
    pass


# https://medium.com/nerd-for-tech/easily-split-your-directory-into-train-validation-and-testing-format-f1359f34dd93
# https://github.com/jfilter/split-folders

# Use splitfolders tool directly
# Usage:
#    splitfolders [--output] [--ratio] [--fixed] [--seed] [--oversample] [--group_prefix] [--move] folder_with_images
# Options:
#    --output        path to the output folder. defaults to `output`. Get created if non-existent.
#    --ratio         the ratio to split. e.g. for train/val/test `.8 .1 .1 --` or for train/val `.8 .2 --`.
#    --fixed         set the absolute number of items per validation/test set. The remaining items constitute
#                    the training set. e.g. for train/val/test `100 100` or for train/val `100`.
#                    Set 3 values, e.g. `300 100 100`, to limit the number of training values.
#    --seed          set seed value for shuffling the items. defaults to 1337.
#    --oversample    enable oversampling of imbalanced datasets, works only with --fixed.
#    --group_prefix  split files into equally-sized groups based on their prefix
#    --move          move the files instead of copying
# Example:
#    splitfolders --ratio .8 .1 .1 --output folder_name folder_with_images


@cli.command(help="Split dataset into train, val, test")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--train", type=float, default=0.7)
@click.option("--val", type=float, default=0.2)
@click.option("--test", type=float, default=0.1)
def split_folders(in_dir: str, out_dir: str, train: float, val: float, test: float):
    """Split dataset into train, val, test"""
    import splitfolders as sf
    import math

    # https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    total = train + val + test
    if not math.isclose(total, 1.0):
        print(
            f"wrong split ratio total: {total}, train: {train}, val: {val}, test: {test}"
        )
        return

    sf.ratio(
        in_dir,
        output=out_dir,
        seed=43,
        ratio=(train, val, test),
        group_prefix=None,
        move=False,
    )


@cli.command(help="Resize images recursively")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=False, dir_okay=True)
)
@click.option("--ext", type=str, required=True)
@click.option("--width", type=int, required=True)
@click.option("--height", type=int, default=0)
def resize_all_img(in_dir: str, out_dir: str, ext: str, width: int, height: int):
    """Resize images recursively"""
    if width <= 0 or height < 0:
        print("width must > 0 and height must >= 0")
        return

    result = [str(f) for f in pathlib.Path(in_dir).glob(f"**/*.{ext}")]
    print(f"result:\n{result}")
    if not result:
        print(f"failed to find *.{ext} in folder {in_dir}")
        return

    import cv2

    def resize(fn: str, ofn: str, rwidth: int, rheight: int) -> str:
        img = cv2.imread(fn)
        if img is None:
            return f"error: failed to read {fn}"

        (h, w) = img.shape[:2]
        if rheight == 0:
            r = rwidth / float(w)
            dim = (rwidth, int(h * r))
        else:
            dim = (rwidth, rheight)

        n_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(ofn, n_img)

        return f"resize {fn} to {ofn}"

    copy_dir_only(in_dir, out_dir)
    resize_p = partial(resize, rwidth=width, rheight=height)
    process_all(result, in_dir=in_dir, out_dir=out_dir, func=resize_p, out_ext="auto")


@cli.command(help="plot all wave files recursively")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=False, dir_okay=True)
)
@click.option("--ext", type=str, required=True)
@click.option("--width", type=int, required=True, help="unit must be inch")
@click.option(
    "--dpi", type=int, required=True, help="width and dpi decide the image pixel"
)
@click.option(
    "-t",
    "--ptype",
    type=click.Choice(["spectrogram", "scalogram"]),
    required=True,
)
def plot_all_wav(in_dir: str, out_dir: str, ext: str, width: int, dpi: int, ptype: str):
    """plot wave recursively"""
    if width <= 0:
        print("width must > 0")
        return

    result = [str(f) for f in pathlib.Path(in_dir).glob(f"**/*.{ext}")]
    print(f"result:\n{result}")
    if not result:
        print(f"failed to find *.{ext} in folder {in_dir}")
        return

    copy_dir_only(in_dir, out_dir)

    def plot_sp(fn: str, ofn) -> str:
        d, sr, _ = read_snd_file(fn, sr=None, mono=False, scale=True)
        plot_spectro(d, sr, ofn, dim=("inch", width, width), dpi=224)
        return f"plot {fn} spectrogram to {ofn}"

    def plot_sc(fn: str, ofn) -> str:
        d, sr, _ = read_snd_file(fn, sr=None, mono=False, scale=True)
        plot_scalo(d, sr, ofn, dim=("inch", width, width), dpi=224)
        return f"plot {fn} scalogram to {ofn}"

    match ptype:
        case "spectrogram":
            process_all(
                result,
                in_dir=in_dir,
                out_dir=out_dir,
                func=plot_sp,
                out_ext="png",
                processes=8,
                partition=8,
            )
        case "scalogram":
            process_all(
                result,
                in_dir=in_dir,
                out_dir=out_dir,
                func=plot_sc,
                out_ext="png",
                processes=8,
                partition=8,
            )
        case _:
            print(f"Unknown plot type: {ptype}")


@cli.command(help="Processing xeno-canto sound peaks recursively")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=False, dir_okay=True)
)
@click.option("--sr", type=int, required=True)
@click.option("--back", type=float, required=True, default=0.2)
@click.option("--forth", type=float, required=True, default=2.0)
def xeno_canto_peaks(in_dir: str, out_dir: str, sr: int, back: float, forth: float):
    result = [str(f) for f in pathlib.Path(in_dir).glob(f"**/*.wav")]
    print(f"result:\n{result}")
    if not result:
        print(f"failed to find *.wav in folder {in_dir}")
        return

    copy_dir_only(in_dir, out_dir)

    def process_peaks(fn: str, od: str, back: float, forth: float) -> str:
        snd_peaks(fn, sr=sr, back=back, forth=forth, out_dir=od)
        return f"find peaks in {fn} and output to {od}"

    peaks_f = partial(process_peaks, back=back, forth=forth)
    process_all(result, in_dir=in_dir, out_dir=out_dir, func=peaks_f, out_ext="dir")


@cli.command(help="Normalizing xeno-canto sound recursively")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=False, dir_okay=True)
)
@click.option("--sr", type=int, required=True)
@click.option("--tsr", type=int, required=True)
def xeno_canto_normalize(in_dir: str, out_dir: str, sr: int, tsr: int):
    meta_dsn_list = xeno_canto_meta(in_dir)
    if isinstance(meta_dsn_list, str):
        print(meta_dsn_list)
        return

    # print(meta_dsn_list)
    print_f = lambda t: print(f"{t[0].name}\t\t{t[1]}")
    print(f"meta file to dataset name:\n")
    pyf.seq(meta_dsn_list).for_each(print_f)

    o_dir_p = check_create_folder(out_dir)
    # create subfoloders
    pyf.seq(meta_dsn_list).for_each(lambda t: check_create_folder(str(o_dir_p / t[1])))

    read_snd = lambda p: read_snd_file(str(p), sr=sr, mono=True, scale=False)

    norm_p = partial(normalize, tsr=tsr)

    def write_wav(infp: pathlib.Path, od: pathlib.Path, y, sr: int) -> str:
        # print(f"in file path: {infp}")
        # print(f"out dir: {od}")

        out_fn = append_suffix(str(od / infp.name), "mono")
        if infp.suffix != ".wav":
            out_fn = change_ext(out_fn, ".wav")
        sf.write(out_fn, y, sr)
        return f"normalize {infp} and save to {od}"

    def normalize_batch(pl: List[pathlib.Path], out_d: pathlib.Path) -> List[str]:
        # print(f"in files: {pl}")
        # print(f"out dir: {out_d}")
        write_wav_f = partial(write_wav, od=out_d)

        y_sr_l = pyf.seq(pl).map(read_snd).map(lambda t: norm_p(t[0], t[1])).list()
        # print(f"y_sr_l:\n{y_sr_l[:2]}")

        ret = (
            pyf.seq(y_sr_l)
            .zip(pyf.seq(pl))
            .map(lambda t: write_wav_f(t[1], y=t[0][0], sr=t[0][1]))
        )
        return ret

    result = (
        pyf.seq(meta_dsn_list)
        .map(lambda t: (fetch_sound_files(t[0]), t[1]))
        .map(lambda t: normalize_batch(t[0], o_dir_p / t[1]))
        .list()
    )
    print(f"result:\n")
    pyf.seq(result).for_each(print)


def fetch_sound_files(p: pathlib.Path) -> List[pathlib.Path]:
    ext = [".wav", ".mp3"]
    return list(filter(lambda p: p.suffix in ext, p.parent.glob("**/*")))


@cli.command(help="Generate src sep dataset")
@click.option(
    "-i",
    "--in_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
    help="the raw dataset folder",
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
# @click.option("--sr", type=int, required=True)
# @click.option("--tsr", type=int, required=True)
@click.option(
    "--mux", type=int, required=True, default=1, help="multiplexer of mix sample count"
)
@click.option(
    "--n_src", type=int, required=True, default=2, help="source count, support 2 or 3"
)
@click.option(
    "--main_src",
    type=str,
    required=True,
    help="the main source folder name, must specify a folder contains train/val/test subfoloder when used with --train_ds option",
)
@click.option(
    "--gh_src",
    type=str,
    required=False,
    default="gh-21",
    help="the grasshopper source folder name, default is gh-21",
)
@click.option(
    "--fix_len",
    type=int,
    required=False,
    default=0,
    help="specify the fix length of soundtrack, such as 44100",
)
@click.option(
    "--second_src",
    type=click.Choice(["bird", "gh", "cricket"]),
    required=True,
    default="bird",
    help="specify the 2nd source, default is bird",
)
@click.option(
    "--third_src",
    type=click.Choice(["bird", "gh", "cricket"]),
    required=True,
    default="cricket",
    help="specify the 3rd source, default is cricket",
)
@click.option("--noise/--no-noise", default=False, help="if add noise source to mix")
@click.option(
    "--train_ds",
    nargs=3,
    type=int,
    required=False,
    help="sample count multiplexer of train, val, test",
)
def sep_data(
    in_dir: str,
    out_dir: str,
    mux: int,
    n_src: int,
    second_src: str,
    third_src: str,
    noise: bool,
    train_ds: tuple[int, int, int],
    fix_len: int,
    main_src: str,
    gh_src: str,
):
    print(f"input folder: {in_dir}")
    print(f"output folder: {out_dir}")
    print(f"n_src: {n_src}")
    print(f"noise: {noise}")
    print(f"train_ds: {train_ds}")

    if train_ds is not None:
        if not (pathlib.Path(in_dir) / main_src / "train").exists():
            # print(
            #     f"when --train_ds is specified, main_src folder must contain train/val/test subfolders"
            # )
            # return
            raise click.ClickException(
                f"when --train_ds is specified, main_src folder must contain train/val/test subfolders"
            )

    out_p = pathlib.Path(out_dir)
    if any(out_p.iterdir()):
        raise click.ClickException(
            f"output folder {out_p} has content already, please double check"
        )

    ret = create_sep_dataset(
        pathlib.Path(in_dir),
        out_p,
        main_src=main_src,
        gh_src=gh_src,
        n_src=n_src,
        mux=mux,
        second_src=second_src,
        third_src=third_src,
        addnoise=noise,
        train_ds=train_ds,
        fix_len=fix_len,
    )
    match ret:
        case Exception():
            raise click.ClickException(str(ret))
        case _:
            print(ret)


@cli.command(help="Generate src sep dataset")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "--fix_len",
    type=int,
    required=False,
    default=0,
    help="specify the fix length of soundtrack, such as 44100",
)
def check_len(in_dir: str, fix_len: int):
    import librosa

    if fix_len <= 0:
        print(f"fix_len can NOT be less or equal than 0")
        return

    p = pathlib.Path(in_dir)
    fl = list(p.glob("**/*.wav"))

    def snd_len(fn: str) -> int:
        y, _ = librosa.load(fn, sr=None, mono=True)
        return len(y)

    ret = (
        pyf.seq(fl)
        .map(lambda fn: (fn, snd_len(str(fn))))
        .filter(lambda t: t[1] != fix_len)
    )
    print(f"soundtracks whose len is NOT {fix_len}:\n{pyf.seq(ret)}")


if __name__ == "__main__":
    print(f"python version is {sys.version_info}")
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit("this program needs python 3.10 and above to run")

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    print(f"sys.path:\n{sys.path}")

    # l_fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    # logging.basicConfig(level=logging.ERROR, format=l_fmt)

    l_fmt = "[%(name)s %(levelname)s] %(asctime)s - %(message)s"
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(l_fmt))
    logger = logging.getLogger("dataset_tool")
    logger.addHandler(ch)
    logger.setLevel(logging.ERROR)

    cli()
