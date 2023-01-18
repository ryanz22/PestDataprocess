import os
import sys
import glob
import pathlib
import numpy as np
import pprint
from typing import Tuple, List
import click
import logging
import functional
from functools import partial

from dataprocess.util.file import copy_dir_only
from dataprocess.util.data_process import process_all
from dataprocess.cwt.scalogram import (
    plot_spectro,
    plot_scalo,
)
from dataprocess.util.data_process import read_snd_file


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
    print(f'result:\n{result}')
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
    process_all(result, in_dir=in_dir, out_dir=out_dir, func=resize_p)


@cli.command(help="plot all wave files recursively")
@click.option(
    "-i", "--in_dir", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=False, dir_okay=True)
)
@click.option("--ext", type=str, required=True)
@click.option("--width", type=int, required=True, help="unit must be inch")
@click.option("--dpi", type=int, required=True, help="width and dpi decide the image pixel")
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
    print(f'result:\n{result}')
    if not result:
        print(f"failed to find *.{ext} in folder {in_dir}")
        return

    copy_dir_only(in_dir, out_dir)

    def plot_sp(fn: str, ofn) -> str:
        d, sr, _ = read_snd_file(fn, sr=None, mono=False, scale=True)
        plot_spectro(d, sr, ofn, dim=("inch", width, width), dpi=224)
        return f"resize {fn} to {ofn}"

    def plot_sc(fn: str, ofn) -> str:
        d, sr, _ = read_snd_file(fn, sr=None, mono=False, scale=True)
        plot_scalo(d, sr, ofn, dim=("inch", width, width), dpi=224)
        return f"resize {fn} to {ofn}"

    match ptype:
        case "spectrogram":
            process_all(result, in_dir=in_dir, out_dir=out_dir, func=plot_sp, out_ext="png", processes=4, partition=4)
        case "scalogram":
            process_all(result, in_dir=in_dir, out_dir=out_dir, func=plot_sc, out_ext="png", processes=4, partition=4)
        case _:
            print(f'Unknown plot type: {ptype}')


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
