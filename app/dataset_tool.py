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

    fp = f"{in_dir}/**/*.{ext}"
    print(f"search for {fp}")
    result = glob.glob(fp)
    if not result:
        print(f"failed to find *.{ext} in folder {in_dir}")
        return

    import cv2

    def copy_dir_only(idir: str, odir: str):
        import shutil

        # defining the function to ignore the files
        # if present in any folder
        def ignore_files(dir, files):
            return [f for f in files if os.path.isfile(os.path.join(dir, f))]

        # calling the shutil.copytree() method and
        # passing the src,dst,and ignore parameter
        shutil.copytree(idir, odir, ignore=ignore_files)

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
        if not nfn.startswith(outdir):
            return f"error[output_fn -> fn is not in outdir]: fn {fn} outdir {outdir}"

        return nfn

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
    functional.pseq(fn_pair_list, processes=8, partition_size=100).map(
        lambda t: resize(t[0], t[1], width, height)
    ).for_each(print)


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
