import os
import sys
import glob
import pathlib
import numpy as np
import pprint
from typing import Tuple
import click
import logging
import cv2

from dataprocess.image.split import overlap_split as o_split
from dataprocess.util.file import append_suffix, change_ext, check_create_folder


@click.group()
def cli():
    pass


@cli.command(help="Split large image into small ones with overlap")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--width", type=int, required=True)
@click.option("--height", type=int, required=True)
@click.option("--overlap", type=float, required=True, default=0.2)
@click.option(
    "--out_dir", required=True, type=click.Path(exists=False, file_okay=False)
)
def overlap_split(in_fn: str, width: int, height: int, overlap: float, out_dir: str):
    """Split large image into small ones with overlap"""

    img = cv2.imread(in_fn)
    imgs = o_split(
        img,
        split_width=width,
        split_height=height,
        overlap=overlap,
    )

    out_path = check_create_folder(out_dir)

    frmt: str = ".jpg"

    tmp_fn = pathlib.Path(in_fn).name
    for i, c in enumerate(imgs):
        out_fn = append_suffix(tmp_fn, str(i))
        if pathlib.Path(tmp_fn).suffix != frmt:
            out_fn = change_ext(out_fn, frmt)

        cv2.imwrite(str(out_path / out_fn), c)


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
    logger = logging.getLogger("dataprocess")
    logger.addHandler(ch)
    logger.setLevel(logging.ERROR)

    cli()
