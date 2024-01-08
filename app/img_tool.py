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

from dataprocess.image.split import overlap_split as o_split, split_image
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


@cli.command(help="Split large image into small ones")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--row", type=int, required=True)
@click.option("--col", type=int, required=True)
@click.option("--square", type=bool, required=False, default=False)
@click.option(
    "--out_dir", required=True, type=click.Path(exists=False, file_okay=False)
)
def split(in_fn: str, row: int, col: int, square: bool, out_dir):
    if os.path.isdir(in_fn):
        for file in os.listdir(in_fn):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                split_image(
                    os.path.join(in_fn, file), row, col, square, False, False, out_dir
                )
    else:
        split_image(in_fn, row, col, square, False, False, out_dir)


@cli.command(help="Resize images")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--width", type=int, required=True)
@click.option("--height", type=int, required=True)
@click.option("--out_dir", required=True, type=click.Path(exists=True, file_okay=False))
def resize(in_fn: str, width: int, height: int, out_dir: str):
    from PIL import Image

    def resize_img(fn: str, w: int, h: int, out: str):
        img = Image.open(fn)
        new_img = img.resize((w, h))
        new_img.save(os.path.join(out, os.path.basename(fn)))

    if os.path.isdir(in_fn):
        for file in os.listdir(in_fn):
            if (
                file.endswith(".jpg")
                or file.endswith(".jpeg")
                or file.endswith(".png")
                or file.endswith(".bmp")
            ):
                resize_img(os.path.join(in_fn, file), width, height, out_dir)
    else:
        resize_img(in_fn, width, height, out_dir)


@cli.command(help="Dump raw bytes of image files")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--out_dir", required=True, type=click.Path(exists=True, file_okay=False))
def dump_raw(in_fn: str, out_dir: str):
    from PIL import Image
    import numpy as np

    def dump(fn: str, out: str):
        # Open the BMP file
        with Image.open(fn) as img:
            # Ensure the image is in grayscale mode
            if img.mode != "L":
                img = img.convert("L")

            # Convert image data to a numpy array
            img_data = np.array(img)

            # Get byte data from the numpy array
            byte_data = img_data.tobytes()

            # Write the byte data to a file
            fn2 = change_ext(fn, ".bin")
            with open(os.path.join(out, os.path.basename(fn2)), "wb") as file:
                file.write(byte_data)

    if os.path.isdir(in_fn):
        for file in os.listdir(in_fn):
            if (
                file.endswith(".jpg")
                or file.endswith(".jpeg")
                or file.endswith(".png")
                or file.endswith(".bmp")
            ):
                dump(os.path.join(in_fn, file), out_dir)
    else:
        dump(in_fn, out_dir)


@cli.command(help="Load raw bytes and convert to image files")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--width", type=int, required=True)
@click.option("--height", type=int, required=True)
@click.option("--out_dir", required=True, type=click.Path(exists=True, file_okay=False))
def convert_raw(in_fn: str, width: int, height: int, out_dir: str):
    from PIL import Image
    import io

    def raw_bmp(fn: str, w: int, h: int, out: str):
        # Open the BMP file
        with open(fn, "rb") as file:
            raw_bytes = file.read()
            # bytes_io = io.BytesIO(raw_bytes)
            img = Image.new("L", (w, h))
            img.frombytes(raw_bytes)

            # Write the byte data to a file
            fn2 = change_ext(fn, ".bmp")
            with open(os.path.join(out, os.path.basename(fn2)), "wb") as outf:
                img.save(outf)

    if os.path.isdir(in_fn):
        for file in os.listdir(in_fn):
            if file.endswith(".bin"):
                raw_bmp(os.path.join(in_fn, file), width, height, out_dir)
    else:
        raw_bmp(in_fn, width, height, out_dir)


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
