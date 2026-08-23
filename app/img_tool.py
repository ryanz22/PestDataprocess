import os
import pathlib
import numpy as np
import pprint
from typing import Tuple
import click
import cv2
import functional as pyf

from dataprocess.image.split import overlap_split as o_split, split_image
from dataprocess.util.file import append_suffix, change_ext, check_create_folder
from cli_bootstrap import bootstrap_cli


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


@cli.command(help="Convert gray-level image to black/white image")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option("--threshold", type=int, required=True, default=128)
@click.option(
    "--type",
    type=click.Choice(["rgb-gray", "gray-bw"]),
    default="rgb-gray",
    show_default=True,
)
@click.option("--out_dir", required=True, type=click.Path(exists=True, file_okay=False))
def rgb_gray_bw(in_fn: str, threshold: int, type: str, out_dir: str):
    import cv2
    from PIL import Image

    def gr_bw(fn: str, th: int, out: str):
        gimg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        _, bwimg = cv2.threshold(gimg, th, 255, cv2.THRESH_BINARY)

        # Write the byte data to a file
        fn2 = change_ext(fn, ".bmp")
        tmp_fn = os.path.join(out, os.path.basename(fn2))
        cv2.imwrite(tmp_fn, bwimg)

    def rgb_gr(fn: str, out: str):
        cimg = Image.open(fn)
        g_img = cimg.convert("L")

        # Write the byte data to a file
        fn2 = change_ext(fn, ".bmp")
        tmp_fn = os.path.join(out, os.path.basename(fn2))
        g_img.save(tmp_fn)

    def conv(fn: str, th: int, type: str, out: str):
        match type:
            case "rgb-gray":
                rgb_gr(fn, out)
            case "gray-bw":
                gr_bw(fn, threshold, out)

    if os.path.isdir(in_fn):
        for file in os.listdir(in_fn):
            if file.endswith(".bmp") or file.endswith(".png"):
                conv(os.path.join(in_fn, file), threshold, type, out_dir)
    else:
        conv(in_fn, threshold, type, out_dir)


@cli.command(help="Flip image")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "--direction",
    type=click.Choice(["v", "h", "vh"]),
    default="v",
    show_default=True,
)
@click.option("--out_dir", required=True, type=click.Path(exists=True, file_okay=False))
def flip(in_fn: str, direction: str, out_dir: str):
    from PIL import Image

    def f(fn: str, direct: str, out: str):
        img = Image.open(fn)
        match direct:
            case "v":
                f_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            case "h":
                f_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            case "vh":
                t_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                f_img = t_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Write the byte data to a file
        tmp_fn = os.path.join(out, os.path.basename(fn))
        f_img.save(tmp_fn)

    if os.path.isdir(in_fn):
        for file in os.listdir(in_fn):
            if file.endswith(".bmp"):
                f(os.path.join(in_fn, file), direction, out_dir)
    else:
        f(in_fn, direction, out_dir)


@cli.command(help="Convert image format, such as jpg to png")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "--from_format",
    type=click.Choice(["heic", "jpg", "bmp", "png"]),
    default="bmp",
    show_default=True,
)
@click.option(
    "--to_format",
    type=click.Choice(["jpg", "bmp", "png"]),
    default="bmp",
    show_default=True,
)
@click.option("--out_dir", required=True, type=click.Path(exists=True, file_okay=False))
def convert_format(in_fn: str, from_format: str, to_format: str, out_dir: str):
    from PIL import Image
    import pyheif

    def prepare_output(fn: str, out: str, to_f: str) -> tuple[str, str]:
        match to_f:
            case "bmp":
                fn2 = change_ext(fn, ".bmp")
                s_type = "BMP"
            case "png":
                fn2 = change_ext(fn, ".png")
                s_type = "PNG"
            case _:
                raise click.ClickException(f"Unsupport output format {to_f}")

        # Write the byte data to a file
        tmp_fn = os.path.join(out, os.path.basename(fn2))
        return tmp_fn, s_type

    def f(fn: str, to_f: str, out: str):
        print(f"Process {fn}")
        fp = pathlib.Path(fn)
        fext = fp.suffix

        match fext:
            case ".heic" | ".HEIC":
                heif_f = pyheif.read(fn)
                image = Image.frombytes(
                    heif_f.mode,
                    heif_f.size,
                    heif_f.data,
                    "raw",
                    heif_f.mode,
                    heif_f.stride,
                )
                tmp_fn, s_type = prepare_output(fn, out, to_f)
                image.save(tmp_fn, s_type)
            case ".jpg" | ".jpeg" | ".JPG":
                image = Image.open(fn)
                tmp_fn, s_type = prepare_output(fn, out, to_f)
                image.save(tmp_fn, s_type)
            case _:
                raise click.ClickException(f"Unsupport input format {fext}")

    in_path = pathlib.Path(in_fn)
    if in_path.is_dir():
        for file in in_path.glob(f"*.{from_format}"):
            print(f"#1: Process {file}")
            f(str(file), to_format, out_dir)
    else:
        f(in_fn, to_format, out_dir)


@cli.command(help="Augment image")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-t",
    "--transform",
    type=click.Choice(["rotate", "blur", "gamma", "bright", "contrast", "gaussnoise"]),
    default=["blur", "bright"],
    multiple=True,
    show_default=True,
    help="can be single or multiple values, -t blue -t gamma",
)
@click.option(
    "--type",
    type=click.Choice(["rgb", "gray"]),
    default="gray",
    show_default=True,
)
@click.option(
    "-m",
    "--mfactor",
    type=int,
    default=10,
    show_default=True,
    required=False,
    help="how many augemnted sound generated by given each single input",
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
def augment(in_fn: str, transform: str, type: str, mfactor: int, out_dir: str):
    import cv2

    # https://github.com/albumentations-team/albumentations#list-of-augmentations
    # https://albumentations-demo.herokuapp.com/
    # https://albumentations.ai/docs/examples/
    import albumentations as A

    print(f"Transform choices: {transform}")

    def f(fn: str, tr, type: str, m: int, out: str):
        match type:
            case "rgb":
                img = cv2.imread(fn, cv2.IMREAD_COLOR)
            case "gray":
                img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            case _:
                raise click.ClickException(f"Unsupport input format type: {type}")

        olist = pyf.seq(range(m)).map(lambda _: tr(image=img)["image"]).list()

        tmp_fn = pathlib.Path(fn).name
        out_p = pathlib.Path(out)
        out_fn_list = (
            pyf.seq(range(m))
            .map(lambda i: str(out_p / append_suffix(tmp_fn, f"{i}")))
            .list()
        )
        print(f"output fn:\n{out_fn_list}")
        pyf.seq(out_fn_list).zip(pyf.seq(olist)).for_each(
            lambda t: cv2.imwrite(t[0], t[1])
        )

    trans = A.Compose(
        [
            A.Rotate(limit=(-5, 5), p=0.3),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.05, 0.05), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            # A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.Blur(blur_limit=2, p=0.3),
        ]
    )

    in_path = pathlib.Path(in_fn)
    if in_path.is_dir():
        for file in in_path.glob(f"*.bmp"):
            print(f"#1: Process {file}")
            f(str(file), trans, type, mfactor, out_dir)
    else:
        f(in_fn, trans, type, mfactor, out_dir)


@cli.command(help="Center crop image")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "-w",
    "--width",
    type=int,
)
@click.option(
    "-h",
    "--height",
    type=int,
)
@click.option(
    "--type",
    type=click.Choice(["rgb", "gray"]),
    default="rgb",
    show_default=True,
)
@click.option(
    "-o", "--out_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
def center_crop(in_fn: str, width: int, height: int, type: str, out_dir: str):
    import cv2

    # https://github.com/albumentations-team/albumentations#list-of-augmentations
    # https://albumentations-demo.herokuapp.com/
    # https://albumentations.ai/docs/examples/
    import albumentations as A

    def f(fn: str, tr, type: str, out: str):
        match type:
            case "rgb":
                img = cv2.imread(fn, cv2.IMREAD_COLOR)
            case "gray":
                img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            case _:
                raise click.ClickException(f"Unsupport input format type: {type}")
        nimg = tr(image=img)["image"]
        tmp_fn = os.path.join(out, os.path.basename(fn))
        cv2.imwrite(tmp_fn, nimg)

    trans = A.Compose(
        [
            A.CenterCrop(width=width, height=height, p=1.0),
        ]
    )

    in_path = pathlib.Path(in_fn)
    if in_path.is_dir():
        for file in in_path.glob(f"*.bmp"):
            print(f"#1: Process {file}")
            f(str(file), trans, type, out_dir)
    else:
        f(in_fn, trans, type, out_dir)


if __name__ == "__main__":
    bootstrap_cli()
    cli()
