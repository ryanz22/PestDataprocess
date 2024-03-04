import sys
import os
import typer
from typing_extensions import Annotated
from pathlib import Path
from enum import Enum
import shutil

from dataprocess.util.file import append_suffix, change_ext, check_create_folder


class TorchTask(str, Enum):
    si_snr = "si-snr"
    wav_conv = "wav-conv"


app = typer.Typer()


@app.command(help="pick wav files from dataset to the given folder")
def rename(
    in_dir: Annotated[
        Path,
        typer.Option(
            # default=...,
            exists=True,
            file_okay=True,
            dir_okay=True,
            help="input file or dir",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            # default=...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="output dir",
        ),
    ],
):
    print(f"input dir is: {in_dir}")
    print(f"output dir is: {out_dir}")

    def f(fn: Path, out: Path):
        nameonly = fn.stem
        fext = fn.suffix
        new_name = nameonly[:3] + fext

        # Write the byte data to a file
        # fn2 = Path(change_ext(str(fn), ".bmp"))
        tmp_fn = out / new_name
        shutil.copy2(fn, tmp_fn)

    if in_dir.is_dir():
        for file in in_dir.glob("*.*"):
            f(in_dir / file, out_dir)
    else:
        f(in_dir, out_dir)


@app.command("test")
def test():
    pass


if __name__ == "__main__":
    print(f"python version is {sys.version_info}")
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit("this program needs python 3.10 and above to run")

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    print(f"sys.path:\n{sys.path}")

    app()
