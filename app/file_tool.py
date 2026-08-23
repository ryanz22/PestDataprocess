import typer
from typing_extensions import Annotated
from pathlib import Path
from enum import Enum
import shutil

from cli_bootstrap import bootstrap_cli


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
    bootstrap_cli()
    app()
