import sys
import os
from timeit import default_timer as timer
from datetime import timedelta
import psutil
import click
import logging
import librosa
from dataprocess.cwt.cwt2 import batch_extract, scaleo_extract, rd_file, cwt3, cwt2
from dataprocess.cwt.scalogram import plot_file
from dataprocess.util.data_process import replace_zeroes
from dataprocess.util.file import change_ext, check_create_folder
from dataprocess.sound.plot_wav import show_sources
from dataprocess.sound.nussl import AudioSignal

CMAP = "magma"
SR = 22050
TRAIN_DIR = "data/sound/cornell-birdcall/train_audio"


@click.group()
def cli():
    pass


@cli.command(help="extract the cwt images from given sound files")
@click.option(
    "-i",
    "--in_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "-o",
    "--out_dir",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
)
@click.option("--threshold", type=int, default=-60)
@click.option("--imgsize", type=int, default=256)
def extract(in_dir: str, out_dir: str, threshold, imgsize):
    check_create_folder(out_dir)

    flist = [
        "/btbwar/XC139608.mp3",
        "/btbwar/XC51863.mp3",
        "/btbwar/XC134502.mp3",
        "/btbwar/XC415596.mp3",
    ]
    plist = [f"{TRAIN_DIR}{f}" for f in flist]
    print(f"plist: {plist}")
    start = timer()
    core_cnt = psutil.cpu_count(logical=False)
    print(f"this computer has {core_cnt} physical cores")
    # 27s to finish
    batch_extract(plist, out_dir, batch=3, thres=threshold, imgsize=imgsize)

    # 41s to finish
    # for f in plist:
    #     scaleo_extract(f, outdir=out_dir)

    # 30s to finish
    # ray.init(num_cpus=4)
    # # fn_id = ray.put()
    # ray.get([scaleo_extract.remote(ray.put(f), outdir=out_dir) for f in plist])
    end = timer()
    print(f"======== total time: {timedelta(seconds=end-start)}")


@cli.command(help="extract the cwt images from a given sound file")
@click.option(
    "-i",
    "--in_fn",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option(
    "-o",
    "--out_dir",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
)
@click.option("--threshold", type=int, default=-60)
@click.option("--imgsize", type=int, default=256)
def single_extract(in_fn: str, out_dir: str, threshold, imgsize):
    check_create_folder(out_dir)
    scaleo_extract(in_fn, outdir=out_dir, thres=threshold, img_size=imgsize)


@cli.command(help="plot given sound file")
@click.option(
    "-i",
    "--in_fn",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option(
    "-t",
    "--ptype",
    type=click.Choice(["waveshow", "spectrogram", "scalogram", "fft", "all"]),
    required=True,
)
@click.option("--threshold", type=int, default=-60)
@click.option(
    "-o",
    "--out_fn",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
)
def plot(in_fn: str, ptype: str, threshold, out_fn: str):
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    fig = plot_file(in_fn, ptype, threshold)
    fig.savefig(out_fn)
    # plt.close()


@cli.command(help="plot two wav files")
@click.option(
    "-f",
    "--fore_fn",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-b",
    "--back_fn",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--out_fn",
    type=click.Path(exists=False, dir_okay=False),
)
def plot_sources(fore_fn: str, back_fn: str, out_fn: str):
    y_fore, sr_fore = librosa.load(fore_fn, sr=None, mono=True)
    y_back, sr_back = librosa.load(back_fn, sr=None, mono=True)

    if sr_fore != sr_back:
        print(f"{fore_fn} SR {sr_fore} is different from {back_fn} SR {sr_back}")
        return

    dur_fore = librosa.get_duration(y=y_fore, sr=sr_fore)
    dur_back = librosa.get_duration(y=y_back, sr=sr_back)

    if dur_fore != dur_back:
        print(
            f"{fore_fn} duration {dur_fore} is different from {back_fn} duration {dur_back}"
        )
        new_dur = dur_fore if dur_fore < dur_back else dur_back
        print(f"shorter duration {new_dur} will be used")

    y_fore, sr_fore = librosa.load(fore_fn, sr=None, mono=True, duration=new_dur)
    y_back, sr_back = librosa.load(back_fn, sr=None, mono=True, duration=new_dur)

    fore_s = AudioSignal(audio_data_array=y_fore, sample_rate=sr_fore)
    back_s = AudioSignal(audio_data_array=y_back, sample_rate=sr_back)
    fig = show_sources({"foreground": fore_s, "background": back_s})
    fig.savefig(out_fn)


@cli.command(help="test functions")
def test():
    pass


if __name__ == "__main__":
    print(f"python version is {sys.version_info}")
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit("this program needs python 3.10 and above to run")

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    # print(f'sys.path:\n{sys.path}')

    l_fmt = "[%(name)s %(levelname)s] %(asctime)s - %(message)s"
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(l_fmt))
    logger = logging.getLogger("dataprocess")
    logger.addHandler(ch)
    logger.setLevel(logging.ERROR)

    cli()
