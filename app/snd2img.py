import sys
import os
from timeit import default_timer as timer
from datetime import timedelta
import psutil
import click
import logging
import librosa
from typing import Tuple, List
import functional as pyfun

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
    "--fn_list",
    type=click.Tuple([str, str]),
    multiple=True,
    required=True,
    help='-f tag file_path'
)
@click.option(
    "--sr",
    type=int,
    required=True
)
@click.option(
    "--duration",
    type=float,
    required=True
)
@click.option(
    "-o",
    "--out_fn",
    type=click.Path(exists=False, dir_okay=False),
    required=True
)
def plot_sources(fn_list: Tuple[Tuple[str, str]], sr: int, duration: float, out_fn: str):
    print(type(fn_list))
    print(fn_list)

    for tag, fn in fn_list:
        print(f'{tag:20s}{fn}')

    all_fn = pyfun.seq(fn_list).map(lambda t: t[1]).list()

    missing_fn = pyfun.seq(all_fn)\
        .filter_not(lambda f: os.path.exists(fn))\
        .list()

    if missing_fn:
        pyfun.seq(missing_fn).for_each(lambda fn: print(f'{fn} doesn\'t exist\n'))
        return

    def get_data(fn: str, duration: float = None):
        y_t, sr_t = librosa.load(fn, sr=None, mono=True, duration=duration)
        dur_t = librosa.get_duration(y=y_t, sr=sr_t)
        return fn, y_t, sr_t, dur_t

    all_data = pyfun.seq(all_fn).map(get_data).list()

    def check_wav(fn: str, sr_i: int, duration_i: float) -> str | None:
        if sr_i != sr and duration_i < duration:
            msg = f'{fn} sample rate [{sr_i}] is NOT {sr} and duration [{duration_i}] is shorter than {duration}\n'
            return msg

        if sr_i != sr:
            msg = f'{fn} sample rate [{sr_i}] is NOT {sr}\n'
            return msg

        if duration_i < duration:
            msg = f'{fn} duration [{duration_i}] is shorter than {duration}\n'
            return msg
        else:
            return None
        
    # check sample rate and duration
    wrong_list = pyfun.seq(all_data)\
        .map(lambda t: check_wav(t[0], t[2], t[3]))\
        .filter_not(lambda r: r is None)\
        .list()

    if wrong_list:
        pyfun.seq(wrong_list).for_each(print)
        return

    # force specified duration
    new_all_data = pyfun.seq(all_fn)\
        .map(lambda f: get_data(f, duration))\
        .map(lambda t: AudioSignal(audio_data_array=t[1], sample_rate=t[2]))\
        .list()

    # new_all_data = []
    # for fn in all_fn:
    #     _, y_t, sr_t, _ = get_data(fn, duration)
    #     signal = AudioSignal(audio_data_array=y_t, sample_rate=sr_t)
    #     new_all_data.append(signal)

    all_tag = pyfun.seq(fn_list).map(lambda t: t[0]).list()
    tmp = pyfun.seq(all_tag).zip(pyfun.seq(all_fn)).dict()
    print(tmp)

    meta = pyfun.seq(all_tag).zip(pyfun.seq(new_all_data)).dict()
    print(meta)

    # (_, y_t, sr_t, _) = get_data(all_fn[0], duration)
    # gh = AudioSignal(audio_data_array=y_t, sample_rate=sr_t)
    # (_, y_t, sr_t, _) = get_data(all_fn[1], duration)
    # drone = AudioSignal(audio_data_array=y_t, sample_rate=sr_t)
    # fig = show_sources({"foreground": gh, "background": drone})

    #fig = show_sources({"foreground": meta['drone'], "background": meta['drone']})
    fig = show_sources(meta)
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
