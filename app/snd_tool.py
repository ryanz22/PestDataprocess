import os
import sys
import glob
import pathlib
import numpy as np
import pprint
from typing import Tuple
import librosa
import click
import soundfile as sf
import logging
import functional as pyf

from dataprocess.util.file import append_suffix, change_ext, check_create_folder
from dataprocess.sound.audio_augment import augment_single
from dataprocess.sound.preprocess import (
    denoise as deno,
    to_mono,
    sound_file_info,
    resample as resam,
    is_stereo_sound,
    retrieve_clips,
)
from dataprocess.sound.filter_util import load_audio_file, freq_filter
import warnings

warnings.filterwarnings("ignore")  # get rid of librosa warnings


@click.group()
def cli():
    pass


@cli.command(help="denoise input sound file and output to the same location")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
def denoise(in_fn: str):
    data, sr = librosa.load(in_fn, sr=None, mono=True)
    od, sr = deno(data, sr)
    out_fn = append_suffix(in_fn, "denoised")
    if pathlib.Path(in_fn).suffix != ".wav":
        out_fn = change_ext(out_fn, ".wav")
    print(f"output file name: {out_fn}")
    sf.write(out_fn, od, sr)


@cli.command(help="mono input sound file and output to the same location")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
def mono(in_fn: str):
    if is_stereo_sound(in_fn):
        data, sr = librosa.load(in_fn, sr=None, mono=False)
        od, sr = to_mono(data, sr)
        out_fn = append_suffix(in_fn, "mono")
        if pathlib.Path(in_fn).suffix != ".wav":
            out_fn = change_ext(out_fn, ".wav")
        sf.write(out_fn, od, sr)
    else:
        print("this is a mono sound track")


@cli.command(help="normalize input sound file and output to the same location")
@click.option(
    "-f",
    "--in_fn",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="convert any types of sound to mono 22050 wav",
)
def normalize(in_fn: str):
    data, sr = librosa.load(in_fn, sr=22050, mono=True)
    out_fn = append_suffix(in_fn, "mono_22050")
    if pathlib.Path(in_fn).suffix != ".wav":
        out_fn = change_ext(out_fn, ".wav")
    sf.write(out_fn, data, sr)


@cli.command(help="resample input sound file and output to the same location")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-t", "--tsr", default=22050)
def resample(in_fn: str, tsr: int):
    data, sr = librosa.load(in_fn, sr=None, mono=False)
    od, nsr = resam(data, sr, tsr=tsr)
    out_fn = append_suffix(in_fn, str(nsr))
    if pathlib.Path(in_fn).suffix != ".wav":
        out_fn = change_ext(out_fn, ".wav")
    sf.write(out_fn, od, nsr)


@cli.command(help="show info of input sound file")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
def info(in_fn: str):
    t = sound_file_info(in_fn)
    pprint.pprint(t, indent=2)


@cli.command(help="filter input sound file and output to the same location")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(["lowpass", "highpass", "bandpass", "bandstop"]),
    required=True,
)
@click.option("--fc", type=int)
@click.option("--fr", nargs=2, type=int)
@click.option("--sr", default=22050)
def filter(in_fn: str, type: str, fc: None | int, fr: None | Tuple[int, int], sr: int):
    _, fsr = librosa.load(in_fn, sr=None, mono=False)
    if fsr != sr:
        raise ValueError(
            f"input sound has {fsr} sample rate which is different than {sr}"
        )

    match type:
        case None:
            raise ValueError("miss filter")
        case "lowpass" | "highpass":
            if fc is None:
                raise ValueError(f"filter {type} miss fc")
            else:
                params = {"freqs": {"f_c": fc}}
        case "bandpass" | "bandstop":
            if fr is None:
                raise ValueError(f"filter {type} miss fr")
            elif fr[0] >= fr[1]:
                raise ValueError(f"{fr[0]} >= {fr[1]}")
            else:
                params = {"freqs": {"f_l": fr[0], "f_h": fr[1]}}
        case _:
            raise ValueError(f"unsupported filter: {type}")

    in_frames = load_audio_file(in_fn)

    out_fn = append_suffix(in_fn, "filter")
    if pathlib.Path(in_fn).suffix != ".wav":
        out_fn = change_ext(out_fn, ".wav")

    out_frames = freq_filter(
        in_frames=in_frames,
        filter_type=type,
        params=params,
        Fs=sr,
        do_plot=True,
        plot_dir=".",
    )

    print(f"output: {out_fn}")
    print(f"outframes:\n{out_frames[:20]}")
    from scipy.io.wavfile import write

    tmp = np.array(out_frames, dtype=np.int16)
    write(out_fn, sr, tmp)
    # sf.write(out_fn, out_frames, sr)


@cli.command(help="slice input sound file and output to the same location")
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-l", "--length", type=float, required=True)
@click.option("-o", "--offset", type=float, required=False, default=0.0)
def single_slice(in_fn: str, offset: float, length: float):
    """_summary_

    Args:
        in_fn (str): _description_
        offset (float): _description_
        length (float): _description_
    """
    print(f"slice {in_fn} offset {offset} length {length}")

    y, sr = librosa.load(in_fn, sr=None, mono=False, offset=offset, duration=length)
    out_fn = append_suffix(in_fn, f"{length:.1f}_sliced")
    if pathlib.Path(in_fn).suffix != ".wav":
        out_fn = change_ext(out_fn, ".wav")
    sf.write(out_fn, y, sr)


@cli.command(
    help="hop slice input sound file and output multiple sound tracks \
             to the given output folder"
)
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--slice_len", type=int, required=True, help="the length of slice in second"
)
@click.option("--hop", type=float, required=True, help="hop forward in second")
@click.option(
    "--out_dir", required=True, type=click.Path(exists=False, file_okay=False)
)
def hop_slice(in_fn: str, slice_len: int, hop: float, out_dir: str):
    print(f"slice {in_fn} slice_len {slice_len} hop {hop} out_dir {out_dir}")
    _, sr = librosa.load(in_fn, sr=None, mono=False)
    slices_gen = librosa.stream(
        in_fn,
        block_length=1,
        frame_length=sr * slice_len,
        hop_length=int(sr * hop),
        mono=True,
        fill_value=0,
    )

    out_path = check_create_folder(out_dir)

    tmp_fn = pathlib.Path(in_fn).name
    slices_l = [s for s in slices_gen]
    for i, s in enumerate(slices_l):
        out_fn = append_suffix(tmp_fn, str(i))
        if pathlib.Path(tmp_fn).suffix != ".wav":
            out_fn = change_ext(out_fn, ".wav")

        sf.write(out_path / out_fn, s, sr)


@cli.command(
    help="find the sound peaks in given sound file and output multiple sound tracks to given output folder"
)
@click.option(
    "-f", "--in_fn", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--sr", type=int, required=False)
@click.option("--back", type=float, required=True, default=0.5)
@click.option("--forth", type=float, required=True, default=2.0)
@click.option(
    "--out_dir", required=True, type=click.Path(exists=False, file_okay=False)
)
def peaks(in_fn: str, sr: int, back: float, forth: float, out_dir: str):
    print(f"sr: {sr}")
    if sr is None:
        y, sr = librosa.load(in_fn, sr=None, mono=True)
    else:
        y, _ = librosa.load(in_fn, sr=sr, mono=True)

    print(f"sr: {sr}")
    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        # hop_length=1024,
        # aggregate=np.median
    )
    # print(f'number count onset_env len {len(onset_env_nc)}:\n{onset_env_nc}')

    # onset_nc = librosa.onset.onset_detect(y=slices[0], sr=sr2, units='time')
    # print(f'number count onset detect len {len(onset_nc)}:\n{onset_nc}')

    peaks_l = librosa.util.peak_pick(
        onset, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.8, wait=10
    )
    clips = retrieve_clips(fn=in_fn, sr=sr, peaks=peaks_l, back=back, forth=forth)

    out_path = check_create_folder(out_dir)

    tmp_fn = pathlib.Path(in_fn).name
    for i, c in enumerate(clips):
        out_fn = append_suffix(tmp_fn, str(i))
        if pathlib.Path(tmp_fn).suffix != ".wav":
            out_fn = change_ext(out_fn, ".wav")

        sf.write(out_path / out_fn, c, sr)


@cli.command(help="stretch the sound file or folder to output folder")
@click.option(
    "-f",
    "--in_fn",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-d",
    "--in_dir",
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--rate", type=float, required=True)
@click.option(
    "--out_dir", required=True, type=click.Path(exists=False, file_okay=False)
)
def stretch(in_fn: str, in_dir: str, rate: float, out_dir: str):
    if rate == 1.0:
        print(f"rate is {rate}, no need to stretch")
        return

    s_type = "fast" if rate > 1.0 else "slow"

    if in_fn is not None:
        print("process in_fn")
        y, sr = librosa.load(in_fn, sr=None, mono=True)
        ny = librosa.effects.time_stretch(y, rate=rate)

        out_path = check_create_folder(out_dir)

        tmp_fn = pathlib.Path(in_fn).name
        out_fn = append_suffix(tmp_fn, f"{s_type}_{rate}")
        if pathlib.Path(tmp_fn).suffix != ".wav":
            out_fn = change_ext(out_fn, ".wav")

        sf.write(out_path / out_fn, ny, sr)
    elif in_dir is not None:
        print("process in_dir")
        out_path = check_create_folder(out_dir)

        for wav_file in pathlib.Path(in_dir).glob("*.wav"):
            print(f"file {wav_file}, type {type(wav_file)}")
            y, sr = librosa.load(wav_file, sr=None, mono=True)
            ny = librosa.effects.time_stretch(y, rate=rate)

            tmp_fn = wav_file.name
            out_fn = append_suffix(tmp_fn, f"{s_type}_{rate}")

            sf.write(out_path / out_fn, ny, sr)
    else:
        print("must specify either in_fn or in_dir")


@cli.command(help="Convert to wav file")
@click.option(
    "-f",
    "--in_fn",
    type=click.Path(exists=True, dir_okay=False),
)
def to_wav(in_fn: str):
    y, sr = librosa.load(in_fn, sr=None, mono=True)
    out_fn = change_ext(in_fn, ".wav")
    sf.write(out_fn, y, sr)


@cli.command(help="Mix two wav files, give two wav file names and output file name")
@click.option(
    "-i",
    "--in_fn",
    nargs=2,
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "-o", "--out_fn", type=click.Path(exists=False, dir_okay=False), required=True
)
def mix(in_fn: Tuple[str, str], out_fn: str):
    fn_1, fn_2 = in_fn
    y_1, sr_1 = librosa.load(fn_1, sr=None, mono=True)
    y_2, sr_2 = librosa.load(fn_2, sr=None, mono=True)

    if sr_1 != sr_2:
        print(f"{fn_1} SR {sr_1} is different from {fn_2} SR {sr_2}")
        return

    dur_1 = librosa.get_duration(y=y_1, sr=sr_1)
    dur_2 = librosa.get_duration(y=y_2, sr=sr_2)

    if dur_1 != dur_2:
        print(f"{fn_1} duration {dur_1} is different from {fn_2} duration {dur_2}")
        new_dur = dur_1 if dur_1 < dur_2 else dur_2
        print(f"shorter duration {new_dur} will be used")

    y_1, sr_1 = librosa.load(fn_1, sr=None, mono=True, duration=new_dur)
    y_2, sr_2 = librosa.load(fn_2, sr=None, mono=True, duration=new_dur)
    y_mix = y_1 + y_2
    sf.write(out_fn, y_mix, sr_1)


@cli.command(help="Augment input sound file or folder")
@click.option(
    "-i",
    "--in_fn",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    required=True,
)
@click.option(
    "-b",
    "--bg",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.option(
    "-c",
    "--count",
    type=int,
    default=10,
    show_default=True,
    required=False,
    help="how many augemnted sound generated by given each single input",
)
@click.option(
    "-o",
    "--out",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
def augment(in_fn: str, bg: str, count: int, out: str):
    p = pathlib.Path(in_fn)

    if p.is_dir():
        print(f"Augment input folder: {in_fn}")
        wav_list = [f for f in p.glob("*.wav")]
        pyf.seq(wav_list).for_each(print)
        pyf.seq(wav_list).for_each(
            lambda f: augment_single(f, count=count, bg=bg, out=out)
        )
    else:
        print(f"Augment input file: {in_fn}")
        augment_single(in_fn, count=count, bg=bg, out=out)


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
