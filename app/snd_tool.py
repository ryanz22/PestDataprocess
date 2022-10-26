import os
import sys
import pathlib
import numpy as np
import pprint
from typing import Tuple
import librosa
import click
import soundfile as sf
import logging

from dataprocess.util.file import append_suffix, change_ext
from dataprocess.sound.preprocess import (denoise as deno, to_mono, sound_file_info, 
    resample as resam, is_stereo_sound)
from dataprocess.sound.filter_util import (load_audio_file, freq_filter)
import warnings
warnings.filterwarnings('ignore') # get rid of librosa warnings


@click.group()
def cli():
    pass


@cli.command(help='denoise input sound file and output to the same location')
@click.option('-f', '--in_fn', required=True, type=click.Path(exists=True, dir_okay=False))
def denoise(in_fn: str):
    data, sr = librosa.load(in_fn, sr=None, mono=True)
    od, sr = deno(data, sr)
    out_fn = append_suffix(in_fn, 'denoised')
    if pathlib.Path(in_fn).suffix != '.wav':
        out_fn = change_ext(out_fn, '.wav')
    print(f'output file name: {out_fn}')
    sf.write(out_fn, od, sr)


@cli.command(help='mono input sound file and output to the same location')
@click.option('-f', '--in_fn', required=True, type=click.Path(exists=True, dir_okay=False))
def mono(in_fn: str):
    if is_stereo_sound(in_fn):
        data, sr = librosa.load(in_fn, sr=None, mono=False)
        od, sr = to_mono(data, sr)
        out_fn = append_suffix(in_fn, 'mono')
        if pathlib.Path(in_fn).suffix != '.wav':
            out_fn = change_ext(out_fn, '.wav')
        sf.write(out_fn, od, sr)
    else:
        print('this is a mono sound track')


@cli.command(help='normalize input sound file and output to the same location')
@click.option('-f', '--in_fn', required=True, 
              type=click.Path(exists=True, dir_okay=False),
              help='convert any types of sound to mono 22050 wav')
def normalize(in_fn: str):
    data, sr = librosa.load(in_fn, sr=22050, mono=True)
    out_fn = append_suffix(in_fn, 'mono_22050')
    if pathlib.Path(in_fn).suffix != '.wav':
        out_fn = change_ext(out_fn, '.wav')
    sf.write(out_fn, data, sr)


@cli.command(help='resample input sound file and output to the same location')
@click.option('-f', '--in_fn', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--tsr', default=22050)
def resample(in_fn: str, tsr: int):
    data, sr = librosa.load(in_fn, sr=None, mono=False)
    od, nsr = resam(data, sr, tsr=tsr)
    out_fn = append_suffix(in_fn, str(nsr))
    if pathlib.Path(in_fn).suffix != '.wav':
        out_fn = change_ext(out_fn, '.wav')
    sf.write(out_fn, od, nsr)


@cli.command(help='show info of input sound file')
@click.option('-f', '--in_fn', required=True, type=click.Path(exists=True, dir_okay=False))
def info(in_fn: str):
    t = sound_file_info(in_fn)
    pprint.pprint(t, indent=2)


@cli.command(help='filter input sound file and output to the same location')
@click.option('-f', '--in_fn', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--type', type=click.Choice(['lowpass', 'highpass', 'bandpass', 'bandstop']), required=True)
@click.option('--fc', type=int)
@click.option('--fr', nargs=2, type=int)
@click.option('--sr', default=22050)
def filter(in_fn: str, type: str, fc: None | int, fr: None | Tuple[int, int], sr: int):
    _, fsr = librosa.load(in_fn, sr=None, mono=False)
    if fsr != sr:
        raise ValueError(f'input sound has {fsr} sample rate which is different than {sr}')

    match type:
        case None:
            raise ValueError('miss filter')
        case 'lowpass' | 'highpass':
            if fc is None:
                raise ValueError(f'filter {type} miss fc')
            else:
                params = {"freqs": {"f_c": fc}}
        case 'bandpass' | 'bandstop':
            if fr is None:
                raise ValueError(f'filter {type} miss fr')
            elif fr[0] >= fr[1]:
                raise ValueError(f'{fr[0]} >= {fr[1]}')
            else:
                params = {"freqs": {"f_l": fr[0], "f_h": fr[1]}}
        case _:
            raise ValueError(f'unsupported filter: {type}')

    in_frames = load_audio_file(in_fn)

    out_fn = append_suffix(in_fn, 'filter')
    if pathlib.Path(in_fn).suffix != '.wav':
        out_fn = change_ext(out_fn, '.wav')

    out_frames = freq_filter(
        in_frames=in_frames,
        filter_type=type,
        params=params,
        Fs=sr,
        do_plot=True,
        plot_dir='.'
    )

    print(f'output: {out_fn}')
    print(f'outframes:\n{out_frames[:20]}')
    from scipy.io.wavfile import write
    tmp = np.array(out_frames, dtype=np.int16)
    write(out_fn, sr, tmp)
    # sf.write(out_fn, out_frames, sr)


@cli.command(help='slice input sound file and output to the same location')
@click.option('-f', '--in_fn', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-l', '--length', type=float, required=True)
@click.option('-o', '--offset', type=float, required=False, default=0.0)
def slice(in_fn: str, offset: float, length: float):
    y, sr = librosa.load(in_fn, sr=None, mono=False, offset=offset, duration=length)
    out_fn = append_suffix(in_fn, 'sliced')
    if pathlib.Path(in_fn).suffix != '.wav':
        out_fn = change_ext(out_fn, '.wav')
    sf.write(out_fn, y, sr)


if __name__ == '__main__':
    print(f'python version is {sys.version_info}')
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit('this program needs python 3.10 and above to run')

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    print(f'sys.path:\n{sys.path}')

    # l_fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    # logging.basicConfig(level=logging.ERROR, format=l_fmt)

    l_fmt = '[%(name)s %(levelname)s] %(asctime)s - %(message)s'
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(l_fmt))
    logger = logging.getLogger('dataprocess')
    logger.addHandler(ch)
    logger.setLevel(logging.ERROR)

    cli()
