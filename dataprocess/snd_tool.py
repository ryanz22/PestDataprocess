import os
import sys
import argparse
import pathlib
import pprint
import librosa
import soundfile as sf
from scipy.io import wavfile
from util import append_suffix, change_ext
from sound import (denoise, mono, sound_file_info, resample, load_audio_file,
                test_static, is_stereo_sound)


def filter(args):
    match args.type:
        case None:
            raise ValueError('miss filter')
        case 'lowpass' | 'highpass':
            if args.fc is None:
                raise ValueError(f'filter {args.type} miss fc')
            else:
                params = {"freqs": {"f_c": args.fc}}
        case 'bandpass' | 'bandstop':
            if args.fl is None:
                raise ValueError(f'filter {args.type} miss fl')
            if args.fh is None:
                raise ValueError(f'filter {args.type} miss fh')
            params = {"freqs": {"f_l": args.fl, "f_h": args.fh}}
        case _:
            raise ValueError(f'unsupported filter: {args.type}')

    in_fn = args.file
    in_frames = load_audio_file(in_fn)

    out_fn = append_suffix(in_fn, 'filter')
    if pathlib.Path(in_fn).suffix != '.wav':
        out_fn = change_ext(out_fn, '.wav')

    test_static(
        in_frames=in_frames,
        filter_type=args.type,
        params=params,
        Fs=args.sample_rate,
        output=out_fn,
        plot_dir='.'
    )


def main(args):
    match args.command:
        case 'denoise':
            in_fn = args.input_file
            # sr, data = wavfile.read(in_fn)
            data, sr = librosa.load(in_fn, sr=None, mono=False)
            od, sr = denoise(data, sr)
            out_fn = append_suffix(in_fn, 'denoised')
            print(f'output file name: {out_fn}')
            # wavfile.write(out_fn, sr, od)
            sf.write(out_fn, od, sr)
        case 'mono':
            in_fn = args.input_file
            if is_stereo_sound(in_fn):
                data, sr = librosa.load(in_fn, sr=None, mono=False)
                od, sr = mono(data, sr)
                out_fn = append_suffix(in_fn, 'mono')
                if pathlib.Path(in_fn).suffix != '.wav':
                    out_fn = change_ext(out_fn, '.wav')
                sf.write(out_fn, od, sr)
            else:
                print('this is a mono sound track')
        case 'resample':
            in_fn = args.input_file
            tsr = args.sr
            data, sr = librosa.load(in_fn, sr=None, mono=False)
            od, nsr = resample(data, sr, tsr=tsr)
            out_fn = append_suffix(in_fn, str(nsr))
            if pathlib.Path(in_fn).suffix != '.wav':
                out_fn = change_ext(out_fn, '.wav')
            sf.write(out_fn, od, nsr)
        case 'info':
            t = sound_file_info(args.file_name)
            pprint.pprint(t, indent=2)
        case 'filter':
            filter(args)
        case _:
            print(f'unknown command [{args.command}]')

if __name__ == '__main__':
    print(f'python version is {sys.version_info}')
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit('this program needs python 3.10 and above to run')

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    aparser = argparse.ArgumentParser(description='CWT Utility')
    sub = aparser.add_subparsers(dest='command', help='command to run', required=True)

    denoise_arg = sub.add_parser('denoise', help='denoise a sound file')
    denoise_arg.add_argument('-i', '--input_file', type=str, help='input file',
                          required=True)

    mono_arg = sub.add_parser('mono', help='convert stereo to mono')
    mono_arg.add_argument('-i', '--input_file', type=str, help='input file',
                          required=True)

    resample_arg = sub.add_parser('resample', help='convert sample rate')
    resample_arg.add_argument('-i', '--input_file', type=str, help='input file',
                          required=True)
    resample_arg.add_argument('-r', '--sr', type=int, help='output sample rate, default 22050',
                          default=22050, required=True)

    info_arg = sub.add_parser('info', help='get sound file info')
    info_arg.add_argument('-f', '--file_name', type=str, help='file name',
                          required=True)

    filter_arg = sub.add_parser('filter', help='filter sound')
    filter_arg.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help="Prefilter sound file location, default = None",
    )
 
    filter_arg.add_argument(
        '-t', '--type',
        type=str,
        required=True,
        choices=['lowpass', 'highpass', 'bandpass', 'bandstop' ],
        help="Type of filter, options: lowpass, highpass, bandpass, bandstop",
    )

    filter_arg.add_argument(
        "--fc",
        type=float,
        help="Cutoff frequency",
    )

    filter_arg.add_argument(
        "--fl",
        type=float,
        help="Lower cutoff frequency",
    )

    filter_arg.add_argument(
        "--fh",
        type=float,
        help="Upper cutoff frequency",
    )
    
    filter_arg.add_argument(
        "--sample_rate",
        type=float,
        default=22050.0,
        help="Sampling rate, default: 22050.0",
    )
    targs = aparser.parse_args()

    main(targs)
