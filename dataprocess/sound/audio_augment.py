from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    AddGaussianSNR,
    AddGaussianNoise,
    PitchShift,
    Shift,
    AddBackgroundNoise,
    PolarityInversion,
    Reverse,
)

import pathlib
import functional as pyf
import soundfile as sf

from dataprocess.util.data_process import read_snd_file
from dataprocess.util.file import append_suffix


def augment(d, sr: int, bg: str, count: int = 20, noise: bool = False):
    transforms = [
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.3),
        Reverse(p=0.3),
    ]

    if noise:
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.3),

    if bg is not None:
        transforms.append(
            AddBackgroundNoise(
                sounds_path=bg,
                # sounds_path='../data/sound/background_sounds/mono_5s/bird_mono_44100_2.0_sliced.wav',
                min_snr_in_db=10.0,
                max_snr_in_db=30.0,
                noise_transform=PolarityInversion(),
                p=0.5,
            )
        )
    aug = Compose(transforms)

    return pyf.seq(range(count)).map(lambda _: aug(d, sr)).list()


def augment_single(in_fn: str, count: int, bg: str, out: str, noise: bool):
    d, sr, _ = read_snd_file(in_fn, sr=None, mono=False, scale=True)
    olist = augment(d, sr, bg=bg, count=count, noise=noise)
    tmp_fn = pathlib.Path(in_fn).name
    out_p = pathlib.Path(out)
    out_fn_list = (
        pyf.seq(range(len(olist)))
        .map(lambda i: str(out_p / append_suffix(tmp_fn, f"{i}")))
        .list()
    )
    print(f"output fn:\n{out_fn_list}")
    pyf.seq(out_fn_list).zip(pyf.seq(olist)).for_each(
        lambda t: sf.write(t[0], t[1], sr)
    )
