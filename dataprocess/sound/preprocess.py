import noisereduce as nr
import librosa
import pathlib
from util import append_suffix, change_ext
from typing import Dict, Tuple
from numpy.typing import NDArray
import wave


# https://github.com/timsainb/noisereduce
def denoise(data: NDArray, sr: int) -> Tuple[NDArray, int]:
    print(f'sample rate: {sr}')
    print(f'data shape: {data.shape}')
    print(f'array dtype: {data.dtype}')
    # drone_rate, drone_data = wavfile.read(DRONE_WAV_FN)
    # reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=drone_data)
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    return reduced_noise, sr


def mono(y: NDArray, sr: int) -> Tuple[NDArray, int]:
    print(f'shape: {y.shape}')
    print(f'rate: {sr}')
    if is_stereo(y):
        print('this is a stereo sound track')
        y_mono = librosa.to_mono(y)
        print(f'after to mono, shape: {y_mono.shape}')
    return y_mono, sr


def resample(data: NDArray, sr: int, tsr: int) -> Tuple[NDArray, int]:
    y = librosa.resample(data, sr, tsr)
    print(f'shape: {y.shape}')
    print(f'rate from {sr} to {tsr}')
    return y, tsr


def is_stereo(y: NDArray) -> bool:
    return y.shape[0] == 2


def is_stereo_sound(fn: str) -> bool:
    y, sr = librosa.load(fn, sr=None, mono=False)
    print(f'shape: {y.shape}')
    print(f'rate: {sr}')
    return is_stereo(y)


def sound_file_info(fn: str) -> Dict:
    y, sr = librosa.load(fn, sr=None, mono=False)
    print(f'sound_file_info y dtype: {y.dtype}')
    dur = librosa.get_duration(y=y, sr=sr)

    if pathlib.Path(fn).suffix == '.wav':
        ro = wave.open(fn, 'rb')
        wav_info = {
            'is_wav': True,
            'sample_width': ro.getsampwidth(),
        }
        ro.close()
    else:
        wav_info = {
            'is_wav': False
        }

    return {
        'file_name': fn,
        'size (bytes)': y.size * y.itemsize,
        'sample_rate': sr,
        'duration (sec)': dur,
        'is_stereo': is_stereo(y),
        'wav_info': wav_info
    }
