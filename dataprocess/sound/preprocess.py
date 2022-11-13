import noisereduce as nr
import librosa
import pathlib
from typing import Dict, Tuple
from numpy.typing import NDArray
import wave
import logging
from dataprocess.util import data_process

logger = logging.getLogger(__name__)


# https://github.com/timsainb/noisereduce
def denoise(data: NDArray, sr: int) -> Tuple[NDArray, int]:
    logger.debug(f"sample rate: {sr}")
    logger.debug(f"data shape: {data.shape}")
    logger.debug(f"array dtype: {data.dtype}")
    # drone_rate, drone_data = wavfile.read(DRONE_WAV_FN)
    # reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=drone_data)
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    return reduced_noise, sr


def to_mono(y: NDArray, sr: int) -> Tuple[NDArray, int]:
    logger.debug(f"shape: {y.shape}")
    logger.debug(f"rate: {sr}")
    if is_stereo(y):
        logger.debug("this is a stereo sound track")
        y_mono = librosa.to_mono(y)
        logger.debug(f"after to mono, shape: {y_mono.shape}")
    return y_mono, sr


def resample(data: NDArray, sr: int, tsr: int) -> Tuple[NDArray, int]:
    y = librosa.resample(data, sr, tsr)
    logger.debug(f"shape: {y.shape}")
    logger.debug(f"rate from {sr} to {tsr}")
    return y, tsr


def is_stereo(y: NDArray) -> bool:
    return y.shape[0] == 2


def is_stereo_sound(fn: str) -> bool:
    y, sr = librosa.load(fn, sr=None, mono=False)
    logger.debug(f"shape: {y.shape}")
    logger.debug(f"rate: {sr}")
    return is_stereo(y)


def sound_file_info(fn: str) -> Dict:
    y, sr = librosa.load(fn, sr=None, mono=False)
    logger.debug(f"sound_file_info y dtype: {y.dtype}")
    dur = librosa.get_duration(y=y, sr=sr)

    if pathlib.Path(fn).suffix == ".wav":
        ro = wave.open(fn, "rb")
        wav_info = {
            "is_wav": True,
            "sample_width": ro.getsampwidth(),
        }
        ro.close()
    else:
        wav_info = {"is_wav": False}

    return {
        "file_name": fn,
        "size (bytes)": y.size * y.itemsize,
        "sample_rate": sr,
        "duration (sec)": dur,
        "is_stereo": is_stereo(y),
        "wav_info": wav_info,
    }


def retrieve_clips(
    fn: str, peaks, sr: int = None, back: float = 0.5, forth: float = 1.5
):
    y, sr = librosa.load(fn, sr=sr)
    dura = librosa.get_duration(y=y, sr=sr)
    cur_pos = 0.0
    clips = []
    onset_times = librosa.frames_to_time(peaks)
    logger.debug(f"onset times:\n{onset_times}")

    for i in onset_times:
        logger.debug(f"onset time: {i}")
        if i > cur_pos and i <= dura:
            logger.debug(f"cur_pos at loop begin: {cur_pos}")
            off = round(i) - back
            logger.debug(f"offset before adjust: {off}")
            off = off if off > 0 else 0
            logger.debug(f"offset after adjust: {off}")
            sub_1, _ = librosa.load(fn, sr=sr, offset=off, duration=forth)
            t_dura = librosa.get_duration(sub_1, sr)
            if t_dura < forth:
                sub_1 = data_process.fill_fix_len(sub_1, sr, forth)
            clips.append(sub_1)
            cur_pos = off + forth
            logger.debug(f"cur_pos at loop end: {cur_pos}")
        else:
            logger.debug("bypass, too close")

    return clips
