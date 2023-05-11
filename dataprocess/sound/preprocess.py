import noisereduce as nr
import librosa
import pathlib
from typing import Dict, Tuple
from numpy.typing import NDArray
import wave
import logging
import soundfile as sf

from dataprocess.util import data_process
from dataprocess.util.file import append_suffix, change_ext, check_create_folder
from dataprocess.util.data_process import read_snd_file

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
    else:
        return y, sr


def resample(data: NDArray, sr: int, tsr: int) -> Tuple[NDArray, int]:
    if sr != tsr:
        y = librosa.resample(data, sr, tsr)
        logger.debug(f"shape: {y.shape}")
        logger.debug(f"rate from {sr} to {tsr}")
        return y, tsr
    else:
        return data, sr


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
    y, sr: int, dura: float, peaks, back: float = 0.2, forth: float = 2.0
):
    cur_pos = 0.0
    clips = []
    onset_times = librosa.frames_to_time(peaks, sr=sr)
    logger.debug(f"onset times:\n{onset_times}")

    for i in onset_times:
        logger.debug(f"onset time: {i}")
        if i > cur_pos and i <= dura:
            logger.debug(f"cur_pos at loop begin: {cur_pos}")
            off = i - back
            logger.debug(f"offset before adjust: {off}")
            off = off if off > 0 else 0
            logger.debug(f"offset after adjust: {off}")
            # sub_1, _ = librosa.load(fn, sr=sr, offset=off, duration=forth)
            start = off * sr
            end = (off + forth) * sr
            end = end if end <= len(y) else len(y)
            sub_1 = y[int(start) : int(end)]
            t_dura = len(sub_1) / sr
            if t_dura < forth:
                logger.debug(f"padding: t_dura {t_dura}, forth {forth}")
                sub_1 = data_process.fill_fix_len(sub_1, sr, forth)
            clips.append(sub_1)
            cur_pos = off + forth
            logger.debug(f"cur_pos at loop end: {cur_pos}")
        else:
            logger.debug(
                f"bypass, covered by previous pick, i: {i}, cur_pos: {cur_pos}, dura: {dura}"
            )

    return clips


def find_peaks(y, sr: int):
    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        # hop_length=1024,
        # aggregate=np.median
    )
    logger.debug(f"number count onset_env len {len(onset)}")

    # onset_nc = librosa.onset.onset_detect(y=slices[0], sr=sr2, units='time')
    # print(f'number count onset detect len {len(onset_nc)}:\n{onset_nc}')

    peaks_l = librosa.util.peak_pick(
        onset, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.8, wait=10
    )
    logger.debug(f"found {len(peaks_l)} peaks")
    onset_times = librosa.frames_to_time(peaks_l, sr=sr)
    logger.debug(onset_times)

    return peaks_l


# check /home/zhangjw/tmp/xeno-canto-normal/chorthippus/gh-23/XC751338 - Chorthippus bornhalmi_mono.wav
# some empty 2 sec sound tracks
# check /home/zhangjw/tmp/xeno-normal/gh-23/XC752484 - Chorthippus bornhalmi_mono.wav
# file numbers are not continuous
def snd_peaks(in_fn: str, sr: int, back: float, forth: float, out_dir: str):
    print(f"sr: {sr}")
    if sr is None:
        y, sr, dura = read_snd_file(in_fn, sr=None, mono=True, scale=True)
    else:
        y, _, dura = read_snd_file(in_fn, sr=sr, mono=True, scale=True)

    print(f"y.shape: {y.shape}, sr: {sr}, daration: {dura}")

    peaks_l = find_peaks(y, sr)
    clips = retrieve_clips(y, sr, dura, peaks=peaks_l, back=back, forth=forth)

    out_path = check_create_folder(out_dir)

    tmp_fn = pathlib.Path(in_fn).name
    for i, c in enumerate(clips):
        out_fn = append_suffix(tmp_fn, str(i))
        if pathlib.Path(tmp_fn).suffix != ".wav":
            out_fn = change_ext(out_fn, ".wav")

        sf.write(out_path / out_fn, c, sr)


def normalize(y, sr: int, tsr: int) -> Tuple[NDArray, int]:
    y2, sr2 = to_mono(y, sr)
    y3, sr3 = resample(y, sr2, tsr)
    y4, sr4 = denoise(y3, sr3)

    return y4, sr4


def mix(fn_1: str | tuple[NDArray, int], fn_2: str) -> Exception | tuple[NDArray, int]:
    if isinstance(fn_1, str):
        y_1, sr_1 = librosa.load(fn_1, sr=None, mono=True)
    else:
        y_1, sr_1 = fn_1
    y_2, sr_2 = librosa.load(fn_2, sr=None, mono=True)

    if sr_1 != sr_2:
        return Exception(f"{fn_1} SR {sr_1} is different from {fn_2} SR {sr_2}")

    l_1 = len(y_1)
    l_2 = len(y_2)

    new_l = l_1
    if l_1 != l_2:
        print(f"{fn_1} len {l_1} is different from {fn_2} len {l_2}")
        new_l = l_1 if l_1 < l_2 else l_2
        print(f"shorter len {new_l} will be used")

    y_1 = y_1[:new_l]
    y_2 = y_2[:new_l]
    y_mix = y_1 + y_2

    return y_mix, sr_1
