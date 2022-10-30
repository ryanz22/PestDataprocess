import math
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import librosa
import scipy
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
import cv2
from functools import partial
import io
from typing import List, Tuple
import gc
from timeit import default_timer as timer
from datetime import timedelta

# import ray
import gc
import pywt
from dataprocess.util.data_process import replace_zeroes

import logging

logger = logging.getLogger(__name__)


# Continous Wavelet Transform with Morlet wavelet
# Original code by Alexander Neergaard, https://github.com/neergaard/CWT
#
# Parameters:
#   data: input data
#   nv: # of voices (scales) per octave
#   sr: sampling frequency (Hz)
#   low_freq: lowest frequency (Hz) of interest (limts longest scale)
def cwt2(data, nv=10, sr=1.0, low_freq=0.0):
    data -= np.mean(data)
    n_orig = data.size
    ds = 1 / nv
    dt = 1 / sr

    # Pad data symmetrically
    padvalue = n_orig // 2
    x = np.concatenate((np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))
    n = x.size
    logger.debug(f"size of x: {n}, memory of x {x.size * x.itemsize}")

    # Define scales
    _, _, wavscales = getDefaultScales(n_orig, ds, sr, low_freq)
    num_scales = wavscales.size
    logger.debug(f"num of scales: {num_scales}")
    logger.debug(f"scales:\n{wavscales}")

    # Frequency vector sampling the Fourier transform of the wavelet
    # omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
    omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float32)
    omega *= (2 * np.pi) / n
    omega = np.concatenate(
        (
            np.array([0]),
            omega,
            -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1],
        )
    )
    logger.debug(
        f"size of omega: {omega.size}, memory of omega {omega.size * omega.itemsize}"
    )

    # Compute FFT of the (padded) time series
    f = np.fft.fft(x)
    logger.debug("size of f: %i, memory of f %i", f.size, f.size * f.itemsize)

    # Loop through all the scales and compute wavelet Fourier transform
    psift, freq = waveft(omega, wavscales)
    logger.debug(
        f"size of psift: {psift.size}, memory of psift {psift.size * psift.itemsize}"
    )

    # Inverse transform to obtain the wavelet coefficients.
    cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)
    logger.debug(
        f"size of cwtcfs: {cwtcfs.size}, memory of cwtcfs {cwtcfs.size * cwtcfs.itemsize}"
    )
    cfs = cwtcfs[:, padvalue : padvalue + n_orig]
    freq = freq * sr

    del psift
    del cwtcfs

    return cfs, freq


def calc_scales(totalscal: int = 256, wavename: str = "cmor3-3"):
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    return scales


def cwt3(data, nv=10, sr=1.0, low_freq=0.0):
    data -= np.mean(data)
    # n_orig = data.size
    # ds = 1 / nv
    # _, _, wavscales = getDefaultScales(n_orig, ds, sr, low_freq)
    wavlet = "cmor1.5-1"
    wavscales = calc_scales(256, wavlet)
    # _, _, wavscales = getDefaultScales(data.size, 1/nv, sr, 40.0)
    logger.debug(f"num of scales: {wavscales.size}")
    logger.debug(f"scales:\n{wavscales}")

    cfs, freq = pywt.cwt(data, wavscales, wavlet, 1 / sr)

    return cfs, freq


def getDefaultScales(n, ds, sr, low_freq):
    nv = 1 / ds
    # Smallest useful scale (default 2 for Morlet)
    s0 = 2

    # Determine longest useful scale for wavelet
    max_scale = n // (np.sqrt(2) * s0)
    if max_scale <= 1:
        max_scale = n // 2
    max_scale = np.floor(nv * np.log2(max_scale))
    a0 = 2**ds
    scales = s0 * a0 ** np.arange(0, max_scale + 1)

    # filter out scales below low_freq
    fourier_factor = 6 / (2 * np.pi)
    frequencies = sr * fourier_factor / scales
    frequencies = frequencies[frequencies >= low_freq]
    scales = scales[0 : len(frequencies)]

    return s0, ds, scales


def waveft(omega, scales):
    num_freq = omega.size
    num_scales = scales.size
    wft = np.zeros([num_scales, num_freq])

    gC = 6
    mul = 2
    for jj, scale in enumerate(scales):
        expnt = -((scale * omega - gC) ** 2) / 2 * (omega > 0)
        wft[jj,] = (
            mul * np.exp(expnt) * (omega > 0)
        )

    fourier_factor = gC / (2 * np.pi)
    frequencies = fourier_factor / scales

    return wft, frequencies


def img_resize(cs, w=512, h=512, log=True, lthres=-30, cmap: str = "magma"):
    buf = io.BytesIO()
    if log == True:
        plt.imsave(
            buf, 20 * np.log10(np.abs(cs)), cmap=cmap, format="png", vmax=0, vmin=lthres
        )
    else:
        plt.imsave(buf, np.abs(cs), cmap=cmap, format="png")
    buf.seek(0)
    img_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


# Parameters:
#   filename: mp3-file
#   voices: # of scales per octave
#   sr: sampling frequency (Hz)
#   low_freq: low freq cutoff (Hz)
#   thres: scaleogram threshold (dB)
#   prom: peak detect prominence
#   peakdur: peak extension (s)
#   sigthres: smallest signature detection to process (s)
#   siglen: length of output signature (s)
#   img_size: output image size
#   outdir: output directory
# @ray.remote
def scaleo_extract(
    filename,
    voices=12,
    sr=22050,
    low_freq=40,
    thres=-30,
    prom=0.3,
    peakdur=0.3,
    sigthres=1,
    siglen=2,
    img_size=512,
    outdir=".",
):
    logger.debug(f"threshold: {thres}, img_size: {img_size}")

    start = timer()
    d, sr, dura = rd_file(filename, sr=sr)
    logger.debug(f"scaleo_extract, data type: {d.dtype}")

    end = timer()
    logger.debug(f"rd_file time: {timedelta(seconds=end-start)}")

    start = timer()
    cs, _ = cwt2(d, nv=voices, sr=sr, low_freq=low_freq)  # wavelet transform
    # cs, _ = cwt3(d, nv=voices, sr=sr, low_freq=low_freq) # wavelet transform
    # cs = replace_zeroes(cs)

    # del d # free d
    # gc.collect()
    end = timer()
    logger.debug(f"cwt time: {timedelta(seconds=end-start)}")

    start = timer()
    v = calc_var(cs, thres)  # coefficient variance
    peaks, _ = signal.find_peaks(v, prominence=prom)
    m = mask_sig(len(v), peaks, sr=sr, dur=peakdur)  # create signal mask
    df = get_regions(m, sr, filename.split("/")[-2], filename.split("/")[-1])
    df = df[df.Duration >= sigthres]  # filter out insignificant signatures
    df = df.reset_index(drop=True)
    end = timer()
    logger.debug(
        f"calc_vars mask_sig, get_regions time: {timedelta(seconds=end-start)}"
    )

    start = timer()
    if len(df) > 0:
        for i in range(len(df)):
            img = img_resize(
                cs[:, df.Start[i] : df.Start[i] + siglen * sr],
                w=img_size,
                h=img_size,
                log=True,
                lthres=thres,
            )
            fn = (
                df.Species[i]
                + "-"
                + filename.split("/")[-1].split(".")[-2]
                + "-{:03d}.jpg".format(i)
            )
            cv2.imwrite(outdir + "/" + fn, img)

    end = timer()
    logger.debug(f"img_resize, imwrite time: {timedelta(seconds=end-start)}")

    # return df


def rd_file(
    fname, sr: int = 22050, offset=0, duration=60
) -> Tuple[NDArray, int, float]:
    data, sr = librosa.load(
        fname, sr=sr, mono=True, offset=offset, duration=duration, dtype=np.float32
    )
    logger.debug(f"data type: {data.dtype}")
    # logger.debug(data[2000:2020])
    mean = data.mean()
    data = preprocessing.minmax_scale(data - mean, feature_range=(-1, 1))
    # logger.debug(data[2000:2020])
    duration = librosa.get_duration(y=data, sr=sr)
    logger.debug(f"data size should be {duration * sr * 4}")
    logger.debug(f"size of rd_file data: {data.size * data.itemsize}")

    return data, sr, duration
    # return data.astype(np.float16)


# calculate variance of coefficients
def calc_var(cs, thres):
    c = 20 * np.log10(np.abs(cs))
    c[c < thres] = 0.0
    e = np.var(c, axis=0)
    return e / max(e)


def mask_sig(n, peaks, sr=22050, dur=0.1):
    mask = np.zeros(n)
    subm = int(sr * dur * 0.5)
    if len(peaks > 0):
        for i in range(len(peaks)):
            mask[max(peaks[i] - subm, 0) : min(peaks[i] + subm, n)] = 1
    return mask


def get_mask(vdata, prom=0.2, dur=0.2, sr=22050):
    peaks, _ = find_peaks(vdata, prominence=prom)
    return mask_sig(len(vdata), peaks, sr, dur)


def get_regions(mask, sr, species, filename):
    regions = scipy.ndimage.find_objects(scipy.ndimage.label(mask)[0])
    regs = []
    for r in regions:
        dur = round((r[0].stop - r[0].start) / sr, 3)
        regs.append([r[0].start, r[0].stop, dur, species, filename])
    return pd.DataFrame(regs, columns=["Start", "End", "Duration", "Species", "File"])


def plot_sigx2(
    d1,
    d2,
    name1="data 1",
    name2="data 2",
    SR=22050,
    CMAP="magma",
    cwt=True,
    db_range=30,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    d = [d1, d2]
    name = [name1, name2]
    for i in range(2):
        if cwt == True:
            cs, _ = cwt2(d[i], nv=12, sr=SR, low_freq=40)
            axes[i].imshow(
                20 * np.log10(np.abs(cs)),
                cmap=CMAP,
                aspect="auto",
                norm=None,
                vmax=0,
                vmin=-db_range,
            )
        else:
            f, t, Sxx = signal.spectrogram(d[i], SR)
            axes[i].pcolormesh(
                t,
                f,
                20 * np.log10(Sxx),
                shading="auto",
                cmap=CMAP,
                vmax=-60,
                vmin=-60 - db_range,
            )
        axes[i].set_title(name[i])
    plt.show()


def batch_extract(
    plist: List[str], out_dir: str, batch: int = 4, thres: int = -30, imgsize: int = 512
):
    # def f(i):
    #     scaleo_extract(i, outdir=out_dir)

    f2 = partial(scaleo_extract, outdir=out_dir, thres=thres, img_size=imgsize)

    from multiprocessing import Pool

    with Pool(batch) as p:
        p.map(f2, plist)
