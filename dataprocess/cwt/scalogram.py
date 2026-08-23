import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display
import cv2

from dataprocess.cwt.cwt2 import cwt3, cwt2, calc_scales, img_resize
from dataprocess.util.data_process import replace_zeroes
from dataprocess.sound.filter_util import preproc_time_input

import logging

logger = logging.getLogger(__name__)


def plot_scalogram(
    y: NDArray[np.float32],
    sr: int,
    wavename: str = "cmor3-3",
    totalscal: int = 256,
    db: bool = False,
):
    N = y.size
    t = np.linspace(0, N / sr, N, endpoint=False)
    scales = calc_scales(totalscal, wavename)
    if db:
        y = 20 * np.log10(np.abs(y))
    cwtmatr, frequencies = pywt.cwt(y, scales, wavename, 1.0 / sr)
    plt.contourf(t, frequencies, abs(cwtmatr))

    dpi = 100
    plt.axis("off")
    plt.gcf().set_size_inches(256 / dpi, 256 / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # x = r'./cwt_picture/train/' + str(i) + '-' + str(y_train[i]) + '.jpg'
    # plt.savefig(x)


def fft_process(d2, sr):
    from scipy.fft import rfft, rfftfreq

    N = len(d2)
    # print(f"N: {N}")
    yf = rfft(d2)
    xf = rfftfreq(N, 1 / sr)
    # yf = np.fft.rfft(d2)
    # xf = np.fft.rfftfreq(N, 1 / sr)

    return xf, yf


def dim_to_inches(t: str, w: float, h: float, dpi: int) -> Tuple[float, float]:
    if t == "cm":
        return w / 2.54, h / 2.54
    elif t == "px":
        return w / dpi, h / dpi
    return w, h


def dim_to_pixels(t: str, w: float, h: float, dpi: int) -> Tuple[float, float]:
    if t == "cm":
        return w * dpi / 2.54, h * dpi / 2.54
    elif t == "inch":
        return w * dpi, h * dpi
    return w, h


def plot_waveshow(
    d,
    sr: int,
    out_fn: str,
    dim=("inch", 10, 4),
    show_scale: bool = False,
    dpi: int = 256,
):
    t, w, h = dim
    dim_w, dim_h = dim_to_inches(t, w, h, dpi)
    fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
    librosa.display.waveshow(y=d, sr=sr, ax=ax)
    if show_scale:
        ax.set(title="Wave Show")
    else:
        ax.set_axis_off()

    fig.savefig(out_fn)
    plt.close(fig)


def plot_spectro(
    d,
    sr: int,
    out_fn: str,
    threshold: int = -60,
    cmap: str = "magma",
    dim=("inch", 10, 4),
    show_scale: bool = False,
    dpi: int = 256,
    fmax: int = -1,
):
    F_MAX = fmax if fmax > 0 else sr // 2
    S = librosa.feature.melspectrogram(y=d, sr=sr, n_mels=256, fmax=F_MAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    t, w, h = dim
    dim_w, dim_h = dim_to_inches(t, w, h, dpi)

    if show_scale:
        fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
        # fig = plt.figure(figsize=(dim_w, dim_h))

        img = librosa.display.specshow(
            # S_db, x_axis="time", y_axis="mel", sr=sr, fmax=F_MAX, cmap="jet", ax=ax
            S_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            fmax=F_MAX,
            cmap=cmap,
            ax=ax,
            vmax=0,
            vmin=threshold,
        )

        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="Mel-Frequency Spectrogram")

        fig.savefig(out_fn)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(dim_w, dim_h))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            fmax=F_MAX,
            cmap=cmap,
            ax=ax,
            vmax=0,
            vmin=threshold,
        )

        # import matplotlib

        # bbox = matplotlib.transforms.Bbox(((0, 0), (dim_w, dim_h)))
        fig.savefig(out_fn, bbox_inches=0, pad_inches=0, dpi=dpi)
        plt.close(fig)
        # dim_w, dim_h = dim_to_pixels(t, w, h, dpi)
        # img = img_resize(
        #     S_db, w=int(dim_w), h=int(dim_h), log=False, lthres=threshold, cmap=cmap
        # )
        # cv2.imwrite(out_fn, img)


def plot_scalo(
    d,
    sr: int,
    out_fn: str,
    threshold: int = -60,
    cmap: str = "magma",
    dim=("inch", 10, 4),
    show_scale: bool = False,
    dpi: int = 256,
):
    cs1, f1 = cwt2(d, nv=12, sr=sr, low_freq=40)
    # cs1, f1 = cwt3(d1, nv=12, sr=sr1, low_freq=40)
    logger.debug(f"shape of cs1: {cs1.shape}")
    logger.debug(f"cs1:\n{cs1[10][52000:52020]}")
    logger.debug(f"shape of f1: {f1.shape}")

    t, w, h = dim

    cs1 = replace_zeroes(cs1)

    if show_scale:
        dim_w, dim_h = dim_to_inches(t, w, h, dpi)
        fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h), dpi=dpi)
        ax.set_xlabel("Time")
        ax.set(title="Scalogram")
        img = ax.imshow(
            20 * np.log10(np.abs(cs1)),
            cmap=cmap,
            aspect="auto",
            norm=None,
            vmax=0,
            vmin=threshold,
            extent=[0.0, len(d) / float(sr), cs1.shape[0], 0],
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.savefig(out_fn)
        plt.close(fig)
    else:
        dim_w, dim_h = dim_to_pixels(t, w, h, dpi)
        img = img_resize(
            cs1, w=int(dim_w), h=int(dim_h), log=True, lthres=threshold, cmap=cmap
        )
        cv2.imwrite(out_fn, img)


def plot_fft(
    d,
    sr: int,
    out_fn: str,
    dim=("inch", 10, 4),
    show_scale: bool = False,
    dpi: int = 256,
):
    xf, yf = fft_process(d, sr)

    t, w, h = dim
    dim_w, dim_h = dim_to_inches(t, w, h, dpi)

    fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
    ax.plot(xf, np.abs(yf))

    if show_scale:
        ax.set(title="FFT")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
    else:
        ax.set_axis_off()

    fig.savefig(out_fn)
    plt.close(fig)


def plot_all(
    d,
    sr,
    out_fn: str | None,
    threshold: int = -60,
    cmap: str = "magma",
    dim=("inch", 10, 14),
    show_scale: bool = False,
    dpi: int = 256,
):
    F_MAX = sr // 2
    S = librosa.feature.melspectrogram(y=d, sr=sr, n_mels=256, fmax=F_MAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    cs1, f1 = cwt2(d, nv=12, sr=sr, low_freq=40)
    # cs1, f1 = cwt3(d1, nv=12, sr=sr1, low_freq=40)
    logger.debug(f"shape of cs1: {cs1.shape}")
    logger.debug(f"cs1:\n{cs1[10][52000:52020]}")
    logger.debug(f"shape of f1: {f1.shape}")

    t, w, h = dim
    dim_w, dim_h = dim_to_inches(t, w, h, dpi)

    # print(f"plot graph as {dim_w}inch x {dim_h}inch")

    fig, axes = plt.subplots(4, 1, figsize=(dim_w, dim_h), sharex=False)
    # fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    librosa.display.waveshow(y=d, sr=sr, ax=axes[0])

    librosa.display.specshow(
        S_db,
        x_axis="time" if show_scale else None,
        y_axis="mel" if show_scale else None,
        sr=sr,
        fmax=F_MAX,
        cmap=cmap,
        ax=axes[1],
        vmin=threshold,
    )

    cs1 = replace_zeroes(cs1)
    axes[2].imshow(
        20 * np.log10(np.abs(cs1)),
        cmap=cmap,
        aspect="auto",
        norm=None,
        vmax=0,
        vmin=threshold,
        extent=[0.0, len(d) / float(sr), cs1.shape[0], 0],
    )

    xf, yf = fft_process(d, sr)
    axes[3].plot(xf, np.abs(yf))

    if show_scale:
        axes[0].set(title="Wavefrom (time-domain)")
        axes[1].set(title="Mel-frequency spectrogram (STFT)")
        axes[2].set(title="Scalogram (CWT)")
        axes[3].set(title="Fast Fourier Transform (frequency-domain)")
        # axes[3].set_xlabel("Frequence")
        # axes[3].set_ylabel("mag")
    else:
        # plt.axis("off")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[2].set_axis_off()
        axes[3].set_axis_off()

    fig.subplots_adjust(hspace=0.5)

    if out_fn:
        fig.savefig(out_fn)

    return fig


def plot_peaks(y, onset, peaks, sr: int):
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr)
    # print(f'times:\n{times[:10]}')
    D = np.abs(librosa.stft(y))
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis="time", y_axis="log"
    )
    plt.title("Power spectrogram")
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(times, onset, label="Onset strength")
    plt.vlines(
        times[peaks],
        0,
        onset.max(),
        color="r",
        alpha=0.9,
        linestyle="--",
        label="Onsets",
    )
    plt.axis("tight")
    plt.legend(frameon=True, framealpha=0.75)

    return fig
