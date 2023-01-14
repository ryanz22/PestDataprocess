import numpy as np
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display
from dataprocess.cwt.cwt2 import rd_file, cwt3, cwt2, calc_scales
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
    print(f"N: {N}")
    yf = rfft(d2)
    xf = rfftfreq(N, 1 / sr)
    # yf = np.fft.rfft(d2)
    # xf = np.fft.rfftfreq(N, 1 / sr)

    return xf, yf


def plot_data(
    d,
    sr,
    ptype: str,
    threshold: int = -60,
    cmap: str = "magma",
    dim=("inch", 10, 4),
    show_scale: bool = False,
):
    if ptype == "spectrogram" or ptype == "all":
        F_MAX = sr // 2
        S = librosa.feature.melspectrogram(y=d, sr=sr, n_mels=256, fmax=F_MAX)
        S_db = librosa.power_to_db(S, ref=np.max)

    if ptype == "scalogram" or ptype == "all":
        cs1, f1 = cwt2(d, nv=12, sr=sr, low_freq=40)
        # cs1, f1 = cwt3(d1, nv=12, sr=sr1, low_freq=40)
        logger.debug(f"shape of cs1: {cs1.shape}")
        logger.debug(f"cs1:\n{cs1[10][52000:52020]}")
        logger.debug(f"shape of f1: {f1.shape}")

    dim_t, dim_w, dim_h = dim
    match dim_t:
        case "cm":
            cm = 1 / 2.54
            dim_w *= cm
            dim_h *= cm
        case "px":
            dpi = plt.rcParams["figure.dpi"]  # pixel in inches
            print(f"calc px, dpi: {dpi}")
            px = 1 / dpi
            dim_w *= px
            dim_h *= px

    print(f"plot graph as {dim_w}inch x {dim_h}inch")

    match ptype:
        case "waveshow":
            # fig = plt.figure(figsize=(10,4))
            fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
            librosa.display.waveshow(y=d, sr=sr, ax=ax)
            if show_scale:
                ax.set(title="Wave Show")

        case "spectrogram":
            fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
            # fig = plt.figure(figsize=(dim_w, dim_h))

            if show_scale:
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                ax.set(title="Mel-Frequency Spectrogram")

            img = librosa.display.specshow(
                # S_db, x_axis="time", y_axis="mel", sr=sr, fmax=F_MAX, cmap="jet", ax=ax
                S_db,
                x_axis="time" if show_scale else None,
                y_axis="mel" if show_scale else None,
                sr=sr,
                fmax=F_MAX,
                cmap=cmap,
                ax=ax,
                vmin=threshold,
            )

        case "fft":
            xf, yf = fft_process(d, sr)
            fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
            ax.plot(xf, np.abs(yf))

            if show_scale:
                ax.set(title="FFT")
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Magnitude")

        case "scalogram":
            cs1 = replace_zeroes(cs1)

            # fig, ax = plt.subplots(1, 1, figsize=(dim_w, dim_h))
            fig = plt.figure(figsize=(dim_w, dim_h))

            if show_scale:
                ax = plt.Axes(fig, [0.1, 0.1, 0.8, 0.8])
                fig.add_axes(ax)
                ax.set_xlabel("Time")
                ax.set(title="Scalogram")
            else:
                ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                fig.add_axes(ax)
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                ax.set_axis_off()
                # fig.patch.set_visible(False)

            ax.imshow(
                20 * np.log10(np.abs(cs1)),
                cmap=cmap,
                aspect="auto",
                norm=None,
                vmax=0,
                vmin=threshold,
                extent=[0.0, len(d) / float(sr), cs1.shape[0], 0],
            )

        case "all":
            fig, axes = plt.subplots(4, 1, figsize=(dim_w, dim_h * 2.5), sharex=False)
            # fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
            librosa.display.waveshow(y=d, sr=sr, ax=axes[0])

            img = librosa.display.specshow(
                S_db,
                x_axis="time",
                y_axis="mel",
                sr=sr,
                fmax=F_MAX,
                cmap=cmap,
                ax=axes[1],
            )

            cs1 = replace_zeroes(cs1)
            axes[2].imshow(
                20 * np.log10(np.abs(cs1)),
                cmap=cmap,
                aspect="auto",
                norm=None,
                vmax=0,
                vmin=-60,
                extent=[0.0, len(d) / float(sr), cs1.shape[0], 0],
            )

            xf, yf = fft_process(d, sr)
            axes[3].plot(xf, np.abs(yf))

            if show_scale:
                axes[0].set(title="wave show")
                axes[1].set(title="Mel-frequency spectrogram")
                axes[2].set(title="Scalogram")
                axes[3].set(title="FFT")
                # axes[3].set_xlabel("Frequence")
                # axes[3].set_ylabel("mag")

            fig.subplots_adjust(hspace=0.5)

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
