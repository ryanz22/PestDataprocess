import numpy as np
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display
from dataprocess.cwt.cwt2 import rd_file, cwt3, cwt2, calc_scales
from dataprocess.util.data_process import replace_zeroes


def plot_scalogram(y: NDArray[np.float32], sr: int, wavename: str = 'cmor3-3',
                    totalscal: int = 256, db: bool = False):
    N = y.size
    t = np.linspace(0, N / sr, N, endpoint=False)
    scales = calc_scales(totalscal, wavename)
    if db:
        y = 20 * np.log10(np.abs(y))
    cwtmatr, frequencies = pywt.cwt(y, scales, wavename, 1.0 / sr)
    plt.contourf(t, frequencies, abs(cwtmatr))

    dpi = 100
    plt.axis('off')
    plt.gcf().set_size_inches(256 / dpi, 256 / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # x = r'./cwt_picture/train/' + str(i) + '-' + str(y_train[i]) + '.jpg'
    # plt.savefig(x)


def plot(in_fn: str, type: str, threshold):
    d1, sr1, dura1 = rd_file(in_fn)

    if type == 'spectrogram' or type == 'all':
        F_MAX = sr1 // 2
        S = librosa.feature.melspectrogram(y=d1, sr=sr1, n_mels=256, fmax=F_MAX)
        S_db = librosa.power_to_db(S, ref=np.max)

    if type == 'scalogram' or type == 'all':
        #cs1, f1 = cwt2(d1, nv=12, sr=sr1, low_freq=40)
        cs1, f1 = cwt3(d1, nv=12, sr=sr1, low_freq=40)
        print(f'shape of cs1: {cs1.shape}')
        print(f'cs1:\n{cs1[10][52000:52020]}')
        print(f'shape of f1: {f1.shape}')

    match type:
        case 'waveshow':
            #fig = plt.figure(figsize=(10,4))
            fig, ax = plt.subplots(1, 1, figsize=(10,4))
            librosa.display.waveshow(y=d1, sr=sr1, ax=ax)
            ax.set(title='wave show')

        case 'spectrogram':
            fig, ax = plt.subplots(1, 1, figsize=(10,4))
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                 sr=sr1, fmax=F_MAX, cmap='jet', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')

        case 'scalogram':
            fig, ax = plt.subplots(1, 1, figsize=(10,4))
            cs1 = replace_zeroes(cs1)
            ax.imshow(20*np.log10(np.abs(cs1)), cmap='magma', aspect='auto', norm=None, vmax=0, vmin=-60)
            ax.set(title='Scalogram')

        case 'all':
            fig, axes = plt.subplots(3, 1, figsize=(10,10), sharex=False)
            librosa.display.waveshow(y=d1, sr=sr1, ax=axes[0])
            axes[0].set(title='wave show')

            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                 sr=sr1, fmax=F_MAX, cmap='jet', ax=axes[1])
            axes[1].set(title='Mel-frequency spectrogram')

            cs1 = replace_zeroes(cs1)
            axes[2].imshow(20*np.log10(np.abs(cs1)), cmap='magma', aspect='auto', norm=None, vmax=0, vmin=-60)
            axes[2].set(title='Scalogram')

            fig.subplots_adjust(hspace=0.4)

    return fig
