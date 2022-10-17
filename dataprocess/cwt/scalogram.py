import numpy as np
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt
import pywt


def calc_scales(totalscal: int=256, wavename: str='cmor3-3'):
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    return scales

    
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