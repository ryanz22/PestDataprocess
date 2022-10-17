import numpy as np
from numpy.lib.npyio import save
from numpy.testing._private.utils import integer_repr
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


from .utils import (
    preproc_freq_output,
    preproc_time_input,
    plot_frames,
)
from .audio_filter import AudioFilter


def test_static(in_frames, filter_type, params, Fs, output=None, plot_dir='.'):
    """
    Test filters for recorded audio / static sinusoidal signals (mixture of frequencies).

    Inputs:
        in_freqs (list): List of frequencies to mix
        filter_type (str): Type of filter to use
        Fs (float): Sampling frequency

    Outputs:
        out_frames (np.ndarray): Processed signal
    """

    in_fft_freqs, in_fft = preproc_time_input(in_frames, Fs)

    filter = AudioFilter(filter_type, in_fft_freqs, params=params)

    plt.xlabel("Frequency")
    plt.ylabel("|H(z)|")
    plt.plot(
        in_fft_freqs[in_fft_freqs >= 0],
        np.abs(filter.filter[: filter.filter.size // 2]),
    )
    plt.savefig(f"{plot_dir}/filter.jpg")
    plt.close()

    freq_output = filter(in_fft)
    out_frames = preproc_freq_output(freq_output, in_frames.size)

    plot_frames(
        {
            "Input signal": in_frames,
            "Input FFT": (
                in_fft_freqs[in_fft_freqs >= 0],
                np.abs(in_fft[: in_fft.size // 2]),
            ),
            "Output Signal": out_frames,
            "Output FFT": (
                in_fft_freqs[in_fft_freqs >= 0],
                np.abs(freq_output[: freq_output.size // 2]),
            ),
        },
        filename=f'{plot_dir}/signal.jpg'
    )

    if output is not None:
        print(f'output: {output}')
        print(f'Fs: {Fs}')
        print(f'outframes:\n{out_frames[:20]}')
        tmp = np.array(out_frames, dtype=np.int16)
        write(output, int(Fs), tmp)
