import numpy as np
from numpy.lib.npyio import save
from numpy.testing._private.utils import integer_repr
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

from .audio_filter import AudioFilter


def freq_filter(in_frames, filter_type, params, Fs, do_plot=False, plot_dir='.'):
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

    if do_plot:
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

    if do_plot:
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

    return out_frames

# ---------------------------- AUDIO UTILS ----------------------------#


def load_audio_file(prerec_file):
    """
    To record sound input from file.

    Inputs:
        loc (str): File from which sound is recorded.

    Outputs:
        np_frames (np.ndarray): input frames as numpy array.
    """

    soundfile = read(prerec_file)
    print(f'sound shape: {soundfile[1].shape}')
    print(f'sound data type: {soundfile[1].dtype}')
    in_frames = np.array(soundfile[1], dtype=float)

    return in_frames


# ---------------------------- PLOT & FFT UTILS ----------------------------#


def plot_frames(frame_dict, n_cols=2, filename="../assets/plots/signals.jpg"):
    """
    Plot frames as matplotlib plot.

    Inputs:
        frames (np.ndarray): Frames (signal) to plot

    Outputs:
        None
    """

    frames = list(frame_dict.values())
    titles = list(frame_dict)

    fig, axes = plt.subplots(
        len(frames) // n_cols, n_cols, constrained_layout=True
    )

    pos = 0
    for row in axes:
        for col in row:
            if isinstance(frames[pos], Tuple):
                col.plot(frames[pos][0], frames[pos][1])
            else:
                col.plot(frames[pos])

            col.set_title(titles[pos])
            pos += 1

    fig.savefig(filename)
    plt.close()


def generate_mix_freq(freqs, noise=False):
    """
    Generate frequency mixture of sine waves of freqeuncies in freqs

    Inputs:
        freqs (list): list of frequencies
        size (int): size of sample

    Outputs:
        result (np.ndarray): Sum of sine-waves of all frequencies in freqs.
    """

    Fs = 22050
    T = 4
    range = np.arange(T * Fs) / Fs
    result = np.zeros_like(range)

    for freq in freqs:
        sin_freq = np.sin(2 * np.pi * freq * range)
        result += sin_freq

    if noise:
        result += np.random.randn(result.size)

    return result


def next_2_pow(n):
    return 2 ** (np.ceil(np.log2(n)))


def preproc_time_input(in_frames, Fs):
    """
    Preprocess the input signal and obtain its (zero-centered) FFT.

    Inputs:
        in_frames (np.ndarray): input signals in time domain in form of np.array

    Outputs:
        freq_output (np.ndarray): frequency domain (FFT) output for given input.
    """

    input_fft = np.fft.fft(in_frames, n=int(next_2_pow(in_frames.shape[-1])))
    fft_freqs = np.fft.fftfreq(n=len(input_fft), d=1 / Fs)
    return fft_freqs, input_fft


def preproc_freq_output(freq_output, size):
    """
    Preprocess the frequency domain output signal and obtain its time domain signal.

    Inputs:
        freq_output (np.ndarray): output signals in frequency domain in form of np.array

    Outputs:
        freq_output (np.ndarray): time domain (IFFT) output for given input.
    """

    ifft = np.fft.ifft(freq_output)[:size].real

    return ifft
