import matplotlib.pyplot as plt
import numpy as np


def show_sources(sources):
    # if isinstance(sources, list):
    #     sources = {f'Source {i}': s for i, s in enumerate(sources)}

    # plt.figure(figsize=(20, 10))
    # plt.subplot(211)
    # visualize_sources_as_waveform(sources)
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    visualize_sources_as_waveform(sources, ax=ax)
    plt.tight_layout()

    return fig


def visualize_sources_as_waveform(
    audio_signals,
    ax=None,
    ch=0,
    do_mono=False,
    x_axis="time",
    colors=None,
    alphas=None,
    show_legend=True,
    **kwargs,
):
    """
    Visualizes a dictionary or list of sources with overlapping waveforms with transparency.

    The labels of each source are either the key, if a dictionary, or the
    path to the input audio file, if a list.

    Args:
        audio_signals (list or dict): List or dictionary of audio signal objects to be
          plotted.
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make each AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        colors (list, optional): Sequence of colors to use for each signal.
          Defaults to None, which uses the default matplotlib color cycle.
        alphas (list, optional): Sequence of alpha transparency to use for each signal.
          Defaults to None.
        kwargs: Additional keyword arguments to librosa.display.waveplot.
    """
    # import matplotlib.pyplot as plt

    if isinstance(audio_signals, list):
        audio_signals = {
            f"{i}:{a.path_to_input_file}": a for i, a in enumerate(audio_signals)
        }

    sorted_keys = sorted(
        audio_signals.keys(), key=lambda k: audio_signals[k].rms().mean(), reverse=True
    )

    alphas = np.linspace(0.25, 0.75, len(audio_signals)) if alphas is None else alphas

    # PLOTTING WITH PRIDE: COLORS IN MATPLOTLIB
    # https://petercbsmith.github.io/color-tutorial.html
    colors = (
        plt.rcParams["axes.prop_cycle"].by_key()["color"] if colors is None else colors
    )

    for i, key in enumerate(sorted_keys):
        val = audio_signals[key]
        color = colors[i % len(audio_signals)]
        visualize_waveform(
            val,
            ax=ax,
            ch=ch,
            do_mono=do_mono,
            x_axis=x_axis,
            alpha=alphas[i % len(audio_signals)],
            label=key,
            color=color,
        )

    if show_legend:
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2)
        ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=2)


def visualize_waveform(
    audio_signal, ax=None, ch=0, do_mono=False, x_axis="time", **kwargs
):
    """
    Wrapper around `librosa.display.waveplot` for usage with AudioSignals.

    Args:
        audio_signal (AudioSignal): AudioSignal to plot
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make the AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        kwargs: Additional keyword arguments to librosa.display.waveplot.
    """
    import librosa.display

    # import matplotlib.pyplot as plt

    if do_mono:
        audio_signal = audio_signal.to_mono(overwrite=False)

    data = np.asfortranarray(audio_signal.audio_data[ch])
    # librosa.display.waveshow(data, sr=audio_signal.sample_rate, x_axis=x_axis, **kwargs)
    # plt.ylabel('Amplitude')
    librosa.display.waveshow(
        data, ax=ax, sr=audio_signal.sample_rate, x_axis=x_axis, **kwargs
    )
    ax.set_ylabel("Amplitude")
