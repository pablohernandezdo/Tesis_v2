import segyio
import numpy as np
from numpy.random import default_rng

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import scipy.fftpack as sfft

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/Vibroseis/048").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Vibroseis/118").mkdir(exist_ok=True)
    Path("Imgs/Vibroseis/117").mkdir(exist_ok=True)
    Path("Imgs/Vibroseis/047").mkdir(exist_ok=True)

    Path("Animations/Vibroseis/048").mkdir(parents=True, exist_ok=True)
    Path("Animations/Vibroseis/118").mkdir(exist_ok=True)
    Path("Animations/Vibroseis/117").mkdir(exist_ok=True)
    Path("Animations/Vibroseis/047").mkdir(exist_ok=True)

    # File 048
    # 8700 trazas de 30000 muestras
    f048 = '../Data/Vibroseis/PoroTomo_iDAS16043_160325140048.sgy'

    # Sampling frequency and data
    fs, traces = read_segy(f048)

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Vibroseis', '048')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Vibroseis', '048', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', '048')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', 'Vibroseis_norm', norm=True)

    # File 118
    # 8700 trazas de 30000 muestras
    f118 = '../Data/Vibroseis/PoroTomo_iDAS16043_160325140118.sgy'

    # Sampling frequency and data
    fs, traces = read_segy(f118)

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Vibroseis', '118')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Vibroseis', '118', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', '118')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', 'Vibroseis_norm', norm=True)

    # File 117
    # 380 trazas de 30000 muestras
    f117 = '../Data/Vibroseis/PoroTomo_iDAS025_160325140117.sgy'

    # Sampling frequency and data
    fs, traces = read_segy(f117)

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Vibroseis', '117')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Vibroseis', '117', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', '117')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', 'Vibroseis_norm', norm=True)

    # File 047
    # 380 trazas de 30000 muestras
    f047 = '../Data/Vibroseis/PoroTomo_iDAS025_160325140047.sgy'

    # Sampling frequency and data
    fs, traces = read_segy(f047)

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Vibroseis', '047')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Vibroseis', '047', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', '047')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 50, 'Vibroseis', 'Vibroseis_norm', norm=True)


def plot_traces(traces, fs, n, dataset, filename, rand=True, pre_traces=None):
    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Traces to plot
    trtp = []

    if rand:
        # Init rng
        rng = default_rng()

        # Traces to plot numbers
        trtp_ids = rng.choice(len(traces), size=n, replace=False)
        trtp_ids.sort()

        # Retrieve selected traces
        for idx, trace in enumerate(traces):
            if idx in trtp_ids:
                trtp.append(trace)

    else:
        trtp_ids = pre_traces

        # Retrieve selected traces
        for idx, trace in enumerate(traces):
            if idx in trtp_ids:
                trtp.append(trace)

    # Plot traces in trtp with their spectrum
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        gs = gridspec.GridSpec(2, 2)

        pl.figure()
        pl.subplot(gs[0, :])
        pl.plot(t_ax, trace)
        pl.title(f'Traza {dataset} y espectro #{trtp_ids[idx]} archivo {filename}')
        pl.xlabel('Tiempo [s]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        pl.subplot(gs[1, 0])
        pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        pl.subplot(gs[1, 1])
        pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        pl.xlim(-25, 25)
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)
        pl.tight_layout()
        pl.savefig(f'Imgs/{dataset}/{filename}/{trtp_ids[idx]}.png')


def anim_data_spec(traces, fs, inter, dataset, filename, norm=False):
    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Create figures for trace and spectrum animations
    fig_tr = plt.figure()
    fig_sp = plt.figure()

    # List of trace and spectrum plots
    ims_tr = []
    ims_sp = []

    for trace in traces:
        # Normalize if specified
        if norm:
            trace = trace / np.max(np.abs(trace))

        im_tr = plt.plot(t_ax, trace)
        plt.title(f'Trazas dataset {dataset} archivo {filename}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Tiempo [s]')
        plt.grid(True)
        ims_tr.append(im_tr)

    for trace in traces:
        yf = sfft.fftshift(sfft.fft(trace))
        im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.title(f'Espectros dataset {dataset} archivo {filename}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Frecuencia [Hz]')
        plt.grid(True)
        ims_sp.append(im_sp)

    ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=inter, blit=True, repeat=False)
    ani_tr.save(f'Animations/{dataset}/{filename}/{dataset}_{filename}_traces.mp4')

    ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=inter, blit=True, repeat=False)
    ani_sp.save(f'Animations/{dataset}/{filename}/{dataset}_{filename}_spectrums.mp4')


def read_segy(filename):
    with segyio.open(filename, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    return fs, traces


if __name__ == '__main__':
    main()
