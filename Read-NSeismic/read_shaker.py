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
    # Create images folder
    Path("Imgs/Shaker").mkdir(parents=True, exist_ok=True)
    Path("Animations/Shaker").mkdir(parents=True, exist_ok=True)

    # 1984 trazas de 12600 muestras
    f = '../Data/Shaker/large shaker NEES_130910161319 (1).sgy'

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        # fs = segy.header[0][117] NO ES LA REAL

    # Sampling frequency
    fs = 200

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Shaker')

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 100, 'Shaker', 'Shaker')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 100, 'Shaker', 'Shaker_norm', norm=True)


def plot_traces(traces, fs, n, dataset, rand=True, pre_traces=None):
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
        plt.plot(np.squeeze(t_ax), trace)
        pl.title(f'Traza {dataset} y espectro #{trtp_ids[idx]}')
        pl.xlabel('Tiempo [s]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        pl.subplot(gs[1, 0])
        pl.plot(np.squeeze(xf), np.abs(yf) / np.max(np.abs(yf)))
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        pl.subplot(gs[1, 1])
        pl.plot(np.squeeze(xf), np.abs(yf) / np.max(np.abs(yf)))
        pl.xlim(-25, 25)
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)
        pl.tight_layout()
        pl.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}.png')


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
        plt.title(f'Trazas dataset {dataset}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Tiempo [s]')
        plt.grid(True)
        ims_tr.append(im_tr)

    for trace in traces:
        yf = sfft.fftshift(sfft.fft(trace))
        im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.title(f'Espectros dataset {dataset}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Frecuencia [Hz]')
        plt.grid(True)
        ims_sp.append(im_sp)

    ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=inter, blit=True, repeat=False)
    ani_tr.save(f'Animations/{dataset}/{filename}_traces.mp4')

    ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=inter, blit=True, repeat=False)
    ani_sp.save(f'Animations/{dataset}/{filename}_spectrums.mp4')


if __name__ == '__main__':
    main()
