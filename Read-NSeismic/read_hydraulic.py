import h5py
import numpy as np
from numpy.random import default_rng

import scipy.fftpack as sfft

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/Hydraulic/500Pa10sec").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Hydraulic/500Pa100sec").mkdir(exist_ok=True)
    Path("Imgs/Hydraulic/500Pa600sec").mkdir(exist_ok=True)

    # 959 canales, largo 119_999 muestras
    # 500Pa10sec
    f309 = '../Data/Hydraulic/CSULB500Pa10secP_141210174309.mat'

    # Sampling frequency and data
    fs, traces = read_file(f309)

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Hydraulic/500Pa10sec')

    # 959 canales, largo 2_056_235 muestras
    # 500Pa100sec
    f257 = '../Data/Hydraulic/CSULB500Pa100secP_141210175257.mat'

    # Sampling frequency and data
    fs, traces = read_file(f257)

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Hydraulic/500Pa100sec')

    # 959 canales, largo 6_002_723 muestras
    # 500Pa600sec
    f813 = '../Data/Hydraulic/CSULB500Pa600secP_141210183813.mat'

    # Sampling frequency and data
    fs, traces = read_file(f813)

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Hydraulic/500Pa600sec')


def read_file(filename):
    with h5py.File(filename, 'r') as f:
        traces = f['data'][()]
        fs = f['fs_f'][()]
    return fs, traces


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


if __name__ == '__main__':
    main()
