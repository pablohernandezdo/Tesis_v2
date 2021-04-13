import h5py
import numpy as np
from scipy import signal
from numpy.random import default_rng

import scipy.fftpack as sfft

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs").mkdir(exist_ok=True)

    # 959 canales, largo 119_999 muestras
    # 500Pa10sec
    f309 = 'CSULB500Pa10secP_141210174309.mat'

    # Init rng
    rng = default_rng()

    # Sampling frequency and data
    fs, traces = read_file(f309)

    print(traces.shape)

    # # Number of traces to plot
    # n = 100
    #
    # # Pick some traces
    # plot_idx = rng.choice(len(traces), size=n, replace=False)
    #
    # # Data len
    # N = len(traces[0])
    #
    # # Time axis for signal plot
    # t_ax = np.arange(N) / fs
    #
    # # Frequency axis for FFT plot
    # xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)
    #
    # # Figure to plot
    # plt.figure()
    #
    # for idx, trace in enumerate(traces):
    #     if idx in plot_idx:
    #         yf = sfft.fftshift(sfft.fft(trace))
    #
    #         plt.clf()
    #         plt.subplot(2, 1, 1)
    #         plt.plot(t_ax, trace)
    #         plt.title(f'Traza Hydraulic y espectro #{idx}')
    #         plt.xlabel('Tiempo [s]')
    #         plt.ylabel('Amplitud [-]')
    #         plt.grid(True)
    #
    #         plt.subplot(2, 1, 2)
    #         plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #         plt.xlabel('Frecuencia [Hz]')
    #         plt.ylabel('Amplitud [-]')
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(f'Imgs/{idx}.png')


def read_file(filename):
    with h5py.File(filename, 'r') as f:
        traces = f['data'][()]
        fs = f['fs_f'][()].item()
    return fs, traces


if __name__ == '__main__':
    main()
