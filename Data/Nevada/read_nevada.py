import segyio
import numpy as np
from numpy.random import default_rng

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/721").mkdir(parents=True, exist_ok=True)
    Path("Imgs/751").mkdir(exist_ok=True)
    Path("Imgs/747").mkdir(exist_ok=True)
    Path("Imgs/717").mkdir(exist_ok=True)
    Path("Imgs/Padded").mkdir(exist_ok=True)

    # Init rng
    rng = default_rng()

    # File 721
    f721 = 'PoroTomo_iDAS16043_160321073721.sgy'

    # Read file
    traces, fs = read_segy(f721)

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
    #         plt.title(f'Traza Nevada 721 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/721/{idx}.png')

    # File 751
    f751 = 'PoroTomo_iDAS16043_160321073751.sgy'

    # Read file
    traces, fs = read_segy(f751)

    print(traces.shape)

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
    #         plt.title(f'Traza Nevada 751 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/751/{idx}.png')
    #
    # n_trs = [100, 200, 300, 400]
    #
    # for n_tr in n_trs:
    #
    #     scale = np.std(traces[n_tr][-200:])
    #
    #     ns = rng.normal(0, np.abs(scale), 30000)
    #
    #     padded_trace = np.hstack([traces[n_tr], ns])
    #
    #     plt.clf()
    #     plt.plot(padded_trace)
    #     plt.title(f'Traza extendida Nevada {n_tr}')
    #     plt.xlabel('Tiempo [s]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #     plt.savefig(f'Imgs/Padded/{n_tr}.png')

    # File 747
    f747 = 'PoroTomo_iDAS025_160321073747.sgy'

    # Read file
    traces, fs = read_segy(f747)

    print(traces.shape)

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
    #         plt.title(f'Traza Nevada 747 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/747/{idx}.png')

    # File 717
    f717 = 'PoroTomo_iDAS025_160321073717.sgy'

    # Read file
    traces, fs = read_segy(f717)

    print(traces.shape)


def read_segy(filename):
    with segyio.open(filename, ignore_geometry=True) as segy:
        # Memory map, faster
        segy.mmap()

        # Traces and sampling frequency
        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    return traces, fs


if __name__ == '__main__':
    main()
