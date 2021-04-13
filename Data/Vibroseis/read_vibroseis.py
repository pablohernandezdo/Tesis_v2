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
    Path("Imgs/048").mkdir(parents=True, exist_ok=True)
    Path("Imgs/118").mkdir(exist_ok=True)
    Path("Imgs/117").mkdir(exist_ok=True)
    Path("Imgs/047").mkdir(exist_ok=True)
    Path("Imgs/Padded").mkdir(exist_ok=True)

    # Init rng
    rng = default_rng()

    # # File 048
    # # 8700 trazas de 30000 muestras
    # f048 = 'PoroTomo_iDAS16043_160325140048.sgy'
    #
    # # Sampling frequency and data
    # fs, traces = read_segy(f048)
    #
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
    #         plt.title(f'Traza Vibroseis 048 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/048/{idx}.png')

    # File 118
    # 8700 trazas de 30000 muestras
    f118 = 'PoroTomo_iDAS16043_160325140118.sgy'

    # Sampling frequency and data
    fs, traces = read_segy(f118)

    n_trs = [1000, 2000, 3000, 4000]

    plt.figure()

    for n_tr in n_trs:
        scale = np.std(traces[n_tr][-50:])

        ns = rng.normal(0, np.abs(scale), 30000)

        padded_trace = np.hstack([traces[n_tr], ns])

        plt.clf()
        plt.plot(padded_trace)
        plt.title(f'Traza extendida Vibroseis 118 {n_tr}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Padded/{n_tr}.png')

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
    #         plt.title(f'Traza Vibroseis 118 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/118/{idx}.png')

    # # File 117
    # # 380 trazas de 30000 muestras
    # f117 = 'PoroTomo_iDAS025_160325140117.sgy'
    #
    # # Sampling frequency and data
    # fs, traces = read_segy(f117)
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
    #         plt.title(f'Traza Vibroseis 117 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/117/{idx}.png')
    #
    # # File 047
    # # 380 trazas de 30000 muestras
    # f047 = 'PoroTomo_iDAS025_160325140047.sgy'
    #
    # # Sampling frequency and data
    # fs, traces = read_segy(f047)
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
    #         plt.title(f'Traza Vibroseis 047 y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/047/{idx}.png')


def read_segy(filename):
    with segyio.open(filename, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    return fs, traces


if __name__ == '__main__':
    main()
