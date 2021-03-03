import h5py
import numpy as np
from numpy.random import default_rng

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images folder
    Path("Imgs/Tides").mkdir(parents=True, exist_ok=True)

    # 1 registro, largo 259_094_163 muestras
    file = '../Data/Tides/CSULB_T13_EarthTide_earthtide_mean_360_519.mat'

    with h5py.File(file, 'r') as f:
        trace = f['clipdata'][()]

    # Sampling frequency
    fs = 1000

    # Data len
    N = trace.shape[0]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    plt.figure()
    plt.plot(t_ax, trace)
    plt.title('Traza dataset DAS no sísmico Tides')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud normalizada[-]')
    plt.grid(True)
    plt.savefig('Imgs/Tides/Tides_trace.png')

    # # Frequency axis for FFT plot
    # xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)
    #
    # # FFT
    # yf = sfft.fftshift(sfft.fft(trace))
    #
    # # Plot
    #
    # gs = gridspec.GridSpec(2, 2)
    #
    # pl.figure()
    # pl.subplot(gs[0, :])
    # pl.plot(t_ax, trace)
    # pl.title('Traza dataset DAS no sísmico Tides')
    # pl.xlabel('Tiempo [s]')
    # pl.ylabel('Amplitud normalizada[-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 0])
    # pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 1])
    # pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    # pl.xlim(-25, 25)
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    # pl.tight_layout()
    # pl.savefig('Imgs/Tides/Tides_trace.png')


if __name__ == '__main__':
    main()
