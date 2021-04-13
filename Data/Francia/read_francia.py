import h5py
import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from pathlib import Path
from scipy.signal import butter, lfilter


def main():
    # Registro de 1 minuto de sismo M1.9 a 100 Km NE del cable, 6848 trazas de 6000 muestras
    # Create images and animations folders
    Path("Imgs").mkdir(parents=True, exist_ok=True)

    # Init rng
    rng = default_rng()

    # Read file and load data
    f = sio.loadmat("Earthquake_1p9_Var_BP_2p5_15Hz.mat")
    traces = f["StrainFilt"]

    # sampling frequency
    fs = 100

    # Number of traces to plot
    n = 100

    # Pick some traces
    plot_idx = rng.choice(len(traces), size=n, replace=False)

    # Data len
    N = len(traces[0])

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    for idx, trace in enumerate(traces):
        if idx in plot_idx:
            yf = sfft.fftshift(sfft.fft(trace))

            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(t_ax, trace)
            plt.title(f'Traza Francia sísmica y espectro #{idx}')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud [-]')
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
            plt.xlabel('Frecuencia [Hz]')
            plt.ylabel('Amplitud [-]')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'Imgs/{idx}.png')


if __name__ == "__main__":
    main()
