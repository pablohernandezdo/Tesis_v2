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
    Path("Imgs").mkdir(parents=True, exist_ok=True)

    # Init rng
    rng = default_rng()

    # 1984 trazas de 12600 muestras
    f = 'large shaker NEES_130910161319 (1).sgy'

    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        # fs = segy.header[0][117] NO ES LA REAL

    # Sampling frequency
    fs = 200

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
            plt.title(f'Traza Shaker y espectro #{idx}')
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

if __name__ == '__main__':
    main()
