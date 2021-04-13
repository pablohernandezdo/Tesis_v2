import os
import numpy as np
from pathlib import Path
from numpy.random import default_rng

import scipy.fftpack as sfft
import matplotlib.pyplot as plt

from scipy import signal
from obspy.io.seg2 import seg2


def main():
    # Create images folder
    Path("Imgs").mkdir(parents=True, exist_ok=True)

    # Rng
    rng = default_rng()

    dataset_folder = "DATA"

    seg2reader = seg2.SEG2()

    # sampling frequency
    fs = 4000

    # Number of traces to plot
    n = 100

    traces = []

    # Every data folder
    for fold in os.listdir(dataset_folder):

        # Read every file
        for datafile in os.listdir(f"{dataset_folder}/{fold}"):

            data = seg2reader.read_file(
                f"{dataset_folder}/{fold}/{datafile}")

            # To ndarray
            for wave in data:
                # read wave data
                trace = wave.data

                # Hay trazas de 6000 y 8000 muestras
                if trace.size == 6000:
                    trace = np.hstack([trace, np.zeros((2000,))])

                traces.append(trace)

    traces = np.array(traces)

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
            plt.title(f'Traza Coompana y espectro #{idx}')
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
