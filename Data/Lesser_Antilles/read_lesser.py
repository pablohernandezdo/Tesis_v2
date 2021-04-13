import os
import numpy as np
from numpy.random import default_rng

from scipy import signal
import scipy.fftpack as sfft
from obspy.io.segy.core import _read_segy

from pathlib import Path
import matplotlib.pyplot as plt


def main():
    # Create images folder
    Path("Imgs").mkdir(parents=True, exist_ok=True)

    dataset_folder = 'JC149'

    # Init rng
    rng = default_rng()

    # Sampling frequency pre-read from file
    fs = 250

    # Number of traces to plot
    n = 100

    # Preallocate
    traces = []

    # For every file in the dataset folder
    for dataset in os.listdir(dataset_folder):

        # Read dataset
        data = _read_segy(f'{dataset_folder}/{dataset}')

        # For every trace in the dataset
        for wave in data:
            # To ndarray
            trace = wave.data

            # Append to traces list
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
            plt.title(f'Traza Lesser y espectro #{idx}')
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

    print(traces.shape)


if __name__ == '__main__':
    main()