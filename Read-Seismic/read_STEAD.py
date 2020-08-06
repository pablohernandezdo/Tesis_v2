import h5py
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images folder
    Path("Imgs/STEAD").mkdir(parents=True, exist_ok=True)

    # STEAD dataset path
    st = '../Data/STEAD/Train_data.hdf5'

    # Sampling frequency
    fs = 100

    # Number of traces to plot
    n = 4

    # Traces to plot
    trtp = []

    with h5py.File(st, 'r') as h5_file:

        # Seismic traces group
        grp = h5_file['earthquake']['local']

        # Traces to plot ids
        # trtp_ids = [0, 1, 2, 3]

        # Init rng
        rng = default_rng()

        # Traces to plot numbers
        trtp_ids = rng.choice(len(grp), size=n, replace=False)
        trtp_ids.sort()

        for idx, dts in enumerate(grp):
            if idx in trtp_ids:
                trtp.append(grp[dts][:, 0])

    # Sampling frequency
    fs = 100

    # Data len
    N = len(trtp[0])

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    # For trace in traces to print
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza STEAD y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlim(-25, 25)
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/STEAD/{trtp_ids[idx]}.png')


if __name__ == '__main__':
    main()
