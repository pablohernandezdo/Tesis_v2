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
    Path("Imgs").mkdir(parents=True, exist_ok=True)

    # STEAD dataset path
    st = 'STEAD.hdf5'

    # Sampling frequency
    fs = 100

    # Number of traces to plot
    n = 100

    # Traces to plot
    seis_trtp = []
    nseis_trtp = []

    with h5py.File(st, 'r') as h5_file:

        # Seismic traces group
        seis_grp = h5_file['earthquake']['local']
        nseis_grp = h5_file['non_earthquake']['noise']

        # Init rng
        rng = default_rng()

        # Traces to plot numbers
        seis_traces = rng.choice(len(seis_grp), size=n, replace=False)
        seis_traces.sort()

        nseis_traces = rng.choice(len(nseis_grp), size=n, replace=False)
        nseis_traces.sort()

        for idx, dts in enumerate(seis_grp):
            if idx in seis_traces:
                seis_trtp.append(seis_grp[dts][:, 0])

        for idx, dts in enumerate(nseis_grp):
            if idx in nseis_traces:
                nseis_trtp.append(nseis_grp[dts][:, 0])

    # Data len
    N = len(seis_trtp[0])

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Figure to plot
    plt.figure()

    for idx, trace in enumerate(seis_trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(2, 1)
        plt.plot(t_ax, trace)
        plt.title(f'Traza STEAD sísmica y espectro #{seis_traces[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(2, 2)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/{nseis_traces[idx]}.png')

    for idx, trace in enumerate(nseis_trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(2, 1)
        plt.plot(t_ax, trace)
        plt.title(f'Traza STEAD no sísmica y espectro #{nseis_traces[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(2, 2)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/{nseis_traces[idx]}.png')


if __name__ == '__main__':
    main()
