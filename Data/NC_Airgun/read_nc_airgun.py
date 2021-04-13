import numpy as np
from numpy.random import default_rng

import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from obspy.io.segy.core import _read_segy

from pathlib import Path


def main():
    # Create images and animations folders
    Path("Imgs").mkdir(exist_ok=True)

    # Init rng
    rng = default_rng()

    # Datos
    dataset = 'ar56.7984.mgl1408.mcs002.bbobs-a01_geo.segy'
    data = _read_segy(dataset)

    # Sampling frequency
    fs = 100

    # Number of traces to plot
    n = 100

    traces = []

    for wave in data:
        traces.append(wave.data)

    traces = np.array(traces)

    print(traces.shape )

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
    #         plt.title(f'Traza NC_Airgun y espectro #{idx}')
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


if __name__ == '__main__':
    main()