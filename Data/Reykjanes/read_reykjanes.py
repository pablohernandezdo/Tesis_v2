
import re
import numpy as np
from numpy.random import default_rng

import scipy.fftpack as sfft
from scipy.signal import butter, lfilter

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/Telesismo").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Local").mkdir(exist_ok=True)
    Path("Imgs/Padded").mkdir(exist_ok=True)

    # Init rng
    rng = default_rng()

    # Datos Utiles

    # Fig. 3fo and 3bb.
    # Comparacion entre registros de un telesismo por fibra optica y sismometro

    # tele_file = 'Jousset_et_al_2018_003_Figure3_fo.ascii'
    #
    # fs = 20
    #
    # data_fo = {
    #     'head': '',
    #     'strain': []
    # }
    #
    # with open(tele_file, 'r') as f:
    #     for idx, line in enumerate(f):
    #         if idx == 0:
    #             data_fo['head'] = line.strip()
    #         else:
    #             val = line.strip()
    #             data_fo['strain'].append(float(val))
    #
    # # Data len
    # N = len(data_fo['strain'])
    #
    # # Time axis for signal plot
    # t_ax = np.arange(N) / fs
    #
    # # Frequency axis for FFT plot
    # xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)
    #
    # # FFTs
    # yf_fo = sfft.fftshift(sfft.fft(data_fo['strain']))
    #
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(t_ax, data_fo['strain'])
    # plt.title(f'Traza telesismo y espectro')
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud [-]')
    # plt.grid(True)
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(xf, np.abs(yf_fo) / np.max(np.abs(yf_fo)))
    # plt.xlabel('Frecuencia [Hz]')
    # plt.ylabel('Amplitud [-]')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'Imgs/Telesismo/Telesismo.png')

    local = 'Jousset_et_al_2018_003_Figure5b.ascii'

    # Sampling frequency
    fs = 200

    # Number of traces in file
    n_trazas = 2551

    # Read header and traces from file
    header, traces = read_ascii(local, n_trazas)

    n_trs = [100, 200, 300, 400]

    plt.figure()

    for n_tr in n_trs:
        scale = np.std(traces[n_tr][-20:])

        ns = rng.normal(0, np.abs(scale), 6000 - len(traces[0]))

        padded_trace = np.hstack([traces[n_tr], ns])

        plt.clf()
        plt.plot(padded_trace)
        plt.title(f'Traza extendida Reykjanes {n_tr}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/Padded/{n_tr}.png')

    # # Numero de trazas a graficar
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
    #         plt.title(f'Traza Reykjanes local y espectro #{idx}')
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
    #         plt.savefig(f'Imgs/Local/{idx}.png')


def read_ascii(filename, n_trazas):
    # Preallocate
    traces = np.empty((1, n_trazas))

    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                header = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                traces = np.concatenate((traces, np.expand_dims(row, 0)))

    # Delete preallocate empty row and transpose
    traces = traces[1:]
    traces = traces.transpose()

    return header, traces


if __name__ == "__main__":
    main()
