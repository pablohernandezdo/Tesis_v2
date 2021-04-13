import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import scipy.fftpack as sfft

import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Avg").mkdir(exist_ok=True)

    # Init rng
    rng = default_rng()

    # Read file
    # 4192 canales, 42000 muestra por traza
    f = sio.loadmat("mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")
    traces = f['Data_2D']

    # Sampling frequency
    fs = 10

    # Obtain 5 km average trace
    avg_trace = np.mean(traces[3500:4001, :], 0)
    avg_fil1 = butter_bandpass_filter(avg_trace, 0.5, 1, fs, order=5)
    avg_fil2 = butter_bandpass_filter(avg_trace, 0.2, 0.6, 10, order=5)
    avg_fil3 = butter_bandpass_filter(avg_trace, 0.1, 0.3, 10, order=5)

    avgs = {"avg_trace": avg_trace,
            "avg_fil1": avg_fil1,
            "avg_fil2": avg_fil2,
            "avg_fil3": avg_fil3}

    N = traces.shape[1]
    avg_trace = signal.resample(avg_trace, N * 10)

    plt.plot(avg_trace)
    plt.show()

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
    #         plt.title(f'Traza Belgica y espectro #{idx}')
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
    #
    # for trace_name in avgs:
    #     trace = avgs[trace_name]
    #     yf = sfft.fftshift(sfft.fft(trace))
    #
    #     plt.clf()
    #     plt.subplot(2, 1, 1)
    #     plt.plot(t_ax, trace)
    #     plt.title(f'Traza Belgica y espectro {trace_name}')
    #     plt.xlabel('Tiempo [s]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #
    #     plt.subplot(2, 1, 2)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'Imgs/Avg/{trace_name}.png')


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output='ba')
    return b, a


def butter_bandpass_filter(dat, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, dat)
    return y


if __name__ == '__main__':
    main()
