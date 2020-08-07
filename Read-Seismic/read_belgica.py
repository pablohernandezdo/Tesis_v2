import h5py
import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/Belgica").mkdir(parents=True, exist_ok=True)
    Path("Animations/Belgica").mkdir(parents=True, exist_ok=True)

    # Read file
    # 4192 canales, 42000 muestra por traza
    f = sio.loadmat("../Data/Belgica/mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")
    traces = f['Data_2D']

    # Sampling frequency
    fs = 10

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot random traces
    plot_traces(traces, fs, n, 'Belgica')

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Obtain 5 km average trace
    avg_trace = np.mean(traces[3500:4001, :], 0)
    avg_fil1 = butter_bandpass_filter(avg_trace, 0.5, 1, fs, order=5)
    avg_fil2 = butter_bandpass_filter(avg_trace, 0.2, 0.6, 10, order=5)
    avg_fil3 = butter_bandpass_filter(avg_trace, 0.1, 0.3, 10, order=5)

    yf = sfft.fftshift(sfft.fft(avg_trace))
    yf_fil1 = sfft.fftshift(sfft.fft(avg_fil1))
    yf_fil2 = sfft.fftshift(sfft.fft(avg_fil2))
    yf_fil3 = sfft.fftshift(sfft.fft(avg_fil3))

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_trace)
    plt.title(f'Traza promedio Belgica y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    plt.xlim(-1.5, 1.5)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica/Belgica_avg')

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_fil1)
    plt.title(f'Traza promedio Belgica filtrada 0.5 - 1 Hz y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf_fil1) / np.max(np.abs(yf_fil1)))
    plt.xlim(-1.5, 1.5)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica/Belgica_avg_fil1')

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_fil2)
    plt.title(f'Traza promedio Belgica filtrada 0.2 - 0.6 Hz y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf_fil2) / np.max(np.abs(yf_fil2)))
    plt.xlim(-1.5, 1.5)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica/Belgica_avg_fil2')

    plt.clf()
    plt.subplot(211)
    plt.plot(t_ax, avg_fil3)
    plt.title(f'Traza promedio Belgica filtrada 0.1 - 0.3 Hz y espectro')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(xf, np.abs(yf_fil3) / np.max(np.abs(yf_fil3)))
    plt.xlim(-1.5, 1.5)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [-]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Imgs/Belgica/Belgica_avg_fil3')


def plot_traces(traces, fs, n, dataset, rand=True, pre_traces=None):
    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Traces to plot
    trtp = []

    if rand:
        # Init rng
        rng = default_rng()

        # Traces to plot numbers
        trtp_ids = rng.choice(len(traces), size=n, replace=False)
        trtp_ids.sort()

        # Retrieve selected traces
        for idx, trace in enumerate(traces):
            if idx in trtp_ids:
                trtp.append(trace)

    else:
        trtp_ids = pre_traces

        # Retrieve selected traces
        for idx, trace in enumerate(traces):
            if idx in trtp_ids:
                trtp.append(trace)

    # Plot traces in trtp with their spectrum
    for idx, trace in enumerate(trtp):
        yf = sfft.fftshift(sfft.fft(trace))

        plt.clf()
        plt.subplot(211)
        plt.plot(t_ax, trace)
        plt.title(f'Traza {dataset} y espectro #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.xlim(-1.5, 1.5)
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}.png')


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


def butter_lowpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], output='ba')
    return b, a


def butter_lowpasspass_filter(dat, lowcut, highcut, fs, order=5):
    b, a = butter_lowpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, dat)
    return y


if __name__ == '__main__':
    main()
