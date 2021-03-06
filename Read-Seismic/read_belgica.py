import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import scipy.fftpack as sfft

import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

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
    # plot_traces(traces, fs, n, 'Belgica')

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    pl.figure()
    pl.plot(traces[100])
    pl.savefig("traza_100")

    # # Obtain 5 km average trace
    # avg_trace = np.mean(traces[3500:4001, :], 0)
    # avg_fil1 = butter_bandpass_filter(avg_trace, 0.5, 1, fs, order=5)
    # avg_fil2 = butter_bandpass_filter(avg_trace, 0.2, 0.6, 10, order=5)
    # avg_fil3 = butter_bandpass_filter(avg_trace, 0.1, 0.3, 10, order=5)
    #
    # yf = sfft.fftshift(sfft.fft(avg_trace))
    # yf_fil1 = sfft.fftshift(sfft.fft(avg_fil1))
    # yf_fil2 = sfft.fftshift(sfft.fft(avg_fil2))
    # yf_fil3 = sfft.fftshift(sfft.fft(avg_fil3))
    #
    # gs = gridspec.GridSpec(2, 2)
    #
    # pl.figure()
    # pl.subplot(gs[0, :])
    # pl.plot(t_ax, avg_trace)
    # pl.title(f'Traza promedio Belgica y espectro')
    # pl.xlabel('Tiempo [s]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 0])
    # pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 1])
    # pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    # pl.xlim(-1.5, 1.5)
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    # pl.tight_layout()
    # pl.savefig(f'Imgs/Belgica/Belgica_avg')
    #
    # pl.clf()
    # pl.subplot(gs[0, :])
    # pl.plot(t_ax, avg_fil1)
    # pl.title(f'Traza promedio Belgica filtrada 0.5 - 1 Hz y espectro')
    # pl.xlabel('Tiempo [s]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 0])
    # pl.plot(xf, np.abs(yf_fil1) / np.max(np.abs(yf_fil1)))
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 1])
    # pl.plot(xf, np.abs(yf_fil1) / np.max(np.abs(yf_fil1)))
    # pl.xlim(-1.5, 1.5)
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    # pl.tight_layout()
    # pl.savefig(f'Imgs/Belgica/Belgica_avg_fil1')
    #
    # pl.clf()
    # pl.subplot(gs[0, :])
    # pl.plot(t_ax, avg_fil2)
    # pl.title(f'Traza promedio Belgica filtrada 0.2 - 0.6 Hz y espectro')
    # pl.xlabel('Tiempo [s]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 0])
    # pl.plot(xf, np.abs(yf_fil2) / np.max(np.abs(yf_fil2)))
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 1])
    # pl.plot(xf, np.abs(yf_fil2) / np.max(np.abs(yf_fil2)))
    # pl.xlim(-1.5, 1.5)
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    # pl.tight_layout()
    # pl.savefig(f'Imgs/Belgica/Belgica_avg_fil2')
    #
    # pl.clf()
    # pl.subplot(gs[0, :])
    # pl.plot(t_ax, avg_fil3)
    # pl.title(f'Traza promedio Belgica filtrada 0.1 - 0.3 Hz y espectro')
    # pl.xlabel('Tiempo [s]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 0])
    # pl.plot(xf, np.abs(yf_fil3) / np.max(np.abs(yf_fil3)))
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    #
    # pl.subplot(gs[1, 1])
    # pl.plot(xf, np.abs(yf_fil3) / np.max(np.abs(yf_fil3)))
    # pl.xlim(-1.5, 1.5)
    # pl.xlabel('Frecuencia [Hz]')
    # pl.ylabel('Amplitud [-]')
    # pl.grid(True)
    # pl.tight_layout()
    # pl.savefig(f'Imgs/Belgica/Belgica_avg_fil3')


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

        gs = gridspec.GridSpec(2, 2)

        pl.figure()
        pl.subplot(gs[0, :])
        pl.plot(t_ax, trace)
        pl.title(f'Traza {dataset} y espectro #{trtp_ids[idx]}')
        pl.xlabel('Tiempo [s]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        pl.subplot(gs[1, 0])
        pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        pl.subplot(gs[1, 1])
        pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        pl.xlim(-1.5, 1.5)
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)
        pl.tight_layout()
        pl.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}.png')


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
