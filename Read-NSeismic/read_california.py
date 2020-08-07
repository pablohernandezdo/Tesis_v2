import numpy as np
import scipy.io as sio
from numpy.random import default_rng

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/California").mkdir(parents=True, exist_ok=True)
    Path("Animations/California").mkdir(parents=True, exist_ok=True)

    # f = sio.loadmat('../Data/California/FSE-11_1080SecP_SingDec_StepTest (1).mat')
    # 139 trazas de 953432 muestras (196 según documentación ?)
    f = sio.loadmat('../Data/California/FSE-06_480SecP_SingDec_StepTest (1).mat')

    # Seismic traces data
    traces = f['singdecmatrix']
    traces = traces.transpose()

    # Sampling frequency
    fs = 1000

    # Number of traces to plot
    n = 4

    # Traces to plot
    # trtp = [0, 1, 2, 3]

    # Plot predefined traces
    plot_traces(traces, fs, n, 'California')

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'California', 'California')


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
        ax = pl.subplot(gs[0, 0])
        plt.plot(t_ax, trace)
        pl.title(f'Traza {dataset} y espectro #{trtp_ids[idx]}')
        pl.xlabel('Tiempo [s]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        ax = pl.subplot(gs[0, 1])
        pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)

        ax = pl.subplot(gs[1, :])
        pl.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        pl.xlim(-25, 25)
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)
        pl.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}.png')


def anim_data_spec(traces, fs, inter, dataset, filename, norm=False):
    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Create figures for trace and spectrum animations
    fig_tr = plt.figure()
    fig_sp = plt.figure()

    # List of trace and spectrum plots
    ims_tr = []
    ims_sp = []

    for trace in traces:
        # Normalize if specified
        if norm:
            trace = trace / np.max(np.abs(trace))

        im_tr = plt.plot(t_ax, trace)
        plt.title(f'Trazas dataset {dataset}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Tiempo [s]')
        plt.grid(True)
        ims_tr.append(im_tr)

    for trace in traces:
        yf = sfft.fftshift(sfft.fft(trace))
        im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.title(f'Espectros dataset {dataset}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Frecuencia [Hz]')
        plt.grid(True)
        ims_sp.append(im_sp)

    ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=inter, blit=True, repeat=False)
    ani_tr.save(f'Animations/{dataset}/{filename}_traces.mp4')

    ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=inter, blit=True, repeat=False)
    ani_sp.save(f'Animations/{dataset}/{filename}_spectrums.mp4')


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
