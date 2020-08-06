import h5py
import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from scipy.signal import butter, lfilter


def main():
    # Registro de 1 minuto de sismo M1.9 a 100 Km NE del cable, 6848 trazas de 6000 muestras
    # Create images and animations folders
    Path("Imgs/Francia").mkdir(parents=True, exist_ok=True)
    Path("Animations/Francia").mkdir(parents=True, exist_ok=True)

    # Read file and load data
    f = sio.loadmat("../Data/Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")
    traces = f["StrainFilt"]

    # sampling frequency
    fs = 100

    # Number of traces to plot
    n = 4

    # traces to plot
    trtp = [0, 3000, 3100, 4000]

    # Select test dataset traces
    test_data = []

    # Remove mean, calculate std and select
    for trace in traces:
        trace = trace - np.mean(trace)
        st = np.std(trace)

        if st > 50:
            test_data.append(trace)

    # To numpy array and remove last timeseries
    test_data = np.asarray(test_data)
    test_data = test_data[:66]

    # Plot random traces
    # plot_traces(traces, fs, n, 'Francia')

    # Plot predefined traces
    plot_traces(traces, fs, n, 'Francia', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    anim_data_spec(traces, fs, 50, 'Francia', 'Francia')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 50, 'Francia', 'Francia_norm', norm=True)

    # Animate test dataset traces
    anim_data_spec(test_data, fs, 100, 'Francia', 'Test_dataset')


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
        plt.xlim(-25, 25)
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}.png')


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


if __name__ == "__main__":
    main()
