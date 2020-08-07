import segyio
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.fftpack as sfft
import scipy.signal as signal
from scipy.signal import butter, lfilter

from pathlib import Path


def main():
    # Create images and animations folder
    Path("Imgs/Nevada/721").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Nevada/751").mkdir(exist_ok=True)
    Path("Imgs/Nevada/747").mkdir(exist_ok=True)
    Path("Imgs/Nevada/717").mkdir(exist_ok=True)

    Path("Animations/Nevada/721").mkdir(parents=True, exist_ok=True)
    Path("Animations/Nevada/751").mkdir(exist_ok=True)
    Path("Animations/Nevada/747").mkdir(exist_ok=True)
    Path("Animations/Nevada/717").mkdir(exist_ok=True)

    # File 721
    f721 = '../Data/Nevada/PoroTomo_iDAS16043_160321073721.sgy'

    # Read file
    traces, fs = read_segy(f721)

    # Number of traces to plot
    n = 4

    # Plot interval to inspect data
    # plot_inter(traces, fs, 100)

    # Plot random traces
    plot_traces(traces, fs, n, 'Nevada', '721')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Nevada', '721', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Nevada', '721')

    # Animate all time series normalized and spectrums
    # anim_data_spec(traces, fs, 50, 'Nevada', 'Nevada_norm', norm=True)

    # File 751
    f751 = '../Data/Nevada/PoroTomo_iDAS16043_160321073751.sgy'

    # Read file
    traces, fs = read_segy(f751)

    # Plot interval to inspect data
    # plot_inter(traces, fs, 50)

    # Select test dataset traces
    tr1 = traces[50:2800]
    tr2 = traces[2900:4700]
    tr3 = traces[4800:8650]
    test_data = np.vstack((tr1, tr2, tr3))

    # Plot random traces
    plot_traces(traces, fs, n, 'Nevada', '751')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Nevada', '751', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Nevada', '751')

    # Animate all time series normalized and spectrums (Aqui hay un problema, sobreescribe el anterior)
    # anim_data_spec(traces, fs, 50, 'Nevada', '751', norm=True)

    # Animate test dataset traces (Aqui tambien hay un problema con la carpeta)
    # anim_data_spec(test_data, fs, 50, 'Nevada', '751_testdata')

    # File 747
    f747 = '../Data/Nevada/PoroTomo_iDAS025_160321073747.sgy'

    # Read file
    traces, fs = read_segy(f747)

    # Plot interval to inspect data
    # plot_inter(traces, fs, 50)

    # Plot random traces
    plot_traces(traces, fs, n, 'Nevada', '747')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Nevada', '747', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Nevada', '747')

    # Animate all time series normalized and spectrums (Aqui hay un problema, sobreescribe el anterior)
    # anim_data_spec(traces, fs, 50, 'Nevada', '747', norm=True)

    # File 717
    f717 = '../Data/Nevada/PoroTomo_iDAS025_160321073717.sgy'

    # Read file
    traces, fs = read_segy(f717)

    # Plot interval to inspect data
    # plot_inter(traces, fs, 50)

    # Plot random traces
    plot_traces(traces, fs, n, 'Nevada', '717')

    # Plot predefined traces
    # plot_traces(traces, fs, n, 'Nevada', '747', rand=False, pre_traces=trtp)

    # Animate all time series and spectrums
    # anim_data_spec(traces, fs, 50, 'Nevada', '717')

    # Animate all time series normalized and spectrums (Aqui hay un problema, sobreescribe el anterior)
    # anim_data_spec(traces, fs, 50, 'Nevada', '717', norm=True)


def plot_traces(traces, fs, n, dataset, filename, rand=True, pre_traces=None):
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
        plt.title(f'Traza {dataset} y espectro #{trtp_ids[idx]} archivo {filename}')
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
        plt.savefig(f'Imgs/{dataset}/{filename}/{trtp_ids[idx]}.png')


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
        plt.title(f'Trazas dataset {dataset} archivo {filename}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Tiempo [s]')
        plt.grid(True)
        ims_tr.append(im_tr)

    for trace in traces:
        yf = sfft.fftshift(sfft.fft(trace))
        im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
        plt.title(f'Espectros dataset {dataset} archivo {filename}')
        plt.ylabel('Amplitud [-]')
        plt.xlabel('Frecuencia [Hz]')
        plt.grid(True)
        ims_sp.append(im_sp)

    ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=inter, blit=True, repeat=False)
    ani_tr.save(f'Animations/{dataset}/{filename}/{dataset}_{filename}_traces.mp4')

    ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=inter, blit=True, repeat=False)
    ani_sp.save(f'Animations/{dataset}/{filename}/{dataset}_{filename}_spectrums.mp4')


def read_segy(filename):
    with segyio.open(filename, ignore_geometry=True) as segy:
        # Memory map, faster
        segy.mmap()

        # Traces and sampling frequency
        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    return traces, fs


def plot_inter(traces, fs, inter):
    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    for idx, trace in enumerate(traces):

        plt.figure()
        if not (idx % inter):
            # Remove mean
            trace = trace - np.mean(trace)

            plt.clf()
            plt.plot(t_ax, trace)
            plt.title(f'idx: {idx} ')
            plt.grid(True)
            plt.show(block=False)
            plt.pause(1.5)
            plt.close()


if __name__ == '__main__':
    main()
