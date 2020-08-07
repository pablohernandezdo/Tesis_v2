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
    Path("Imgs/Reykjanes/Telesismo").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Reykjanes/Local1").mkdir(parents=True, exist_ok=True)
    Path("Imgs/Reykjanes/Local2").mkdir(parents=True, exist_ok=True)
    Path("Animations/Reykjanes").mkdir(parents=True, exist_ok=True)

    # Datos Utiles

    # Fig. 3fo and 3bb.
    # Comparacion entre registros de un telesismo por fibra optica y sismometro

    file_fo = '../Data/Reykjanes/Jousset_et_al_2018_003_Figure3_fo.ascii'
    file_bb = '../Data/Reykjanes/Jousset_et_al_2018_003_Figure3_bb.ascii'

    # Plot and save figures
    plot_telesismo(file_fo, file_bb, 20)

    # Fig. 5a_fo
    # Registro de sismo local con DAS

    f = '../Data/Reykjanes/Jousset_et_al_2018_003_Figure5a_fo.ascii'

    # Number of traces in file
    n_trazas = 26

    # Sampling frequency
    fs = 200

    # Number of traces to plot
    n = 4

    # Read header and traces from file
    header, traces = read_ascii(f, n_trazas)

    # Plot random traces from file
    plot_traces(traces, fs, n, 'Reykjanes/Local1')

    # Create animation of whole data
    # anim_data_spec(traces, fs, 1000, 'Reykjanes Local 1', 'Local1')

    # # Fig. 5a_gph
    # # Registro de sismo local con geofono

    f = '../Data/Reykjanes/Jousset_et_al_2018_003_Figure5a_gph.ascii'

    # Number of traces in file
    n_trazas = 26

    # Sampling frequency
    fs = 200

    # Read header and traces from file
    header, traces = read_ascii(f, n_trazas)

    # Plot random traces from file
    plot_traces(traces, fs, n, 'Reykjanes/Local1', das=False)

    # Create animation of whole data
    # anim_data_spec(traces, fs, 1000, 'Reykjanes Local 1', 'Local1_geo')

    # Fig. 5b
    # Registro de sismo local con DAS

    f = '../Data/Reykjanes/Jousset_et_al_2018_003_Figure5b.ascii'

    # Number of traces in file
    n_trazas = 2551

    # Sampling frequency
    fs = 200

    # Read header and traces from file
    header, traces = read_ascii(f, n_trazas)

    # Plot random traces
    plot_traces(traces, fs, n, 'Reykjanes/Local2')

    # Create animation of whole data
    # anim_data_spec(traces, fs, 50, 'Reykjanes Local 2', 'Local2')


def plot_telesismo(file_fo, file_bb, fs):

    data_fo = {
        'head': '',
        'strain': []
    }

    data_bb = {
        'head': '',
        'strain': []
    }

    with open(file_fo, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_fo['head'] = line.strip()
            else:
                val = line.strip()
                data_fo['strain'].append(float(val))

    with open(file_bb, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_bb['head'] = line.strip()
            else:
                val = line.strip()
                data_bb['strain'].append(float(val))

    # Data len
    N = len(data_fo['strain'])

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # FFTs
    yf_fo = sfft.fftshift(sfft.fft(data_fo['strain']))
    yf_bb = sfft.fftshift(sfft.fft(data_bb['strain']))

    gs = gridspec.GridSpec(2, 2)

    pl.figure()
    pl.subplot(gs[0, :])
    pl.plot(t_ax, data_fo['strain'])
    pl.xlabel('Tiempo [s]')
    pl.ylabel('Strain [-]')
    pl.title('Registro Reykjanes telesismo DAS')
    pl.grid(True)

    pl.subplot(gs[1, 0])
    pl.plot(xf, np.abs(yf_fo) / np.max(np.abs(yf_fo)))
    pl.xlabel('Frecuencia [-]')
    pl.ylabel('Amplitud [-]')
    pl.grid(True)

    pl.subplot(gs[1, 1])
    pl.plot(xf, np.abs(yf_fo) / np.max(np.abs(yf_fo)))
    pl.xlim(-0.5, 0.5)
    pl.xlabel('Frecuencia [-]')
    pl.ylabel('Amplitud [-]')
    pl.grid(True)
    pl.tight_layout()
    pl.savefig('Imgs/Reykjanes/Telesismo/TelesismoDAS_spec.png')

    pl.clf()
    pl.subplot(gs[0, :])
    pl.plot(t_ax, data_bb['strain'])
    pl.xlabel('Tiempo [s]')
    pl.ylabel('Strain [-]')
    pl.title('Registro Reykjanes telesismo sismómetro')
    pl.grid(True)

    pl.subplot(gs[1, 0])
    pl.plot(xf, np.abs(yf_bb) / np.max(np.abs(yf_bb)))
    pl.xlabel('Frecuencia [-]')
    pl.ylabel('Amplitud [-]')
    pl.grid(True)

    pl.subplot(gs[1, 1])
    pl.plot(xf, np.abs(yf_bb) / np.max(np.abs(yf_bb)))
    pl.xlim(-0.5, 0.5)
    pl.xlabel('Frecuencia [-]')
    pl.ylabel('Amplitud [-]')
    pl.grid(True)
    pl.tight_layout()
    pl.savefig('Imgs/Reykjanes/Telesismo/TelesismoBBS_spec.png')


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


def plot_traces(traces, fs, n, dataset, das=True, rand=True, pre_traces=None):
    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    # Traces to plot
    trtp = []

    if das:
        ins = 'DAS'
    else:
        ins = 'Geofono'

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

        pl.clf()
        pl.subplot(gs[0, :])
        pl.plot(t_ax, trace)
        pl.title(f'Traza {dataset} {ins} y espectro #{trtp_ids[idx]}')
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
        pl.xlim(-25, 25)
        pl.xlabel('Frecuencia [Hz]')
        pl.ylabel('Amplitud [-]')
        pl.grid(True)
        pl.tight_layout()
        pl.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}_{ins}.png')


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
    ani_tr.save(f'Animations/{dataset.strip().split(" ")[0]}/{filename}_traces.mp4')

    ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=inter, blit=True, repeat=False)
    ani_sp.save(f'Animations/{dataset.strip().split(" ")[0]}/{filename}_spectrums.mp4')


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
