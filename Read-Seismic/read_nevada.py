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
    f = '../Data/Nevada/PoroTomo_iDAS16043_160321073721.sgy'

    # Read file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    for trace in traces:
        print(np.mean(trace))

    # # Number of traces to plot
    # n = 4
    #
    # # Traces to plot
    # trtp = []
    #
    # # Init rng
    # rng = default_rng()
    #
    # # Traces to plot numbers
    # trtp_ids = rng.choice(len(traces), size=n, replace=False)
    # trtp_ids.sort()
    #
    # # Retrieve selected traces
    # for idx, trace in enumerate(traces):
    #     if idx in trtp_ids:
    #         trtp.append(trace)
    #
    # # Data len
    # N = traces.shape[1]
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
    # # Plot n random traces with their spectrum
    # for idx, trace in enumerate(trtp):
    #     yf = sfft.fftshift(sfft.fft(trace))
    #
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(t_ax, trace)
    #     plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 721')
    #     plt.xlabel('Tiempo [s]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #
    #     plt.subplot(212)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'Imgs/Nevada/721/Nevada721_{trtp_ids[idx]}')

    # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 721')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/Nevada/721/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 721')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/Nevada/721/Spectrums.mp4')

    # File 751
    f = '../Data/Nevada/PoroTomo_iDAS16043_160321073751.sgy'

    # Read file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        traces = segyio.tools.collect(segy.trace[:])
        fs = segy.header[0][117]

    # # Number of traces to plot
    # n = 4
    #
    # # Traces to plot
    # trtp = []
    #
    # # Traces to plot numbers
    # trtp_ids = rng.choice(len(traces), size=n, replace=False)
    # trtp_ids.sort()
    #
    # # Retrieve selected traces
    # for idx, trace in enumerate(traces):
    #     if idx in trtp_ids:
    #         trtp.append(trace)
    #
    # # Data len
    # N = traces.shape[1]
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
    # # For trace in traces to print
    # for idx, trace in enumerate(trtp):
    #     yf = sfft.fftshift(sfft.fft(trace))
    #
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(t_ax, trace)
    #     plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 751')
    #     plt.xlabel('Tiempo [s]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #
    #     plt.subplot(212)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'Imgs/Nevada/751/Nevada751_{trtp_ids[idx]}')

    # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 751')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/Nevada/751/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 751')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/Nevada/751/Spectrums.mp4')

    # # File 747
    # f = '../Data/Nevada/PoroTomo_iDAS025_160321073747.sgy'
    #
    # # Read file
    # with segyio.open(f, ignore_geometry=True) as segy:
    #     segy.mmap()
    #
    #     traces = segyio.tools.collect(segy.trace[:])
    #     fs = segy.header[0][117]
    #
    # # Number of traces to plot
    # n = 4
    #
    # # Traces to plot
    # trtp = []
    #
    # # Traces to plot numbers
    # trtp_ids = rng.choice(len(traces), size=n, replace=False)
    # trtp_ids.sort()
    #
    # # Retrieve selected traces
    # for idx, trace in enumerate(traces):
    #     if idx in trtp_ids:
    #         trtp.append(trace)
    #
    # # Data len
    # N = traces.shape[1]
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
    # # For trace in traces to print
    # for idx, trace in enumerate(trtp):
    #     yf = sfft.fftshift(sfft.fft(trace))
    #
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(t_ax, trace)
    #     plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 747')
    #     plt.xlabel('Tiempo [s]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #
    #     plt.subplot(212)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'Imgs/747/Nevada747_{trtp_ids[idx]}')

    # # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 747')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/747/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 747')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/747/Spectrums.mp4')

    # File 717
    # f = '../Data/Nevada/PoroTomo_iDAS025_160321073717.sgy'
    #
    # # Read file
    # with segyio.open(f, ignore_geometry=True) as segy:
    #     segy.mmap()
    #
    #     traces = segyio.tools.collect(segy.trace[:])
    #     fs = segy.header[0][117]
    #
    # # Number of traces to plot
    # n = 4
    #
    # # Traces to plot
    # trtp = []
    #
    # # Traces to plot numbers
    # trtp_ids = rng.choice(len(traces), size=n, replace=False)
    # trtp_ids.sort()
    #
    # # Retrieve selected traces
    # for idx, trace in enumerate(traces):
    #     if idx in trtp_ids:
    #         trtp.append(trace)
    #
    # # Data len
    # N = traces.shape[1]
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
    # # For trace in traces to print
    # for idx, trace in enumerate(trtp):
    #     yf = sfft.fftshift(sfft.fft(trace))
    #
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(t_ax, trace)
    #     plt.title(f'Traza Nevada y espectro #{trtp_ids[idx]} archivo 717')
    #     plt.xlabel('Tiempo [s]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #
    #     plt.subplot(212)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.ylabel('Amplitud [-]')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'Imgs/717/Nevada717_{trtp_ids[idx]}')

    # # Data animation
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title('Trazas dataset Nevada archivo 717')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/717/Traces.mp4')
    #
    # # Spectrum animation
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title('Espectro trazas dataset Nevada archivo 717')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/717/Spectrums.mp4')


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
