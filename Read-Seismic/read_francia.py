import h5py
import numpy as np

import scipy.io as sio
import scipy.fftpack as sfft
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from scipy.signal import butter, lfilter


def main():
    # Create images and animations folder

    Path("Imgs/Francia").mkdir(parents=True, exist_ok=True)
    Path("Animations/Francia").mkdir(parents=True, exist_ok=True)

    # Registro de 1 minuto de sismo M1.9 a 100 Km NE del cable
    # 6848 trazas de 6000 muestras
    f = sio.loadmat("../Data/Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    traces = f["StrainFilt"]

    # sampling frequency
    fs = 100

    # traces to print
    trtp = [0, 3000, 3100, 4000]

    # Data len
    N = traces.shape[1]

    # Time axis for signal plot
    t_ax = np.arange(N) / fs

    # Frequency axis for FFT plot
    xf = np.linspace(-fs / 2.0, fs / 2.0 - 1 / fs, N)

    stds = []
    n_st = 0

    for trace in traces:
        trace = trace-np.mean(trace)
        maxx = np.max(np.abs(trace))
        st = np.std(trace)
        rms = np.sqrt(np.sum(np.power(trace, 2)) / len(trace))

        if st > 50:
            n_st += 1
            plt.figure()
            plt.plot(trace)
            plt.title(f'tr #{n_st}, std = {st:5.3f}, max/std = {maxx/st:5.3f}, rms = {rms:5.3f}')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
        stds.append(np.std(trace))

    print(n_st)

    # _ = plt.hist(stds, bins='auto')
    # plt.show()


    # # Create figure for plotting
    # plt.figure()

    # # plot chosen traces
    # for i in trtp:
    #     yf = sfft.fftshift(sfft.fft(traces[i]))
    #
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(t_ax, traces[i])
    #     plt.grid(True)
    #     plt.ylabel('Strain [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.title(f'Serie de tiempo y espectro #{i} Francia')
    #
    #     plt.subplot(212)
    #     plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.grid(True)
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.tight_layout()
    #     plt.savefig(f'Imgs/Francia/Francia_trace_{i}.png')
    #
    # # Create animation of whole data
    # fig_tr = plt.figure()
    # ims_tr = []
    #
    # for trace in traces:
    #     im_tr = plt.plot(t_ax, trace)
    #     plt.title(f'Trazas dataset Francia')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Tiempo [s]')
    #     plt.grid(True)
    #     ims_tr.append(im_tr)
    #
    # ani_tr = animation.ArtistAnimation(fig_tr, ims_tr, interval=50, blit=True, repeat=False)
    # ani_tr.save('Animations/Francia/Francia_traces.mp4')
    #
    # # Create animation of whole data spectrums
    # fig_sp = plt.figure()
    # ims_sp = []
    #
    # for trace in traces:
    #     yf = sfft.fftshift(sfft.fft(trace))
    #     im_sp = plt.plot(xf, np.abs(yf) / np.max(np.abs(yf)))
    #     plt.title(f'Espectro trazas dataset Francia')
    #     plt.ylabel('Amplitud [-]')
    #     plt.xlabel('Frecuencia [Hz]')
    #     plt.grid(True)
    #     ims_sp.append(im_sp)
    #
    # ani_sp = animation.ArtistAnimation(fig_sp, ims_sp, interval=50, blit=True, repeat=False)
    # ani_sp.save('Animations/Francia/Francia_spectrums.mp4')


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
