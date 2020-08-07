import pywt
import numpy as np
from numpy.random import default_rng

from scipy import signal
import matplotlib.pyplot as plt

from pathlib import Path


def main():
    # Create images folder

    Path("Imgs").mkdir(exist_ok=True)
    Path("Imgs/Noise").mkdir(exist_ok=True)
    Path("Imgs/Sin1").mkdir(exist_ok=True)
    Path("Imgs/Sin2").mkdir(exist_ok=True)
    Path("Imgs/Sin3").mkdir(exist_ok=True)
    Path("Imgs/Sin1_ns").mkdir(exist_ok=True)
    Path("Imgs/Sin2_ns").mkdir(exist_ok=True)
    Path("Imgs/Sin3_ns").mkdir(exist_ok=True)
    Path("Imgs/Sin_pad").mkdir(exist_ok=True)
    Path("Imgs/Wavelets").mkdir(exist_ok=True)

    # Init rng
    rng = default_rng()

    # Noise
    ns = gen_noise(rng)

    # Sine waves
    wvs1, wvs2, wvs3, wvs1_ns, wvs2_ns, wvs3_ns = gen_sin(rng)

    # Padded sine waves
    wvs_pad = gen_sinpad()

    # Wavelets
    wavelets = gen_wavelets()

    # Sampling frequency
    fs = 100

    # Number of traces to plot
    n = 4

    # Plot noise
    plt.figure()
    plt.plot(ns)
    plt.xlabel('Muestras [-]')
    plt.ylabel('Amplitud [-]')
    plt.title('Ruido blanco')
    plt.grid(True)
    plt.savefig(f'Imgs/Noise/noise.png')

    # Plot sine waves
    plot_traces(wvs1, fs, n, 'Sin1')
    plot_traces(wvs2, fs, n, 'Sin2')
    plot_traces(wvs3, fs, n, 'Sin3')

    plot_traces(wvs1_ns, fs, n, 'Sin1_ns')
    plot_traces(wvs2_ns, fs, n, 'Sin2_ns')
    plot_traces(wvs3_ns, fs, n, 'Sin3_ns')

    # Plot padded sine waves
    plot_traces(wvs_pad, fs, n, 'Sin_pad')

    # Plot wavelets
    plot_traces(wavelets, fs, n, 'Wavelets')


def gen_noise(rng):
    ns = rng.normal(0, 1, 6000)
    return ns


def gen_sin(rng):
    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # Time axis
    t = np.linspace(0.0, N / fs, N)

    # Number of frequency interval steps
    n = 100

    # Frequency spans
    fr1 = np.linspace(1, 50, n)
    fr2 = np.linspace(0.01, 1, n)

    # Preallocate
    wvs1 = []
    wvs2 = []
    wvs3 = []

    for f1, f2 in zip(fr1, fr2):
        sig1 = np.sin(f1 * 2.0 * np.pi * t)
        sig2 = np.sin(f2 * 2.0 * np.pi * t)
        wvs1.append(sig1)
        wvs2.append(sig2)
        wvs3.append(sig1 + sig2)

    wvs1 = np.array(wvs1)
    wvs2 = np.array(wvs2)
    wvs3 = np.array(wvs3)

    wvs1_ns = wvs1 + 0.5 * rng.normal(0, 1, wvs1.shape)
    wvs2_ns = wvs2 + 0.5 * rng.normal(0, 1, wvs2.shape)
    wvs3_ns = wvs3 + 0.5 * rng.normal(0, 1, wvs3.shape)

    return wvs1, wvs2, wvs3, wvs1_ns, wvs2_ns, wvs3_ns


def gen_sinpad():
    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # Number of intermediate sample points
    ni = [1000, 2000, 4000, 5000]

    # Number of points to zero-pad
    pad = [(N - n) // 2 for n in ni]

    # Time axis for smaller waves
    lts = [np.linspace(0.0, nis / fs, nis) for nis in ni]

    # All frequencies list
    all_fr = []

    # Calculate max period for smaller waves
    max_periods = [n_points / fs for n_points in ni]

    # Preallocate waves
    wvs_pad = []

    # Calculate frequencies for smaller waves
    for per in max_periods:
        freqs = []
        for i in range(1, int(per) + 1):
            if per % i == 0:
                freqs.append(1 / i)
        all_fr.append(freqs)

    # Generate waves and zero pad
    for idx, fr_ls in enumerate(all_fr):
        for fr in fr_ls:
            wv = np.sin(fr * 2.0 * np.pi * lts[idx])
            wv = np.pad(wv, (pad[idx], pad[idx]), 'constant')
            wvs_pad.append(wv)

    return np.asarray(wvs_pad)


def gen_wavelets():
    # Preallocate wavelets
    lets = []

    # Discrete wavelet families
    discrete_families = ['db', 'sym', 'coif', 'bior', 'rbio']

    # Obtain wavelet waveforms, resample and append
    for fam in discrete_families:
        for wavelet in pywt.wavelist(fam):
            wv = pywt.Wavelet(wavelet)
            if wv.orthogonal:
                [_, psi, _] = pywt.Wavelet(wavelet).wavefun(level=5)
                psi = signal.resample(psi, 6000)
                lets.append(psi)

    return np.asarray(lets)


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
        plt.figure()
        plt.plot(t_ax, trace)
        plt.title(f'Traza {dataset} #{trtp_ids[idx]}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [-]')
        plt.grid(True)
        plt.savefig(f'Imgs/{dataset}/{trtp_ids[idx]}.png')


if __name__ == "__main__":
    main()
