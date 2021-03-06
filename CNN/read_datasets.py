import re
import os
import h5py
import segyio
import numpy as np
import scipy.io as sio
from numpy.random import default_rng

from obspy.io.seg2 import seg2
from obspy.io.segy.core import _read_segy

from scipy import signal
from scipy.signal import butter, lfilter, detrend


class Dsets:
    def preprocess(self, traces, fs):

        new_traces = []
        N_new = int(len(traces[0]) * 100 / fs)

        for trace in traces:

            if fs / 2 < 50:
                # Filter 50 Hz
                trace = self.butter_lowpass_filter(trace, 50, fs)

            # Detrending
            trace = detrend(trace)

            # Media cero
            trace = trace - np.mean(trace)

            # Remuestrear a 100 hz
            if fs != 100:
                trace = signal.resample(trace, N_new)

            new_traces.append(trace)

        return np.asarray(new_traces)

    @staticmethod
    def normalize(traces):
        norm_traces = []

        for trace in traces:
            if np.amax(np.abs(trace)):
                trace /=  np.amax(np.abs(trace))

            norm_traces.append(trace)

        return np.asarray(norm_traces)

    @staticmethod
    def butter_lowpass_filter(dat, highcut, fs, order=5):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, output='ba')
        y = lfilter(b, a, dat)
        return y

    @staticmethod
    def read_segy(dataset_path):
        with segyio.open(dataset_path, ignore_geometry=True) as segy:
            # Memory map, faster
            segy.mmap()

            # Traces and sampling frequency
            traces = segyio.tools.collect(segy.trace[:])
            fs = segy.header[0][117]

        return traces, fs


class DatasetFrancia(Dsets):
    def __init__(self, dataset_path):
        super().__init__()

        self.dataset_path = dataset_path
        self.traces = sio.loadmat(self.dataset_path)["StrainFilt"]
        self.fs = 100

        self.traces = self.preprocess(self.traces, self.fs)
        self.traces = self.normalize(self.traces)

    def prune_traces(self):
        pass


class DatasetNevada(Dsets):
    def __init__(self, dataset_path):
        super().__init__()

        self.dataset_path = dataset_path
        self.traces, self.fs = self.read_segy(self.dataset_path)

        # Se muere mi pc si preproceso el dataset
        self.traces = self.preprocess(self.traces, self.fs)
        self.traces = self.padd()
        self.traces = self.normalize(self.traces)

    def padd(self):

        rng = default_rng()
        n_padd = 6000 - self.traces.shape[1]

        padd_traces = []

        for trace in self.traces:
            # 30 ventanas de 100 muestras
            windows = trace.reshape(30, 100)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)


class DatasetBelgica(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

        # Belgica lo voy a dejar para el final, mucho webeo entremedio


class DatasetReykjanes(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.fs = 200
        self.n_traces = 2551
        self.header, self.traces = self.read_ascii()

        self.traces = self.preprocess(self.traces, self.fs)
        self.traces = self.padd()
        self.traces = self.normalize(self.traces)

    def padd(self):
        rng = default_rng()
        n_padd = 6000 - self.traces.shape[1]

        padd_traces = []

        for trace in self.traces:
            # 14 ventanas de 50 muestras
            windows = trace.reshape(14, 50)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)

    def read_ascii(self):
        # Preallocate
        traces = np.empty((1, self.n_traces))

        with open(self.dataset_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    header = line.strip()

                else:
                    row = np.asarray(list(
                        map(float, re.sub(' +', ' ', line).strip().split(' '))))
                    traces = np.concatenate((traces, np.expand_dims(row, 0)))

        # Delete preallocate empty row and transpose
        traces = traces[1:]
        traces = traces.transpose()

        return header, traces


class DatasetCalifornia(Dsets):
    def __init__(self, dataset_paths):
        super().__init__()

        if len(dataset_paths) != 4:
            print("Se necesitan 4 archivos!")

        else:
            self.dataset_paths = dataset_paths
            self.fs = 1000
            self.d1, self.d2, self.d3, self.d4 = self.dataset_paths
            self.traces_d1 = sio.loadmat(self.d1)['singdecmatrix'].T
            self.traces_d2 = sio.loadmat(self.d2)['singdecmatrix'].T
            self.traces_d3 = sio.loadmat(self.d3)['singdecmatrix'].T
            self.traces_d4 = sio.loadmat(self.d4)['singdecmatrix'].T

            # Preprocess datasets
            self.traces_d1 = self.preprocess(self.traces_d1, self.fs)
            self.traces_d2 = self.preprocess(self.traces_d2, self.fs)
            self.traces_d3 = self.preprocess(self.traces_d3, self.fs)
            self.traces_d4 = self.preprocess(self.traces_d4, self.fs)

            # Trim
            self.traces_d1 = self.trim(self.traces_d1)
            self.traces_d2 = self.trim(self.traces_d2)
            self.traces_d3 = self.trim(self.traces_d3)
            self.traces_d4 = self.trim(self.traces_d4)

            # Normalize
            self.traces_d1 = self.normalize(self.traces_d1)
            self.traces_d2 = self.normalize(self.traces_d2)
            self.traces_d3 = self.normalize(self.traces_d3)
            self.traces_d4 = self.normalize(self.traces_d4)

            # Stack
            self.traces = np.vstack([self.traces_d1,
                                     self.traces_d2,
                                     self.traces_d3,
                                     self.traces_d4])

    @staticmethod
    def trim(traces):
        traces = traces[:, :traces.shape[1] // 6000 * 6000]
        traces = traces.reshape((-1, 6000))
        return traces


class DatasetHydraulic(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.fs, self.traces = self.read_file()

        # Preprocesar
        self.traces = self.preprocess(self.traces, self.fs)

        # Trim
        self.traces = self.trim()

        # Normalize
        self.traces = self.normalize(self.traces)

    def trim(self):
        # Repetir última muestra para que sean 120_000
        traces = np.hstack([self.traces, self.traces[:, -1].reshape(-1, 1)])
        traces = traces.reshape(-1, 6000)
        return traces

    def read_file(self):
        with h5py.File(self.dataset_path, 'r') as f:
            traces = f['data'][()]
            fs = f['fs_f'][()].item()
        return fs, traces


class DatasetVibroseis(Dsets):
    def __init__(self, dataset_paths):
        super().__init__()

        if len(dataset_paths) != 4:
            print("Se necesitan 4 archivos!")

        else:
            self.dataset_paths = dataset_paths
            self.fs = 1000
            self.d1, self.d2, self.d3, self.d4 = self.dataset_paths
            self.traces_d1, self.fs = self.read_segy(self.d1)
            self.traces_d2, _ = self.read_segy(self.d2)
            self.traces_d3, _ = self.read_segy(self.d3)
            self.traces_d4, _ = self.read_segy(self.d4)

            # Preprocess datasets
            self.traces_d1 = self.preprocess(self.traces_d1, self.fs)
            self.traces_d2 = self.preprocess(self.traces_d2, self.fs)
            self.traces_d3 = self.preprocess(self.traces_d3, self.fs)
            self.traces_d4 = self.preprocess(self.traces_d4, self.fs)

            # Padd
            self.traces_d1 = self.padd(self.traces_d1)
            self.traces_d2 = self.padd(self.traces_d2)
            self.traces_d3 = self.padd(self.traces_d3)
            self.traces_d4 = self.padd(self.traces_d4)

            # Normalize
            self.traces_d1 = self.normalize(self.traces_d1)
            self.traces_d2 = self.normalize(self.traces_d2)
            self.traces_d3 = self.normalize(self.traces_d3)
            self.traces_d4 = self.normalize(self.traces_d4)

            # Stack
            self.traces = np.vstack([self.traces_d1,
                                     self.traces_d2,
                                     self.traces_d3,
                                     self.traces_d4])

    @staticmethod
    def padd(traces):
        rng = default_rng()
        n_padd = 6000 - traces.shape[1]

        padd_traces = []

        for trace in traces:
            # 30 ventanas de 100 muestras
            windows = trace.reshape(30, 100)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)


class DatasetShaker(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

        with segyio.open(self.dataset_path, ignore_geometry=True) as segy:
            segy.mmap()
            self.traces = segyio.tools.collect(segy.trace[:])

        self.fs = 200

        # Preprocess traces
        self.traces = self.preprocess(self.traces, self.fs)

        # Trim
        self.traces = self.trim()

        # Normalize
        self.traces = self.normalize(self.traces)

    def trim(self):
        traces = self.traces[:, :self.traces.shape[1] // 6000 * 6000]
        traces = traces.reshape(-1, 6000)
        return traces


class DatasetCoompana(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.fs = 4000

        seg2reader = seg2.SEG2()

        traces_6k = []
        traces_8k = []

        # Every data folder
        for fold in os.listdir(self.dataset_path):

            if os.path.isdir(f"{self.dataset_path}/{fold}"):

                # Read every file
                for datafile in os.listdir(f"{self.dataset_path}/{fold}"):

                    data = seg2reader.read_file(
                        f"{self.dataset_path}/{fold}/{datafile}")

                    # To ndarray
                    for wave in data:
                        # read wave data
                        trace = wave.data

                        # Hay trazas de 6000 y 8000 muestras
                        if trace.size == 6000:
                            traces_6k.append(trace)

                        else:
                            traces_8k.append(trace)

        self.traces_6k = np.asarray(traces_6k)
        self.traces_8k = np.asarray(traces_8k)

        # Padd 6k, vstack
        self.padd_6k()
        self.traces = np.vstack([self.traces_6k, self.traces_8k])

        # preprocess
        self.traces = self.preprocess(self.traces, self.fs)

        # padd
        self.traces = self.padd()

        # normalize
        self.traces = self.normalize(self.traces)

    def padd(self):

        rng = default_rng()
        n_padd = 6000 - self.traces.shape[1]

        padd_traces = []

        for trace in self.traces:
            # 20 ventanas de 10 muestras
            windows = trace.reshape(20, 10)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, n_padd)
            trace = np.hstack([trace, ns])
            padd_traces.append(trace)

        return np.asarray(padd_traces)

    def padd_6k(self):

        rng = default_rng()
        padded_6k = []

        for trace in self.traces_6k:
            # 60 ventanas de 100 muestras
            windows = trace.reshape(60, 100)

            # calcular la varianza de ventanas
            stds = np.std(windows, axis=1)

            # generar ruido y padd
            ns = rng.normal(0, np.amin(stds) / 4, 2000)
            trace = np.hstack([trace, ns])
            padded_6k.append(trace)

        self.traces_6k = np.asarray(padded_6k)


class DatasetLesser(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.fs = 250

        traces = []

        # For every file in the dataset folder
        for dataset in os.listdir(self.dataset_path):

            if dataset.split('.')[-1] == 'segy':

                # Read dataset
                data = _read_segy(f'{self.dataset_path}/{dataset}')

                # For every trace in the dataset
                for wave in data:
                    # To ndarray
                    trace = wave.data

                    # Append to traces list
                    traces.append(trace)

        self.traces = np.asarray(traces)

        # Preprocess
        self.traces = self.preprocess(self.traces, self.fs)

        # Normalize
        self.traces = self.normalize(self.traces)


class DatasetNCAirgun(Dsets):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.fs = 100

        data = _read_segy(self.dataset_path)

        traces = []

        for wave in data:
            traces.append(wave.data)

        self.traces = np.array(traces)

        # Preprocess
        self.traces = self.preprocess(self.traces, self.fs)

        # Normalize
        self.traces = self.normalize(self.traces)

