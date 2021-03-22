import re
import pywt
import h5py
import json
import segyio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter
from scipy import signal

from numpy.random import default_rng


def main():
    DASdataset('partial',
               'DAS_seismic.hdf5',
               'DAS_non_seismic.hdf5',
               'DAS_noise.hdf5')


class DASdataset:

    def __init__(self, config, seis_name, nseis_name, noise_name):

        # Loading config file
        print('\nCreating DAS dataset with %s config' % config)
        print('---------------------------------------------------')
        with open('data_conf_sep.json', 'r') as f:
            self.__cfg = json.load(f)[config]

        # Creating hdf5 structure
        self.__seis = h5py.File(self.__cfg['output'] + seis_name, 'w')
        self.__nseis = h5py.File(self.__cfg['output'] + nseis_name, 'w')
        self.__noise = h5py.File(self.__cfg['output'] + noise_name, 'w')

        # Create groups for seismic dataset
        g_seis = self.__seis.create_group('earthquake/local')
        _ = self.__seis.create_group('non_earthquake/noise')

        # Create groups for non seismic dataset
        _ = self.__nseis.create_group('earthquake/local')
        g_nseis = self.__nseis.create_group('non_earthquake/noise')

        # Create groups for noise dataset
        _ = self.__noise.create_group('earthquake/local')
        g_noise = self.__noise.create_group('non_earthquake/noise')

        # Create sub-groups for each dataset & load data
        # LA CLASE QUE TENGO AHORA NO FUNCIONA SI HAGO SUBGRUPOS,
        # ASI QUE POR AHORA NO VOY A HACER SUBGRUPOS
        # for data_name in self.__cfg["datasets"]:
        #
        #     print('Loading %s dataset' % data_name)
        #     # Creates the groups & subgroups
        #     dataset = self.__cfg["datasets"][data_name]
        #     if dataset["type"] == 'earthquake':
        #         subgroup = g_earthquake.create_group(data_name)
        #     else:
        #         subgroup = g_non_earthquake.create_group(data_name)
        #
        #     # Load traces from dataset
        #     traces = self.load_data(data_name)
        #
        #     for i, tr in enumerate(traces):
        #         tr = np.expand_dims(tr, 1)
        #         tr = np.hstack([tr] * 3).astype('float32')
        #         subgroup.create_dataset(data_name + str(i), data=tr)

        for data_name in self.__cfg["datasets"]:

            print('Loading %s dataset' % data_name)

            # Load dataset info from json
            dataset = self.__cfg["datasets"][data_name]

            # Load traces from dataset
            traces = self.load_data(data_name)

            if dataset["type"] == 'earthquake':
                for i, tr in enumerate(traces):
                    tr = np.expand_dims(tr, 1)
                    tr = np.hstack([tr] * 3).astype('float32')
                    g_seis.create_dataset(data_name + str(i), data=tr)

            elif dataset["type"] == 'non_earthquake':
                for i, tr in enumerate(traces):
                    tr = np.expand_dims(tr, 1)
                    tr = np.hstack([tr] * 3).astype('float32')
                    g_nseis.create_dataset(data_name + str(i), data=tr)

            elif dataset["type"] == 'noise':
                for i, tr in enumerate(traces):
                    tr = np.expand_dims(tr, 1)
                    tr = np.hstack([tr] * 3).astype('float32')
                    g_noise.create_dataset(data_name + str(i), data=tr)

            else:
                print(f'Dataset {data_name} bad type')

        # Add test signals group, get signals first
        traces = self.get_signals()

        for i, tr in enumerate(traces):
            tr = np.expand_dims(tr, 1)
            tr = np.hstack([tr] * 3).astype('float32')
            g_nseis.create_dataset('signals' + str(i), data=tr)

    @staticmethod
    def get_signals():
        # Init RNG
        rng = default_rng()

        # Noise
        ns = rng.normal(0, 1, 6000)

        # Sine waves

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

        # Prealocate
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

        # PADDED SINES

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

        # Calculate frequencies for smaller waves
        for per in max_periods:
            freqs = []
            for i in range(1, int(per) + 1):
                if per % i == 0:
                    freqs.append(1 / i)
            all_fr.append(freqs)

        # Preallocate waves
        wvs_pad = []

        # Generate waves and zero pad
        for idx, fr_ls in enumerate(all_fr):
            for fr in fr_ls:
                wv = np.sin(fr * 2.0 * np.pi * lts[idx])
                wv = np.pad(wv, (pad[idx], pad[idx]), 'constant')
                wvs_pad.append(wv)

        wvs_pad = np.array(wvs_pad)

        # Wavelets

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

        lets = np.array(lets)

        # print(f'ns: {ns.shape}\n'
        #       f'wvs1: {wvs1.shape}\n'
        #       f'wvs2: {wvs2.shape}\n'
        #       f'wvs3: {wvs3.shape}\n'
        #       f'wvs1_ns: {wvs1_ns.shape}\n'
        #       f'wvs2_ns: {wvs2_ns.shape}\n'
        #       f'wvs3_ns: {wvs3_ns.shape}\n'
        #       f'wvs_pad: {wvs_pad.shape}\n'
        #       f'lets: {lets.shape}')

        traces = np.vstack((ns,
                            wvs1,
                            wvs2,
                            wvs3,
                            wvs1_ns,
                            wvs2_ns,
                            wvs3_ns,
                            wvs_pad,
                            lets))

        return traces

    def load_data(self, data_name):
        # Add data to each sub-group
        if data_name == 'francia':
            traces = self.get_francia()
        if data_name == 'nevada':
            traces = self.get_nevada()
        if data_name == 'belgica_seis':
            traces = self.get_belgica_seis()
        if data_name == 'belgica_noise':
            traces = self.get_belgica_noise()
        if data_name == 'reykjanes1':
            traces = self.get_reykjanes1()
        if data_name == 'reykjanes2':
            traces = self.get_reykjanes2()
        if data_name == 'california1':
            traces = self.get_california1()
        if data_name == 'california2':
            traces = self.get_california2()
        if data_name == 'california3':
            traces = self.get_california3()
        if data_name == 'california4':
            traces = self.get_california4()
        if data_name == 'hydraulic1':
            traces = self.get_hydraulic1()
        if data_name == 'hydraulic2':
            traces = self.get_hydraulic2()
        if data_name == 'hydraulic3':
            traces = self.get_hydraulic3()
        if data_name == 'tides':
            traces = self.get_tides()
        if data_name == 'utah':
            traces = self.get_utah()
        if data_name == 'vibroseis1':
            traces = self.get_vibroseis1()
        if data_name == 'vibroseis2':
            traces = self.get_vibroseis2()
        if data_name == 'vibroseis3':
            traces = self.get_vibroseis3()
        if data_name == 'vibroseis4':
            traces = self.get_vibroseis4()
        if data_name == 'shaker':
            traces = self.get_shaker()
        print(data_name, '\t', traces.shape)
        return traces

    def get_francia(self):

        # Load francia dataset
        cfg = self.__cfg["datasets"]['francia']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        data = f["StrainFilt"]

        # Select some traces
        traces = []

        for trace in data:
            trace = trace - np.mean(trace)
            st = np.std(trace)

            if st > 50:
                traces.append(trace)

        traces = np.asarray(traces)
        traces = traces[:66]

        return traces

    def get_nevada(self):

        # Load Nevada data file 751
        cfg = self.__cfg["datasets"]['nevada']
        f = cfg['path'] + cfg['file']

        # For every trace in the file
        with segyio.open(f, ignore_geometry=True) as segy:
            segy.mmap()

            # Original signals
            traces = segyio.tools.collect(segy.trace[:])

        # Select dataset basico traces
        tr1 = traces[50:2800]
        tr2 = traces[2900:4700]
        tr3 = traces[4800:8650]
        traces = np.vstack((tr1, tr2, tr3))

        new_traces = []
        for trace in traces:
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_belgica_noise(self):
        # Load belgica dataset
        cfg = self.__cfg["datasets"]['belgica_noise']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        traces = f['Data_2D']

        fs = 10

        with open('das_worst.txt', 'r') as f:
            ln = f.readline()
            worst = np.asarray(list(map(int, ln.strip().split(','))))

        worst_traces = traces[worst]

        final_array = []

        for tr in worst_traces:
            tr = tr.reshape(-1, 6000)
            final_array.append(tr)

        final_array = np.asarray([final_array])

        return final_array.reshape(-1, 6000)

    def get_belgica_seis(self):

        # Load belgica dataset
        cfg = self.__cfg["datasets"]['belgica_seis']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        traces = f['Data_2D']

        fs = 10

        # Predict average 5km of measurements
        avg_data = np.mean(traces[3500:4001, :], 0)

        avg_fil1 = butter_bandpass_filter(avg_data, 0.5, 1, fs, order=5)
        avg_fil2 = butter_bandpass_filter(avg_data, 0.2, 0.6, 10, order=5)
        avg_fil3 = butter_bandpass_filter(avg_data, 0.1, 0.3, 10, order=5)

        traces = np.vstack((avg_data, avg_fil1, avg_fil2, avg_fil3))

        new_traces = []
        for trace in traces:
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_reykjanes1(self):

        # Load reykjanes telesismo dataset
        cfg = self.__cfg["datasets"]['reykjanes1']
        file_fo = cfg['path'] + cfg['file']

        # Dict for header and data
        data_fo = {
            'head': '',
            'strain': []
        }

        # Read fo file and save content to data_fo
        with open(file_fo, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data_fo['head'] = line.strip()
                else:
                    val = line.strip()
                    data_fo['strain'].append(float(val))

        # Resample
        trace = np.array(data_fo['strain'])
        trace = signal.resample(trace, 6000)

        # En este archivo hay solo una traza, hay que agregar una dimension
        # para que la funcion enumerate en el init de la clase no recorra las
        # muestras y obtenga la traza completa

        trace = np.expand_dims(trace, 0)

        return trace

    def get_reykjanes2(self):

        cfg = self.__cfg["datasets"]['reykjanes2']
        # Load reykjanes local dataset
        file_fo = cfg['path'] + cfg['file']

        # Rng
        rng = default_rng()

        data = {
            'head': '',
            'strain': np.empty((1, 2551))
        }

        with open(file_fo, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data['head'] = line.strip()
                else:
                    row = np.asarray(list(
                        map(float, re.sub(' +', ' ', line).strip().split(' '))))
                    data['strain'] = np.concatenate(
                        (data['strain'], np.expand_dims(row, 0)))

        data['strain'] = data['strain'][1:]
        traces = data['strain'].transpose()

        # Number of input samples to model
        final_samples = 6000

        # For every trace in the file
        new_traces = []
        for i, trace in enumerate(traces):
            # Resample
            trace = signal.resample(trace, 700)

            # Random place to put signal in
            idx = rng.choice(final_samples - len(trace), size=1)

            # Number of samples to zero pad on the right side
            right_pad = final_samples - idx - len(trace)

            # Zero pad signal
            new_traces.append(np.pad(trace,
                                     (idx.item(), right_pad.item()),
                                     mode='constant'))

        return np.array(new_traces)

    def get_california1(self):

        # Load california local dataset
        cfg = self.__cfg["datasets"]['california1']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        traces = f['singdecmatrix']
        traces = traces.transpose()

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 62125)
            tr = tr[:60000]
            tr = np.reshape(tr, (10, 6000))
            for k in range(0, 10):
                new_traces.append(tr[k, :])

        return np.array(new_traces)

    def get_california2(self):

        # Load california local dataset
        cfg = self.__cfg["datasets"]['california2']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        traces = f['singdecmatrix']
        traces = traces.transpose()

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 21736)
            tr = tr[:18000]
            tr = np.reshape(tr, (3, 6000))
            for k in range(0, 3):
                new_traces.append(tr[k, :])

        return np.array(new_traces)

    def get_california3(self):

        # Load california local dataset
        cfg = self.__cfg["datasets"]['california3']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        traces = f['singdecmatrix']
        traces = traces.transpose()

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 41228)
            tr = tr[:36000]
            tr = np.reshape(tr, (6, 6000))
            for k in range(0, 6):
                new_traces.append(tr[k, :])

        return np.array(new_traces)

    def get_california4(self):

        # Load california local dataset
        cfg = self.__cfg["datasets"]['california4']
        f = scipy.io.loadmat(cfg['path'] + cfg['file'])

        # Read data
        traces = f['singdecmatrix']
        traces = traces.transpose()

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 95343)
            tr = tr[:90000]
            tr = np.reshape(tr, (15, 6000))
            for k in range(0, 15):
                new_traces.append(tr[k, :])

        return np.array(new_traces)

    def get_hydraulic1(self):

        cfg = self.__cfg["datasets"]['hydraulic1']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read file data
        with h5py.File(file_fo, 'r') as f:
            traces = f['data'][()]

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 12000)

            # Reshape
            for trace in np.reshape(tr, (-1, 6000)):
                new_traces.append(trace)

        return np.array(new_traces)

    def get_hydraulic2(self):

        cfg = self.__cfg["datasets"]['hydraulic2']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read file data
        with h5py.File(file_fo, 'r') as f:
            traces = f['data'][()]

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 205623)

            # Discard extra samples
            tr = tr[:(6000 * 34)]

            # Reshape
            new_traces.append(np.reshape(tr, (-1, 6000)))

        return np.array(new_traces)

    def get_hydraulic3(self):

        cfg = self.__cfg["datasets"]['hydraulic1']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read file data
        with h5py.File(file_fo, 'r') as f:
            traces = f['data'][()]

        # For every trace in the file
        new_traces = []
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 600272)

            # Discard extra samples
            tr = tr[:600000]

            # Reshape
            new_trace.append(np.reshape(tr, (-1, 6000)))

        return np.array(new_trace)

    def get_tides(self):

        cfg = self.__cfg["datasets"]['tides']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read file data
        with h5py.File(file_fo, 'r') as f:
            traces = f['clipdata'][()]

        # Resample to 100 Hz
        traces = signal.resample(traces, 25909416)

        # Discard extra samples
        traces = traces[:25908000]

        # Reshape to matrix of traces
        traces = traces.reshape(-1, 6000)

        return traces

    def get_utah(self):

        cfg = self.__cfg["datasets"]['utah']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read data
        with segyio.open(file_fo, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

        new_traces = []
        for trace in traces:
            # Resample
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_vibroseis1(self):

        cfg = self.__cfg["datasets"]['vibroseis1']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read data
        with segyio.open(file_fo, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

        # For every trace in the file
        new_traces = []
        for trace in traces:
            # Resample
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_vibroseis2(self):

        cfg = self.__cfg["datasets"]['vibroseis1']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read data
        with segyio.open(file_fo, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

        # For every trace in the file
        new_traces = []
        for trace in traces:
            # Resample
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_vibroseis3(self):

        cfg = self.__cfg["datasets"]['vibroseis1']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read data
        with segyio.open(file_fo, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

        # For every trace in the file
        new_traces = []
        for trace in traces:
            # Resample
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_vibroseis4(self):

        cfg = self.__cfg["datasets"]['vibroseis1']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read data
        with segyio.open(file_fo, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

        # For every trace in the file
        new_traces = []
        for trace in traces:
            # Resample
            new_traces.append(signal.resample(trace, 6000))

        return np.array(new_traces)

    def get_shaker(self):
        cfg = self.__cfg["datasets"]['shaker']
        # Load hydraulic local dataset
        file_fo = cfg['path'] + cfg['file']

        # Read data
        with segyio.open(file_fo, ignore_geometry=True) as segy:
            segy.mmap()

            # Traces
            traces = segyio.tools.collect(segy.trace[:])

        # For every trace in the file
        new_traces = []
        for trace in traces:
            # Resample
            trace = signal.resample(trace, 6300)

            # Discard last 300 samples
            new_traces.append(trace[:6000])

        return np.array(new_traces)


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


if __name__ == "__main__":
    main()
