import re
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
    # das = DASdataset('partial', 'DAS_dataset.hdf5')
    DASdataset('partial', 'DAS_dataset.hdf5')


class DASdataset:

    def __init__(self, config, hdf5_name):

        # Loading config file
        print('\nCreating DAS dataset with %s config' % config)
        print('---------------------------------------------------')
        with open('data_conf.json', 'r') as f:
            self.__cfg = json.load(f)[config]

        # Creating hdf5 structure
        self.__hdf = h5py.File(self.__cfg['output'] + hdf5_name, 'w')

        # Create groups with earthquakes and noisy traces
        g_earthquake = self.__hdf.create_group('earthquake')
        g_non_earthquake = self.__hdf.create_group('non_earthquake')

        # Create sub-groups for each dataset & load data
        for data_name in self.__cfg["datasets"]:

            print('Loading %s dataset' % data_name)
            # Creates the groups & subgroups
            dataset = self.__cfg["datasets"][data_name]
            if dataset["type"] == 'earthquake':
                subgroup = g_earthquake.create_group(data_name)
            else:
                subgroup = g_non_earthquake.create_group(data_name)

            # Load traces from dataset
            traces = self.load_data(data_name)
            for i, tr in enumerate(traces):
                subgroup.create_dataset(data_name + str(i), data=tr)

    def load_data(self, data_name):
        # Add data to each sub-group
        if data_name == 'francia':
            traces = self.get_francia()
        if data_name == 'nevada':
            traces = self.get_nevada()
        if data_name == 'belgica':
            traces = self.get_belgica()
        if data_name == 'reykjanes1':
            traces = self.get_reykjanes1()
        if data_name == 'reykjanes2':
            traces = self.get_reykjanes2()
        if data_name == 'california':
            traces = self.get_california()
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

    def get_belgica(self):

        # Load belgica dataset
        cfg = self.__cfg["datasets"]['belgica']
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
        data_fo['strain'] = signal.resample(data_fo['strain'], 6000)

        return data_fo['strain']

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

    def get_california(self):

        # Load california local dataset
        cfg = self.__cfg["datasets"]['california']
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
            new_traces.append(np.reshape(tr, (-1, 6000)))

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

    def get_shaker(net, device, thresh, hist):
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
