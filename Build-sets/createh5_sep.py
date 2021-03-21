import os
import h5py
import argparse
import numpy as np
from numpy.random import default_rng

from obspy.io.seg2 import seg2
from obspy.io.segy.core import _read_segy

from scipy import signal


def main():
    # Create train, validati¿n and test sets from STEAD
    # and non seismic geophone dataset files

    # Args
    parser = argparse.ArgumentParser(description='Dataset creation parameters')
    parser.add_argument('--source_file', default='../Data/STEAD/STEAD.hdf5',
                        help='Source HDF5 file path')
    parser.add_argument('--train_file', default='Train_data_v3.hdf5',
                        help='Output train HDF5 file path')
    parser.add_argument('--val_file', default='Validation_data_v3.hdf5',
                        help='Output validation HDF5 file path')
    parser.add_argument('--test_file', default='Test_data_v2.hdf5',
                        help='Output test HDF5 file path')
    parser.add_argument('--train_traces', type=int, default=52500,
                        help='Number of training seismic traces to copy')
    parser.add_argument('--train_noise', type=int, default=26250,
                        help='Number of training noise traces to copy')
    parser.add_argument('--val_traces', type=int, default=7500,
                        help='Number of validation seismic traces to copy')
    parser.add_argument('--val_noise', type=int, default=3750,
                        help='Number of validation noise traces to copy')
    parser.add_argument('--test_traces', type=int, default=10000,
                        help='Number of test seismic traces to copy')
    parser.add_argument('--test_noise', type=int, default=10000,
                        help='Number of test noise traces to copy')
    args = parser.parse_args()

    # Init rng
    rng = default_rng()

    with open('fault.txt', 'r') as f:
        ln = f.readline()
        faulty = np.asarray(list(map(int, ln.strip().split(','))))

    # Load geophone data
    coompana_traces = read_coompana()
    lesser_traces = read_lesser_antilles_airgun()
    nc_traces = read_north_carolina_airgun()

    # NUMERO DE TRAZAS DE ENTRENAMIENTO, VALIDACION Y PRUEBA GEO
    n_geo_train = 8750
    n_geo_val = 1250
    n_geo_test = 10000

    # Split geophone data
    coompana_traces_train = coompana_traces[:n_geo_train]
    coompana_traces_val = coompana_traces[n_geo_train:n_geo_train + n_geo_val]
    coompana_traces_test = coompana_traces[n_geo_train + n_geo_val:]

    lesser_traces_train = lesser_traces[:n_geo_train]
    lesser_traces_val = lesser_traces[n_geo_train:n_geo_train + n_geo_val]
    lesser_traces_test = lesser_traces[n_geo_train + n_geo_val:
                                       n_geo_train + n_geo_val + n_geo_test]

    nc_traces_train = nc_traces[:n_geo_train]
    nc_traces_val = nc_traces[n_geo_train:n_geo_train + n_geo_val]
    nc_traces_test = nc_traces[n_geo_train + n_geo_val:
                               n_geo_train + n_geo_val + n_geo_test]

    # Read the hdf5 source file
    with h5py.File(args.source_file, 'r') as source:

        # Retrieve file groups
        src_seis = source['earthquake']['local']
        src_ns = source['non_earthquake']['noise']

        # Total number of traces to copy
        seis2copy = args.train_traces + args.val_traces + args.test_traces
        ns2copy = args.train_noise + args.val_noise + args.test_noise

        # Indexes of traces to copy
        seismic_ids = rng.choice(len(src_seis), size=seis2copy, replace=False)
        noise_ids = rng.choice(len(src_ns), size=ns2copy, replace=False)

        # Check faulty datasets selected
        for val in faulty:

            # If faulty selected
            if val in seismic_ids:

                # Delete from array
                idx = np.argwhere(seismic_ids == val)
                seismic_ids = np.delete(seismic_ids, idx)

                # Select new one
                new_val = rng.choice(len(src_seis), size=1)

                # Check if is already in array
                while new_val in seismic_ids:
                    new_val = rng.choice(len(src_seis), size=1)

                # Append to array
                seismic_ids = np.append(seismic_ids, new_val)

        # Indexes of traces to copy to train dataset
        train_seis_ids = seismic_ids[:args.train_traces]
        train_noise_ids = noise_ids[:args.train_noise]

        # Indexes of traces to copy to validation dataset
        val_seis_ids = seismic_ids[args.train_traces:args.train_traces + args.val_traces]
        val_noise_ids = noise_ids[args.train_noise:args.train_noise + args.val_noise]

        # Indexes of traces to copy to test dataset
        test_seis_ids = seismic_ids[args.train_traces + args.val_traces:args.train_traces + args.val_traces+args.test_traces]
        test_noise_ids = noise_ids[args.train_noise + args.val_noise:args.train_noise + args.val_noise+args.test_noise]

        # ARMAR DATASET DE ENTRENAMIENTO Y VALIDACION
        with h5py.File('../Data/STEAD/' + args.train_file, 'w') as train_dst, \
                h5py.File('../Data/STEAD/' + args.val_file, 'w') as val_dst:

            # Create new train file groups
            train_dst_wv = train_dst.create_group('earthquake/local')
            train_dst_ns = train_dst.create_group('non_earthquake/noise')

            # Create new val file groups
            val_dst_wv = val_dst.create_group('earthquake/local')
            val_dst_ns = val_dst.create_group('non_earthquake/noise')

            # For every dataset in source seismic group
            for idx, dset in enumerate(src_seis):

                if idx in train_seis_ids:
                    # Retrieve dataset object
                    data = src_seis[dset]

                    # Copy seismic trace to new train file
                    train_dst_wv.copy(data, dset)

                if idx in val_seis_ids:
                    # Retrieve dataset object
                    data = src_seis[dset]

                    # Copy seismic trace to new train file
                    val_dst_wv.copy(data, dset)

            # For every dataset in source noise group
            for idx, dset in enumerate(src_ns):

                if idx in train_noise_ids:
                    # Retrieve dataset object
                    data = src_ns[dset]

                    # Copy noise trace to new noise file
                    train_dst_ns.copy(data, dset)

                if idx in val_noise_ids:
                    # Retrieve dataset object
                    data = src_ns[dset]

                    # Copy seismic trace to new train file
                    val_dst_ns.copy(data, dset)

            # AGREGAR SEÑALES DE GEOFONOS
            # Coompana
            add_traces2group('Coompana', coompana_traces_train, train_dst_ns)
            add_traces2group('Coompana', coompana_traces_val, val_dst_ns)

            # Lesser
            add_traces2group('Lesser', lesser_traces_train, train_dst_ns)
            add_traces2group('Lesser', lesser_traces_val, val_dst_ns)

            # NC
            add_traces2group('NC', nc_traces_train, train_dst_ns)
            add_traces2group('NC', nc_traces_val, val_dst_ns)

        # ARMAR DATASETS DE PRUEBA
        with h5py.File('../Data/STEAD/STEAD_Seis_TEST.hdf5', 'w') as test_dst:

            # Create new test file groups
            test_dst_wv = test_dst.create_group('earthquake/local')
            test_dst_ns = test_dst.create_group('non_earthquake/noise')

            # For every dataset in source seismic group
            for idx, dset in enumerate(src_seis):

                if idx in test_seis_ids:
                    # Retrieve dataset object
                    data = src_seis[dset]

                    # Copy seismic trace to new train file
                    test_dst_wv.copy(data, dset)

        with h5py.File('../Data/STEAD/STEAD_NSeis_TEST.hdf5', 'w') as test_dst:

            # Create new test file groups
            test_dst_wv = test_dst.create_group('earthquake/local')
            test_dst_ns = test_dst.create_group('non_earthquake/noise')

            # For every dataset in source noise group
            for idx, dset in enumerate(src_ns):

                if idx in test_noise_ids:
                    # Retrieve dataset object
                    data = src_ns[dset]

                    # Copy seismic trace to new train file
                    test_dst_ns.copy(data, dset)

        with h5py.File('../Data/STEAD/GEO_TEST.hdf5', 'w') as test_dst:

            # Create new test file groups
            test_dst_wv = test_dst.create_group('earthquake/local')
            test_dst_ns = test_dst.create_group('non_earthquake/noise')

            # Coompana
            add_traces2group('Coompana', coompana_traces_test, test_dst_ns)

            # Lesser
            add_traces2group('Lesser', lesser_traces_test, test_dst_ns)

            # NC
            add_traces2group('NC', nc_traces_test, test_dst_ns)


def add_traces2group(data_name, traces, group):
    for i, tr in enumerate(traces):
        tr = np.expand_dims(tr, 1)
        tr = np.hstack([tr] * 3).astype('float32')
        group.create_dataset(data_name + str(i), data=tr)


def read_coompana():
    # Rng
    rng = default_rng()

    dataset_folder = "../Data/Coompana"

    seg2reader = seg2.SEG2()

    fs = 4000

    traces = []

    # Every data folder
    for fold in os.listdir(dataset_folder):

        # Read every file
        for datafile in os.listdir(f"{dataset_folder}/{fold}"):

            data = seg2reader.read_file(
                f"{dataset_folder}/{fold}/{datafile}")

            # To ndarray
            for wave in data:
                # read wave data
                trace = wave.data
                trace = trace - np.mean(trace)

                # Estimate noise power
                ns_pwr = np.sqrt(np.var(trace[-200:]) / 200)

                # resample to 100 Hz
                new_samples = int(len(trace) * 100 / fs)
                trace = signal.resample(trace, new_samples)

                # how many noise samples needed
                ns_samples = 6000 - new_samples

                # generate noise
                ns = rng.normal(0, ns_pwr, ns_samples)

                # add front and trailing noise
                trace = np.hstack([ns[:(len(ns) // 2)],
                                   trace,
                                   ns[-(len(ns) // 2):]])

                traces.append(trace)

    traces = np.array(traces)

    print(f"Coompana traces shape: {traces.shape}")

    return traces

    # plot_single_trace(traces, fs,
    #                   'Coompana dataset sample',
    #                   'Time [s]',
    #                   'Amplitude [-]',
    #                   save=True,
    #                   figname='Coompana sample')


def read_lesser_antilles_airgun():
    dataset_folder = "../Data/Lesser_Antilles"

    # Sampling frequency pre-read from file
    fs = 250

    # Preallocate
    traces = []

    # For every file in the dataset folder
    for dataset in os.listdir(dataset_folder):

        # Read dataset
        data = _read_segy(f'{dataset_folder}/{dataset}')

        # For every trace in the dataset
        for wave in data:
            # To ndarray
            trace = wave.data

            # New amount of samples to resample
            new_samples = int(len(trace) * 100 / fs)

            # New samples = 6000 so no need to do anything else
            trace = signal.resample(trace, new_samples)

            # Append to traces list
            traces.append(trace)

    traces = np.array(traces)

    print(f"Lesser Antilles traces shape: {traces.shape}")

    return traces

    # plot_single_trace(traces, fs,
    #                   'Lesser Antilles dataset sample',
    #                   'Time [s]',
    #                   'Amplitude [-]',
    #                   save=True,
    #                   figname='Lesser Antilles sample')


def read_north_carolina_airgun():
    dataset = "../Data/NC_Airgun/ar56.7984.mgl1408.mcs002.bbobs-a01_geo.segy"

    data = _read_segy(dataset)

    fs = 100

    traces = []

    for wave in data:
        traces.append(wave.data)

    traces = np.array(traces)

    print(f"North Carolina Airgun traces shape: {traces.shape}")

    return traces

    # plot_single_trace(traces, fs,
    #                   'North Carolina airgun dataset sample',
    #                   'Time [s]',
    #                   'Amplitude [-]',
    #                   save=True,
    #                   figname='North Carolina airgun sample')


if __name__ == '__main__':
    main()
