import os
import argparse

import h5py
import tqdm
import segyio
import numpy as np
import pandas as pd

from numpy.random import default_rng

from obspy.io.seg2 import seg2
from obspy.io.segy.core import _read_segy

from scipy import signal


def main():
    # Create new train, validation and test sets from STEAD
    # and non seismic geophone dataset files

    # Args
    parser = argparse.ArgumentParser(description='Dataset creation parameters')

    parser.add_argument('--source_file', default='../Data/STEAD/STEAD.hdf5',
                        help='Source HDF5 file path')
    parser.add_argument('--train_file', default='Train_data_v2.hdf5',
                        help='Output train HDF5 file path')
    parser.add_argument('--val_file', default='Validation_data_v2.hdf5',
                        help='Output validation HDF5 file path')
    parser.add_argument('--test_file', default='Test_data_v2.hdf5',
                        help='Output test HDF5 file path')
    parser.add_argument('--train_traces', type=int, default=6000,
                        help='Number of training seismic traces to copy')
    parser.add_argument('--train_noise', type=int, default=6000,
                        help='Number of training noise traces to copy')
    parser.add_argument('--val_traces', type=int, default=2000,
                        help='Number of validation seismic traces to copy')
    parser.add_argument('--val_noise', type=int, default=2000,
                        help='Number of validation noise traces to copy')
    parser.add_argument('--test_traces', type=int, default=2000,
                        help='Number of test seismic traces to copy')
    parser.add_argument('--test_noise', type=int, default=2000,
                        help='Number of test noise traces to copy')
    args = parser.parse_args()

    # Init rng
    rng = default_rng()

    with open('fault.txt', 'r') as f:
        ln = f.readline()
        faulty = np.asarray(list(map(int, ln.strip().split(','))))

    # Read the hdf5 source file
    with h5py.File(args.source_file, 'r') as source:

        # Retrieve file groups
        src_seis = source['earthquake']['local']
        src_ns = source['non_earthquake']['noise']

        # Total number of traces to copy
        seis2copy = args.train_traces + args.val_traces + args.test_traces
        ns2copy = args.train_noise + args.val_noise + args.test_noise

        # Traces to copy
        seismic_ids = rng.choice(len(src_seis),
                                 size=seis2copy, replace=False)

        noise_ids = rng.choice(len(src_ns),
                               size=ns2copy, replace=False)

        # Remove selected faulty traces
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

        train_seis_ids = seismic_ids[:args.train_traces]
        train_noise_ids = noise_ids[:args.train_noise]

        val_seis_ids = seismic_ids[args.train_traces:
                                   args.train_traces + args.val_traces]

        val_noise_ids = noise_ids[args.train_noise:
                                  args.train_noise + args.val_noise]

        test_seis_ids = seismic_ids[args.train_traces + args.val_traces:
                                    args.train_traces + args.val_traces +
                                    args.test_traces]

        test_noise_ids = noise_ids[args.train_noise + args.val_noise:
                                   args.train_noise + args.val_noise +
                                   args.test_noise]

        # Train, validation and test seismic progress bars
        train_seismic_bar = tqdm.tqdm(total=args.train_traces,
                                      desc='Train seismic')

        val_seismic_bar = tqdm.tqdm(total=args.val_traces,
                                    desc='Validation seismic')

        test_seismic_bar = tqdm.tqdm(total=args.test_traces,
                                     desc='Test seismic')

        # Train, validation and test non seismic progress bars
        train_noise_bar = tqdm.tqdm(total=args.train_noise,
                                    desc='Train noise')

        val_noise_bar = tqdm.tqdm(total=args.val_noise,
                                  desc='Validation noise')

        test_noise_bar = tqdm.tqdm(total=args.test_noise,
                                   desc='Test noise')

        # Create new train and test files
        with h5py.File('../Data/STEAD/' + args.train_file, 'w') as train_dst,\
                h5py.File('../Data/STEAD/' + args.val_file, 'w') as val_dst, \
                h5py.File('../Data/STEAD/' + args.test_file, 'w') as test_dst:

            # Create new train file groups
            train_dst_wv = train_dst.create_group('earthquake/local')
            train_dst_ns = train_dst.create_group('non_earthquake/noise')

            # Create new val file groups
            val_dst_wv = val_dst.create_group('earthquake/local')
            val_dst_ns = val_dst.create_group('non_earthquake/noise')

            # Create new test file groups
            test_dst_wv = test_dst.create_group('earthquake/local')
            test_dst_ns = test_dst.create_group('non_earthquake/noise')

            # For every dataset in source seismic group
            for idx, dset in enumerate(src_seis):

                if idx in train_seis_ids:

                    # Retrieve dataset object
                    data = src_seis[dset]

                    # Copy seismic trace to new train file
                    train_dst_wv.copy(data, dset)
                    train_seismic_bar.update()

                if idx in val_seis_ids:

                    # Retrieve dataset object
                    data = src_seis[dset]

                    # Copy seismic trace to new train file
                    val_dst_wv.copy(data, dset)
                    val_seismic_bar.update()

                if idx in test_seis_ids:

                    # Retrieve dataset object
                    data = src_seis[dset]

                    # Copy seismic trace to new train file
                    test_dst_wv.copy(data, dset)
                    test_seismic_bar.update()

            # For every dataset in source noise group
            for idx, dset in enumerate(src_ns):

                if idx in train_noise_ids:

                    # Retrieve dataset object
                    data = src_ns[dset]

                    # Copy noise trace to new noise file
                    train_dst_ns.copy(data, dset)
                    train_noise_bar.update()

                if idx in val_noise_ids:

                    # Retrieve dataset object
                    data = src_ns[dset]

                    # Copy seismic trace to new train file
                    val_dst_ns.copy(data, dset)
                    val_noise_bar.update()

                if idx in test_noise_ids:

                    # Retrieve dataset object
                    data = src_ns[dset]

                    # Copy seismic trace to new train file
                    test_dst_ns.copy(data, dset)
                    test_noise_bar.update()

            # Add geophone dataset traces
            add_coompana_traces(train_dst_ns, test_dst_ns, val_dst_ns)
            add_lesser_antilles_traces(train_dst_ns, test_dst_ns, val_dst_ns)
            add_north_carolina_traces(train_dst_ns, test_dst_ns, val_dst_ns)


def add_coompana_traces(train_group, test_group, val_group):
    # Get dataset traces as numpy array
    traces = read_coompana()

    # Number of traces per dataset
    # n_train = 15284
    # n_test = 1910
    # n_val = 1910

    n_train = 853
    n_test = 341
    n_val = 341

    n_needed = n_train + n_test + n_val
    traces = traces[:n_needed, :]

    # Split traces between train, test and validation
    traces_train = traces[:n_train, :]
    traces_test = traces[n_train:n_train+n_test, :]
    traces_val = traces[n_train+n_test:, :]

    # Add traces to hdf5 dataset
    add_traces2group('Coompana', traces_train, train_group)
    add_traces2group('Coompana', traces_test, test_group)
    add_traces2group('Coompana', traces_val, val_group)


def add_lesser_antilles_traces(train_group, test_group, val_group):
    # Get dataset traces as numpy array
    traces = read_lesser_antilles_airgun()

    # Number of traces per dataset
    # n_train = 17958
    # n_test = 2245
    # n_val = 2245

    n_train = 853
    n_test = 341
    n_val = 341

    n_needed = n_train + n_test + n_val
    traces = traces[:n_needed, :]

    # Split traces between train, test and validation
    traces_train = traces[:n_train, :]
    traces_test = traces[n_train:n_train + n_test, :]
    traces_val = traces[n_train + n_test:, :]

    # Añadir trazas a los datasets correspondientes
    add_traces2group('Lesser_Antilles', traces_train, train_group)
    add_traces2group('Lesser_Antilles', traces_test, test_group)
    add_traces2group('Lesser_Antilles', traces_val, val_group)


def add_north_carolina_traces(train_group, test_group, val_group):
    # Get dataset traces as numpy array
    traces = read_north_carolina_airgun()

    # Number of traces per dataset
    # n_train = 17958
    # n_test = 2245
    # n_val = 2245

    n_train = 854
    n_test = 342
    n_val = 342

    n_needed = n_train + n_test + n_val
    traces = traces[:n_needed, :]

    # Split traces between train, test and validation
    traces_train = traces[:n_train, :]
    traces_test = traces[n_train:n_train + n_test, :]
    traces_val = traces[n_train + n_test:, :]

    # Añadir trazas a los datasets correspondientes
    add_traces2group('North_Carolina', traces_train, train_group)
    add_traces2group('North_Carolina', traces_test, test_group)
    add_traces2group('North_Carolina', traces_val, val_group)


def add_traces2group(data_name, traces, group):
    for i, tr in enumerate(traces):
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

            data = seg2reader.read_file(f"{dataset_folder}/{fold}/{datafile}")

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
