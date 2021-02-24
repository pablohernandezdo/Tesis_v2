import h5py
import argparse
import numpy as np
from numpy.random import default_rng

import tqdm


def main():
    # Create new train, validation and test sets from STEAD
    # and non seismic geophone dataset files

    # Args
    parser = argparse.ArgumentParser(description='Dataset creation parameters')
    parser.add_argument('--source_file', default='../Data/STEAD/STEAD.hdf5', help='Source HDF5 file path')
    parser.add_argument('--train_file', default='Train_data_v2.hdf5', help='Output train HDF5 file path')
    parser.add_argument('--val_file', default='Validation_data_v2.hdf5', help='Output validation HDF5 file path')
    parser.add_argument('--test_file', default='Test_data_v2.hdf5', help='Output test HDF5 file path')
    parser.add_argument('--train_traces', type=int, default=6000, help='Number of training seismic traces to copy')
    parser.add_argument('--train_noise', type=int, default=6000, help='Number of training noise traces to copy')
    parser.add_argument('--val_traces', type=int, default=2000, help='Number of validation seismic traces to copy')
    parser.add_argument('--val_noise', type=int, default=2000, help='Number of validation noise traces to copy')
    parser.add_argument('--test_traces', type=int, default=2000, help='Number of test seismic traces to copy')
    parser.add_argument('--test_noise', type=int, default=2000, help='Number of test noise traces to copy')
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
        seismic_ids = rng.choice(len(src_seis), size=seis2copy, replace=False)
        noise_ids = rng.choice(len(src_ns), size=ns2copy, replace=False)

        # Not faulty datasets in selection
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

        val_seis_ids = seismic_ids[args.train_traces:args.train_traces + args.val_traces]
        val_noise_ids = noise_ids[args.train_noise:args.train_noise + args.val_noise]

        test_seis_ids = seismic_ids[args.train_traces + args.val_traces:args.train_traces + args.val_traces+args.test_traces]
        test_noise_ids = noise_ids[args.train_noise + args.val_noise:args.train_noise + args.val_noise+args.test_noise]

        # Train, validation and test seismic progress bars
        train_seismic_bar = tqdm.tqdm(total=args.train_traces, desc='Train seismic')
        val_seismic_bar = tqdm.tqdm(total=args.val_traces, desc='Validation seismic')
        test_seismic_bar = tqdm.tqdm(total=args.test_traces, desc='Test seismic')

        # Train, validation and test non seismic progress bars
        train_noise_bar = tqdm.tqdm(total=args.train_noise, desc='Train noise')
        val_noise_bar = tqdm.tqdm(total=args.val_noise, desc='Validation noise')
        test_noise_bar = tqdm.tqdm(total=args.test_noise, desc='Test noise')

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


if __name__ == '__main__':
    main()
