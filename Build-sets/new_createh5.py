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


if __name__ == '__main__':
    main()
