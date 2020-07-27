import h5py
import argparse
import numpy as np


def main():
    # Check STEAD dataset

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='../Data/STEAD/STEAD.hdf5', help="HDF5 Dataset path")
    args = parser.parse_args()

    # Open dataset
    with h5py.File(args.dataset_path, 'r') as dset:

        # Read groups
        seismic = dset['earthquake']['local']
        noise = dset['non_earthquake']['noise']

        seismic_bad = 0
        noise_bad = 0

        for idx, dset in enumerate(seismic):

            data = seismic[dset]

            if not np.max(np.abs(data[:, 0])):
                seismic_bad += 1
                print(f'Faulty seismic dataset id: {idx}')

        for idx, dset in enumerate(noise):

            data = noise[dset]

            if not np.max(np.abs(data[:, 0])):
                noise_bad += 1
                print(f'Faulty noise dataset id: {idx}')

        # Print number of traces
        print(f'Number of seismic traces: {len(seismic)}\n'
              f'Number of noise traces: {len(noise)}\n'
              f'Number of faulty seismic datasets: {seismic_bad}\n'
              f'Number of faulty noise datasets: {noise_bad}\n')


if __name__ == "__main__":
    main()
