import h5py
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Check STEAD dataset

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='../Data/STEAD/STEAD.hdf5', help="HDF5 Dataset path")
    parser.add_argument("--type", default='seismic', help="Type of trace to show")
    parser.add_argument("--num", type=int, default=0, help="Number of trace to show")
    args = parser.parse_args()

    # Open dataset
    with h5py.File(args.dataset_path, 'r') as dset:

        # Read groups
        seismic = dset['earthquake']['local']
        noise = dset['non_earthquake']['noise']

        plt.figure()

        if args.type == 'seismic':
            for idx, dset in enumerate(seismic):

                if idx == args.num:
                    data = seismic[dset]
                    plt.plot(data[:, 0])
                    plt.show()
                    break

        else:
            for idx, dset in enumerate(noise):

                if idx == args.num:
                    data = noise[dset]
                    plt.plot(data[:, 0])
                    plt.show()
                    break


if __name__ == "__main__":
    main()
