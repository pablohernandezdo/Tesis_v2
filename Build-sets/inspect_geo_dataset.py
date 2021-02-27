import h5py
import tqdm
import argparse
import numpy as np


def main():
    # Open dataset
    with h5py.File("../Data/STEAD/Train_data_v2.hdf5", 'r') as f:

        # Read groups
        seismic = f['earthquake']['local']
        noise = f['non_earthquake']['noise']

        for dset in noise:
            print(dset) 


if __name__ == "__main__":
    main()
