import h5py


def main():

    with h5py.File('Train_data.hdf5', 'r') as h5_file:
        print(list(h5_file.keys()))

    with h5py.File('Test_geo.hdf5', 'r') as h5_file:
        print(list(h5_file.keys()))


if __name__ == "__main__":
    main()

