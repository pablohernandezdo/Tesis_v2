#!/bin/bash

trn="Train_data.hdf5"
val="Validation_data.hdf5"
tst="Test_data.hdf5"

# Check train dataset
echo "Checking $trn dataset"
python checkdataset.py --dataset_path '../Data/STEAD/Train_data_v3.hdf5'

# Check validation dataset
echo "Checking $val dataset"
python checkdataset.py --dataset_path '../Data/STEAD/Validation_data_v3.hdf5'

# Check test dataset
echo "Checking $tst dataset"
python checkdataset.py --dataset_path '../Data/STEAD/STEAD_Seis_TEST.hdf5'

# Check test dataset
echo "Checking $tst dataset"
python checkdataset.py --dataset_path '../Data/STEAD/STEAD_NSeis_TEST.hdf5'

# Check test dataset
echo "Checking $tst dataset"
python checkdataset.py --dataset_path '../Data/STEAD/GEO_TEST.hdf5'
