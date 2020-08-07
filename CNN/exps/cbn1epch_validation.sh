#!/bin/bash

mkdir -p ../logs/CBN/train
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Train CBN with evaluation for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train_validation.py --train_path $trn --test_path $tst \
              --classifier CBN --model_name CBN_1epch_validation \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/CBN/train/CBN_1epch_validation.txt