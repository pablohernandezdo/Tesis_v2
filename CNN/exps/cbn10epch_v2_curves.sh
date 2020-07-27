#!/bin/bash

mkdir -p ../logs/CBN_v2/train
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Train ClassConv_BN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train_curves.py --train_path $trn --test_path $tst \
              --classifier CBN_v2 --model_name CBN_v2_1epch \
              --n_epochs 10 --batch_size 32 --lr 1e-6  > ../logs/CBN_v2/train/CBN_v2_1epch_curves.txt
