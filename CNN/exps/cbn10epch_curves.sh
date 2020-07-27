#!/bin/bash

mkdir -p ../logs/CBN/train_val
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Train ClassConv_BN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train_curves.py --train_path $trn --test_path $tst \
              --classifier CBN_val --model_name CBN_val_10epch \
              --n_epochs 10 --batch_size 32 --lr 1e-6  > ../logs/CBN/train_val/CBN_val_10epch_curves.txt
