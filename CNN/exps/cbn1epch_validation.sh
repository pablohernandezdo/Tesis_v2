#!/bin/bash

mkdir -p ../logs/CBN/train
mkdir -p ../logs/CBN/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Train CBN with evaluation for 1 epoch
echo "Training CBN model on $trn and validating on $val"
python ../train_validation.py --train_path $trn --val_path $val  \
              --classifier CBN --model_name CBN_1epch_validation \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/CBN/train/CBN_1epch_validation.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier CBN --model_name CBN_1epch_validation > ../logs/CBN/eval/CBN_1epch_validation.txt