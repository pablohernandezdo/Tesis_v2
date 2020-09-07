#!/bin/bash

mkdir -p ../logs/LSTM/train
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"
val="Validation_data.hdf5"

## Train CBN_v2 with evaluation for 1 epoch
#echo "Training CBN model on $trn and validating on $val"
#python ../train_validation.py --train_path $trn --val_path $val \
#              --classifier C --model_name LSTM_1e4_64_validation \
#              --n_epochs 1 --batch_size 64 --lr 1e-4  > ../logs/LSTM/train/LSTM_1e4_64_validation.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier C --model_name LSTM_1e4_64_validation > ../logs/LSTM/eval/LSTM_1e4_64_validation.txt
