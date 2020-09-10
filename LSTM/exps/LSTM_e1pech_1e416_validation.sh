#!/bin/bash

mkdir -p ../logs/LSTM/train
mkdir -p ../logs/LSTM/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Train LSTM with evaluation for 1 epoch
echo "Training CBN model on $trn and validating on $val"
python ../train_validation.py --train_path $trn --val_path $val \
              --classifier LSTM --model_name LSTM_1e4_64_validation \
              --n_epochs 1 --batch_size 64 --lr 1e-4  > ../logs/LSTM/train/LSTM_1e4_64_validation.txt

# Evaluate LSTM on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1e4_64_validation > ../logs/LSTM/eval/LSTM_1e4_64_validation.txt
