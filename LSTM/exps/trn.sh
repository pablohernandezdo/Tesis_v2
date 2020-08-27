#!/bin/bash

mkdir -p ../logs/LSTM/train
mkdir -p ../logs/LSTM/eval
mkdir -p ../models/

trn="Train_data.hdf5"

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier C --model_name LSTM_1epch \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/LSTM/train/LSTM_1epch.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier C --model_name LSTM_1epch > ../logs/LSTM/eval/LSTM_1epch.txt