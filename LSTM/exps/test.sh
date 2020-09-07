#!/bin/bash

mkdir -p ../logs/LSTM/train
mkdir -p ../logs/LSTM/eval
mkdir -p ../models/

trn="Train_data.hdf5"

# Train CBN for 1 epoch
echo "Training LSTM model on $trn dataset"
python ../train.py --train_path $trn  \
              --classifier C --model_name LSTM_1epch_1e4_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_64.txt
