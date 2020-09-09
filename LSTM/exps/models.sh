#!/bin/bash

mkdir -p ../logs/LSTM/train
mkdir -p ../logs/LSTM/eval
mkdir -p ../models/

trn="Train_data.hdf5"


# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_16.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_16 > ../logs/LSTM/eval/LSTM_1epch_1e4_16.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_32.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_32 > ../logs/LSTM/eval/LSTM_1epch_1e4_32.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_64.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_64 > ../logs/LSTM/eval/LSTM_1epch_1e4_64.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_5e5_16 \
              --n_epochs 1 --batch_size 16 --lr 5e-5  > ../logs/LSTM/train/LSTM_1epch_5e5_16.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_5e5_16 > ../logs/LSTM/eval/LSTM_1epch_5e5_16.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_5e5_32 \
              --n_epochs 1 --batch_size 32 --lr 5e-5  > ../logs/LSTM/train/LSTM_1epch_5e5_32.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_5e5_32 > ../logs/LSTM/eval/LSTM_1epch_5e5_32.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_5e5_64 \
              --n_epochs 1 --batch_size 64 --lr 5e-5  > ../logs/LSTM/train/LSTM_1epch_5e5_64.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_5e5_64 > ../logs/LSTM/eval/LSTM_1epch_5e5_64.txt

