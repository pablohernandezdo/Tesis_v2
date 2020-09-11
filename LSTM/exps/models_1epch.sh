#!/bin/bash

mkdir -p ../logs/LSTM/train
mkdir -p ../logs/LSTM/eval
mkdir -p ../models/

tst="Test_data.hdf5"
trn="Train_data.hdf5"

# Train LSTM for 1 epoch
echo "Training LSTM model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_16.txt

# Evaluate LSTM on train and test datasets
echo "Evaluating LSTM model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier LSTM \
              --model_name LSTM_1epch_1e4_16 > ../logs/LSTM/eval/LSTM_1epch_1e4_16.txt

# Train LSTM for 1 epoch
echo "Training LSTM model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_32.txt

# Evaluate LSTM on train and test datasets
echo "Evaluating LSTM model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier LSTM \
              --model_name LSTM_1epch_1e4_32 > ../logs/LSTM/eval/LSTM_1epch_1e4_32.txt

# Train LSTM for 1 epoch
echo "Training LSTM model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM --model_name LSTM_1epch_1e4_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-4  > ../logs/LSTM/train/LSTM_1epch_1e4_64.txt

# Evaluate LSTM on train and test datasets
echo "Evaluating LSTM model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier LSTM \
              --model_name LSTM_1epch_1e4_64 > ../logs/LSTM/eval/LSTM_1epch_1e4_64.txt

# Train LSTM_v2 for 1 epoch
echo "Training LSTM_v2 model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM_v2 --model_name LSTM_v2_1epch_1e4_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-4  > ../logs/LSTM/train/LSTM_v2_1epch_1e4_16.txt

# Evaluate LSTM_v2 on train and test datasets
echo "Evaluating LSTM_v2 model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier LSTM_v2 \
              --model_name LSTM_v2_1epch_1e4_16 > ../logs/LSTM/eval/LSTM_v2_1epch_1e4_16.txt

# Train LSTM_v2 for 1 epoch
echo "Training LSTM_v2 model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM_v2 --model_name LSTM_v2_1epch_1e4_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-4  > ../logs/LSTM/train/LSTM_v2_1epch_1e4_32.txt

# Evaluate LSTM_v2 on train and test datasets
echo "Evaluating LSTM_v2 model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier LSTM_v2 \
              --model_name LSTM_v2_1epch_1e4_32 > ../logs/LSTM/eval/LSTM_v2_1epch_1e4_32.txt

# Train LSTM_v2 for 1 epoch
echo "Training LSTM_v2 model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier LSTM_v2 --model_name LSTM_v2_1epch_1e4_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-4  > ../logs/LSTM/train/LSTM_v2_1epch_1e4_64.txt

# Evaluate LSTM_v2 on train and test datasets
echo "Evaluating LSTM_v2 model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier LSTM_v2 \
              --model_name LSTM_1epch_1e4_64 > ../logs/LSTM/eval/LSTM_v2_1epch_1e4_64.txt

