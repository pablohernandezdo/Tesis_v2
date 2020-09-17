#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Starting training, lr = 1e-4, epochs = 1, batch_size = 16"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-4 --batch_size 16 \
              --classifier M1r --model_name M1r_1e4_16 > ../logs/train/M1r_1e4_16.txt

echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e4_16 > ../logs/eval/M1r_1e4_16.txt
