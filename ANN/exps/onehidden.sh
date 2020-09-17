#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Training model 1h6k, lr = 1e-4, epochs = 5, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 128 \
              --classifier 1h6k --model_name 1h6k_1e4_128 > ../logs/train/1h6k_1e4_128.txt

echo "Evaluating model 1h6k_1e4_128"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h6k --model_name 1h6k_1e4_128 > ../logs/eval/1h6k_1e4_128.txt

echo "Training model 1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h6k --model_name 1h6k_1e4_256 > ../logs/train/1h6k_1e4_256.txt

echo "Evaluating model 1h6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h6k --model_name 1h6k_1e4_256 > ../logs/eval/1h6k_1e4_256.txt
