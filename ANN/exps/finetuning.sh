#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# 2h5k6k_1e4_256

echo "Training model 2h5k6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_10 > ../logs/train/2h5k6k_1e4_256_10.txt

echo "Evaluating model 2h5k6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_10 > ../logs/eval/2h5k6k_1e4_256_10.txt
