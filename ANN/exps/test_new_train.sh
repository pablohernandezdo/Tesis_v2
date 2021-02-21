#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Learning rate 1e-3
echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn     \
              --val_path $val       \
              --model_folder step4  \
              --n_epochs 5          \
              --lr 1e-3             \
              --batch_size 256      \
              --patience 20         \
              --classifier 2h5h5k   \
              --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_20.txt
