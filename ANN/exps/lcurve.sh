#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# lr = 1e-6, bs = 32, Classifier_XXL
echo "Starting training, lr = 1e-7, epochs = 10, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 300 --lr 1e-7 --batch_size 128 \
              --classifier XXL --model_name XXL_1e6_32 > ../logs/train/XXL_1e6_32_XXL.txt

# Classifier_XXL
echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier XXL --model_name XXL_1e6_32 > ../logs/eval/XXL_1e6_32.txt
