#!/bin/bash

mkdir -p ../logs/CBN/train
mkdir -p ../logs/CBN/eval
mkdir -p ../models/

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn --test_path $tst \
              --classifier CBN --model_name CBN_10epch \
              --n_epochs 10 --batch_size 32 --lr 1e-6  > ../logs/CBN/train/CBN_10epch.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN \
              --model_name CBN_10epch > ../logs/CBN/eval/CBN_10epch.txt