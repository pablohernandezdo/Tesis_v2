#!/bin/bash

mkdir -p ../logs/CBN_v2/train
mkdir -p ../logs/CBN_v2/eval
mkdir -p ../models/

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn --test_path $tst \
              --classifier CBN_v2 --model_name CBN_v2_1epch \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/CBN_v2/train/CBN_v2_1epch.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier CBN_v2 --model_name CBN_v2_1epch > ../logs/CBN_v2/eval/CBN_v2_1epch.txt
