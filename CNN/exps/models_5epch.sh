#!/bin/bash

mkdir -p ../logs/CNN/train
mkdir -p ../logs/CNN/eval
mkdir -p ../models/

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Train C for 5 epochs
echo "Training C model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier C --model_name C_5epch_1e6_16 \
              --n_epochs 5 --batch_size 16 --lr 1e-6  > ../logs/CNN/train/C_5epch_1e6_16.txt

# Evaluate C on train and test datasets
echo "Evaluating C model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier C \
              --model_name C_5epch_1e6_16 > ../logs/CNN/eval/C_5epch_1e6_16.txt

# Train C for 5 epochs
echo "Training C model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier C --model_name C_5epch_1e6_32 \
              --n_epochs 5 --batch_size 32 --lr 1e-6  > ../logs/CNN/train/C_5epch_1e6_32.txt

# Evaluate C on train and test datasets
echo "Evaluating C model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier C \
              --model_name C_5epch_1e6_32 > ../logs/CNN/eval/C_5epch_1e6_32.txt

# Train C for 1 epoch
echo "Training C model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier C --model_name C_5epch_1e6_64 \
              --n_epochs 5 --batch_size 64 --lr 1e-6  > ../logs/CNN/train/C_5epch_1e6_64.txt

# Evaluate C on train and test datasets
echo "Evaluating C model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier C \
              --model_name C_5epch_1e6_64 > ../logs/CNN/eval/C_5epch_1e6_64.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier CBN --model_name CBN_5epch_1e6_16 \
              --n_epochs 5 --batch_size 16 --lr 1e-6  > ../logs/CNN/train/CBN_5epch_1e6_16.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN \
              --model_name CBN_5epch_1e6_16 > ../logs/CNN/eval/CBN_5epch_1e6_16.txt

# Train CBN for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier CBN --model_name CBN_5epch_1e6_32 \
              --n_epochs 5 --batch_size 32 --lr 1e-6  > ../logs/CNN/train/CBN_5epch_1e6_32.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN \
              --model_name CBN_5epch_1e6_32 > ../logs/CNN/eval/CBN_5epch_1e6_32.txt

# Train C for 1 epoch
echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier CBN --model_name CBN_5epch_1e6_64 \
              --n_epochs 5 --batch_size 64 --lr 1e-6  > ../logs/CNN/train/CBN_5epch_1e6_64.txt

# Evaluate CBN on train and test datasets
echo "Evaluating CBN model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN \
              --model_name CBN_5epch_1e6_64 > ../logs/CNN/eval/CBN_5epch_1e6_64.txt

# Train CBN_v2 for 1 epoch
echo "Training CBN_v2 model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier CBN_v2 --model_name CBN_v2_5epch_1e6_16 \
              --n_epochs 5 --batch_size 16 --lr 1e-6  > ../logs/CNN/train/CBN_v2_5epch_1e6_16.txt

# Evaluate CBN_v2 on train and test datasets
echo "Evaluating CBN_v2 model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN_v2 \
              --model_name CBN_v2_5epch_1e6_16 > ../logs/CNN/eval/CBN_v2_5epch_1e6_16.txt

# Train CBN_v2 for 1 epoch
echo "Training CBN_v2 model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier CBN_v2 --model_name CBN_v2_5epch_1e6_32 \
              --n_epochs 5 --batch_size 32 --lr 1e-6  > ../logs/CNN/train/CBN_v2_5epch_1e6_32.txt

# Evaluate CBN_v2 on train and test datasets
echo "Evaluating CBN_v2 model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN_v2 \
              --model_name CBN_v2_5epch_1e6_32 > ../logs/CNN/eval/CBN_v2_5epch_1e6_32.txt

# Train C for 1 epoch
echo "Training CBN_v2 model on $trn and $tst datasets"
python ../train.py --train_path $trn \
              --classifier CBN_v2 --model_name CBN_v2_5epch_1e6_64 \
              --n_epochs 5 --batch_size 64 --lr 1e-6  > ../logs/CNN/train/CBN_v2_5epch_1e6_64.txt

# Evaluate CBN_v2 on train and test datasets
echo "Evaluating CBN_v2 model on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst \
              --thresh 0.5 --classifier CBN_v2 \
              --model_name CBN_v2_5epch_1e6_64 > ../logs/CNN/eval/CBN_v2_5epch_1e6_64.txt