#!/bin/bash

mkdir -p ../logs/ANN/train
mkdir -p ../logs/ANN/eval
mkdir -p ../models/

tst="Test_data.hdf5"
trn="Train_data.hdf5"

# Train ANN model C for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier C --model_name C_1epch_1e6_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-6  > ../logs/ANN/train/C_1epch_1e6_16.txt

# Evaluate ANN model C on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier C \
              --model_name C_1epch_1e6_16 > ../logs/ANN/eval/C_1epch_1e6_16.txt

# Train ANN model C for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier C --model_name C_1epch_1e6_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/ANN/train/C_1epch_1e6_32.txt

# Evaluate ANN model C on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier C \
              --model_name C_1epch_1e6_32 > ../logs/ANN/eval/C_1epch_1e6_32.txt

# Train ANN model C for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier C --model_name C_1epch_1e6_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-6  > ../logs/ANN/train/C_1epch_1e6_64.txt

# Evaluate ANN model C on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier C \
              --model_name C_1epch_1e6_64 > ../logs/ANN/eval/C_1epch_1e6_64.txt

# Train ANN model S for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier S --model_name S_1epch_1e6_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-6  > ../logs/ANN/train/S_1epch_1e6_16.txt

# Evaluate ANN model S on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier S \
              --model_name S_1epch_1e6_16 > ../logs/ANN/eval/S_1epch_1e6_16.txt

# Train ANN model S for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier S --model_name S_1epch_1e6_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/ANN/train/S_1epch_1e6_32.txt

# Evaluate ANN model S on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier S \
              --model_name S_1epch_1e6_32 > ../logs/ANN/eval/S_1epch_1e6_32.txt

# Train ANN model S for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier S --model_name S_1epch_1e6_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-6  > ../logs/ANN/train/S_1epch_1e6_64.txt

# Evaluate ANN model S on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier S \
              --model_name S_1epch_1e6_64 > ../logs/ANN/eval/S_1epch_1e6_64.txt

# Train ANN model XS for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XS --model_name XS_1epch_1e6_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-6  > ../logs/ANN/train/XS_1epch_1e6_16.txt

# Evaluate ANN model XS on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XS \
              --model_name XS_1epch_1e6_16 > ../logs/ANN/eval/XS_1epch_1e6_16.txt

# Train ANN model XS for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XS --model_name XS_1epch_1e6_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/ANN/train/XS_1epch_1e6_32.txt

# Evaluate ANN model XS on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XS \
              --model_name XS_1epch_1e6_32 > ../logs/ANN/eval/XS_1epch_1e6_32.txt

# Train ANN model XS for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XS --model_name XS_1epch_1e6_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-6  > ../logs/ANN/train/XS_1epch_1e6_64.txt

# Evaluate ANN model XS on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XS \
              --model_name XS_1epch_1e6_64 > ../logs/ANN/eval/XS_1epch_1e6_64.txt

# Train ANN model XL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XL --model_name XL_1epch_1e6_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-6  > ../logs/ANN/train/XL_1epch_1e6_16.txt

# Evaluate ANN model XL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XL \
              --model_name XL_1epch_1e6_16 > ../logs/ANN/eval/XL_1epch_1e6_16.txt

# Train ANN model XL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XL --model_name XL_1epch_1e6_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/ANN/train/XL_1epch_1e6_32.txt

# Evaluate ANN model XL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XL \
              --model_name XL_1epch_1e6_32 > ../logs/ANN/eval/XL_1epch_1e6_32.txt

# Train ANN model XL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XL --model_name XL_1epch_1e6_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-6  > ../logs/ANN/train/XL_1epch_1e6_64.txt

# Evaluate ANN model XL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XL \
              --model_name XL_1epch_1e6_64 > ../logs/ANN/eval/XL_1epch_1e6_64.txt

# Train ANN model XXL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XXL --model_name XXL_1epch_1e6_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-6  > ../logs/ANN/train/XXL_1epch_1e6_16.txt

# Evaluate ANN model XXL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XXL \
              --model_name XXL_1epch_1e6_16 > ../logs/ANN/eval/XXL_1epch_1e6_16.txt

# Train ANN model XXL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XXL --model_name XXL_1epch_1e6_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/ANN/train/XXL_1epch_1e6_32.txt

# Evaluate ANN model XXL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XXL \
              --model_name XXL_1epch_1e6_32 > ../logs/ANN/eval/XXL_1epch_1e6_32.txt

# Train ANN model XXL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XXL --model_name XXL_1epch_1e6_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-6  > ../logs/ANN/train/XXL_1epch_1e6_64.txt

# Evaluate ANN model XXL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XXL \
              --model_name XXL_1epch_1e6_64 > ../logs/ANN/eval/XXL_1epch_1e6_64.txt

# Train ANN model XXXL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XXXL --model_name XXXL_1epch_1e6_16 \
              --n_epochs 1 --batch_size 16 --lr 1e-6  > ../logs/ANN/train/XXXL_1epch_1e6_16.txt

# Evaluate ANN model XXXL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XXXL \
              --model_name XXXL_1epch_1e6_16 > ../logs/ANN/eval/XXXL_1epch_1e6_16.txt

# Train ANN model XXXL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XXXL --model_name XXXL_1epch_1e6_32 \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/ANN/train/XXXL_1epch_1e6_32.txt

# Evaluate ANN model XXXL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XXXL \
              --model_name XXXL_1epch_1e6_32 > ../logs/ANN/eval/XXXL_1epch_1e6_32.txt

# Train ANN model XXXL for 1 epoch
echo "Training ANN model on $trn and $tst datasets"
python ../train.py --train_path $trn  \
              --classifier XXXL --model_name XXXL_1epch_1e6_64 \
              --n_epochs 1 --batch_size 64 --lr 1e-6  > ../logs/ANN/train/XXXL_1epch_1e6_64.txt

# Evaluate ANN model XXXL on train and test datasets
echo "Evaluating ANN model on $trn and $tst datasets"
python ../eval.py --train_path $trn  \
              --thresh 0.5 --classifier XXXL \
              --model_name XXXL_1epch_1e6_64 > ../logs/ANN/eval/XXXL_1epch_1e6_64.txt
