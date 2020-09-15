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

echo "Starting training, lr = 1e-4, epochs = 1, batch_size = 32"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-4 --batch_size 32 \
              --classifier M1r --model_name M1r_1e4_32 > ../logs/train/M1r_1e4_32.txt

echo "Starting evaluation #2"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e4_32 > ../logs/eval/M1r_1e4_32.txt

echo "Starting training, lr = 1e-4, epochs = 1, batch_size = 64"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-4 --batch_size 64 \
              --classifier M1r --model_name M1r_1e4_64 > ../logs/train/M1r_1e4_64.txt

echo "Starting evaluation #3"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e4_64 > ../logs/eval/M1r_1e4_64.txt

echo "Starting training, lr = 1e-4, epochs = 1, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-4 --batch_size 128 \
              --classifier M1r --model_name M1r_1e4_128 > ../logs/train/M1r_1e4_128.txt

echo "Starting evaluation #4"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e4_128 > ../logs/eval/M1r_1e4_128.txt

echo "Starting training, lr = 1e-5, epochs = 1, batch_size = 16"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-5 --batch_size 16 \
              --classifier M1r --model_name M1r_1e5_16 > ../logs/train/M1r_1e5_16.txt

echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e5_16 > ../logs/eval/M1r_1e5_16.txt

echo "Starting training, lr = 1e-5, epochs = 1, batch_size = 32"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-5 --batch_size 32 \
              --classifier M1r --model_name M1r_1e5_32 > ../logs/train/M1r_1e5_32.txt

echo "Starting evaluation #2"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e5_32 > ../logs/eval/M1r_1e5_32.txt

echo "Starting training, lr = 1e-5, epochs = 1, batch_size = 64"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-5 --batch_size 64 \
              --classifier M1r --model_name M1r_1e5_64 > ../logs/train/M1r_1e5_64.txt

echo "Starting evaluation #3"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e5_64 > ../logs/eval/M1r_1e5_64.txt

echo "Starting training, lr = 1e-5, epochs = 1, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-5 --batch_size 128 \
              --classifier M1r --model_name M1r_1e5_128 > ../logs/train/M1r_1e5_128.txt

echo "Starting evaluation #4"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e5_128 > ../logs/eval/M1r_1e5_128.txt

echo "Starting training, lr = 1e-6, epochs = 1, batch_size = 16"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-6 --batch_size 16 \
              --classifier M1r --model_name M1r_1e6_16 > ../logs/train/M1r_1e6_16.txt

echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e6_16 > ../logs/eval/M1r_1e6_16.txt

echo "Starting training, lr = 1e-6, epochs = 1, batch_size = 32"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-6 --batch_size 32 \
              --classifier M1r --model_name M1r_1e6_32 > ../logs/train/M1r_1e6_32.txt

echo "Starting evaluation #2"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e6_32 > ../logs/eval/M1r_1e6_32.txt

echo "Starting training, lr = 1e-6, epochs = 1, batch_size = 64"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-6 --batch_size 64 \
              --classifier M1r --model_name M1r_1e6_64 > ../logs/train/M1r_1e6_64.txt

echo "Starting evaluation #3"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e6_64 > ../logs/eval/M1r_1e6_64.txt

echo "Starting training, lr = 1e-6, epochs = 1, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 1 --lr 1e-6 --batch_size 128 \
              --classifier M1r --model_name M1r_1e6_128 > ../logs/train/M1r_1e6_128.txt

echo "Starting evaluation #4"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1r --model_name M1r_1e6_128 > ../logs/eval/M1r_1e6_128.txt