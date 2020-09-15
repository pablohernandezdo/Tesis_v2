#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Starting training, lr = 1e-4, epochs = 5, batch_size = 16"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-4 --batch_size 16 \
              --classifier M1l --model_name M1l_1e4_16_5epch > ../logs/train/M1l_1e4_16_5epch.txt

echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e4_16_5epch > ../logs/eval/M1l_1e4_16_5epch.txt

echo "Starting training, lr = 1e-4, epochs = 5, batch_size = 32"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-4 --batch_size 32 \
              --classifier M1l --model_name M1l_1e4_32_5epch > ../logs/train/M1l_1e4_32_5epch.txt

echo "Starting evaluation #2"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e4_32_5epch > ../logs/eval/M1l_1e4_32_5epch.txt

echo "Starting training, lr = 1e-4, epochs = 5, batch_size = 64"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-4 --batch_size 64 \
              --classifier M1l --model_name M1l_1e4_64_5epch > ../logs/train/M1l_1e4_64_5epch.txt

echo "Starting evaluation #3"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e4_64_5epch > ../logs/eval/M1l_1e4_64_5epch.txt

echo "Starting training, lr = 1e-4, epochs = 5, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-4 --batch_size 128 \
              --classifier M1l --model_name M1l_1e4_128_5epch > ../logs/train/M1l_1e4_128_5epch.txt

echo "Starting evaluation #4"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e4_128_5epch > ../logs/eval/M1l_1e4_128_5epch.txt

echo "Starting training, lr = 1e-5, epochs = 5, batch_size = 16"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-5 --batch_size 16 \
              --classifier M1l --model_name M1l_1e5_16_5epch > ../logs/train/M1l_1e5_16_5epch.txt

echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e5_16_5epch > ../logs/eval/M1l_1e5_16_5epch.txt

echo "Starting training, lr = 1e-5, epochs = 5, batch_size = 32"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-5 --batch_size 32 \
              --classifier M1l --model_name M1l_1e5_32_5epch > ../logs/train/M1l_1e5_32_5epch.txt

echo "Starting evaluation #2"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e5_32_5epch > ../logs/eval/M1l_1e5_32_5epch.txt

echo "Starting training, lr = 1e-5, epochs = 5, batch_size = 64"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-5 --batch_size 64 \
              --classifier M1l --model_name M1l_1e5_64_5epch > ../logs/train/M1l_1e5_64_5epch.txt

echo "Starting evaluation #3"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e5_64_5epch > ../logs/eval/M1l_1e5_64_5epch.txt

echo "Starting training, lr = 1e-5, epochs = 5, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-5 --batch_size 128 \
              --classifier M1l --model_name M1l_1e5_128_5epch > ../logs/train/M1l_1e5_128_5epch.txt

echo "Starting evaluation #4"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e5_128_5epch > ../logs/eval/M1l_1e5_128_5epch.txt

echo "Starting training, lr = 1e-6, epochs = 5, batch_size = 16"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-6 --batch_size 16 \
              --classifier M1l --model_name M1l_1e6_16_5epch > ../logs/train/M1l_1e6_16_5epch.txt

echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e6_16_5epch > ../logs/eval/M1l_1e6_16_5epch.txt

echo "Starting training, lr = 1e-6, epochs = 5, batch_size = 32"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-6 --batch_size 32 \
              --classifier M1l --model_name M1l_1e6_32_5epch > ../logs/train/M1l_1e6_32_5epch.txt

echo "Starting evaluation #2"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e6_32_5epch > ../logs/eval/M1l_1e6_32_5epch.txt

echo "Starting training, lr = 1e-6, epochs = 5, batch_size = 64"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-6 --batch_size 64 \
              --classifier M1l --model_name M1l_1e6_64_5epch > ../logs/train/M1l_1e6_64_5epch.txt

echo "Starting evaluation #3"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e6_64_5epch > ../logs/eval/M1l_1e6_64_5epch.txt

echo "Starting training, lr = 1e-6, epochs = 5, batch_size = 128"
python ../train_validation.py \
              --train_path $trn --val_path $val       \
              --n_epochs 5 --lr 1e-6 --batch_size 128 \
              --classifier M1l --model_name M1l_1e6_128_5epch > ../logs/train/M1l_1e6_128_5epch.txt

echo "Starting evaluation #4"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier M1l --model_name M1l_1e6_128_5epch > ../logs/eval/M1l_1e6_128_5epch.txt