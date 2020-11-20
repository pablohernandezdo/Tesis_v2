#!/bin/bash

mkdir -p ../Analysis/logs/train/step3
mkdir -p ../Analysis/logs/eval/step3
mkdir -p ../models/step3

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

## 1 Capa, 16-16
#
#echo "Training model Lstm_16_16_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_30 > ../Analysis/logs/train/step3/Lstm_16_16_1_1_30.txt
#
#echo "Evaluating model Lstm_16_16_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_30 > ../Analysis/logs/eval/step3/Lstm_16_16_1_1_30.txt
#
#echo "Training model Lstm_16_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_2_1 --model_name Lstm_16_16_2_1_30 > ../Analysis/logs/train/step3/Lstm_16_16_2_1_30.txt
#
#echo "Evaluating model Lstm_16_16_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_2_1 --model_name Lstm_16_16_2_1_30 > ../Analysis/logs/eval/step3/Lstm_16_16_2_1_30.txt
#
#echo "Training model Lstm_16_16_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_5_1 --model_name Lstm_16_16_5_1_30 > ../Analysis/logs/train/step3/Lstm_16_16_5_1_30.txt
#
#echo "Evaluating model Lstm_16_16_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_5_1 --model_name Lstm_16_16_5_1_30 > ../Analysis/logs/eval/step3/Lstm_16_16_5_1_30.txt
#
#echo "Training model Lstm_16_16_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_10_1 --model_name Lstm_16_16_10_1_30 > ../Analysis/logs/train/step3/Lstm_16_16_10_1_30.txt
#
#echo "Evaluating model Lstm_16_16_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_10_1 --model_name Lstm_16_16_10_1_30 > ../Analysis/logs/eval/step3/Lstm_16_16_10_1_30.txt
#
#echo "Training model Lstm_16_16_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_20_1 --model_name Lstm_16_16_20_1_30 > ../Analysis/logs/train/step3/Lstm_16_16_20_1_30.txt
#
#echo "Evaluating model Lstm_16_16_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_20_1 --model_name Lstm_16_16_20_1_30 > ../Analysis/logs/eval/step3/Lstm_16_16_20_1_30.txt
#
## 1 Capa, 16-32
#
#echo "Training model Lstm_16_32_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_1_1 --model_name Lstm_16_32_1_1_30 > ../Analysis/logs/train/step3/Lstm_16_32_1_1_30.txt
#
#echo "Evaluating model Lstm_16_32_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_1_1 --model_name Lstm_16_32_1_1_30 > ../Analysis/logs/eval/step3/Lstm_16_32_1_1_30.txt
#
#echo "Training model Lstm_16_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_2_1 --model_name Lstm_16_32_2_1_30 > ../Analysis/logs/train/step3/Lstm_16_32_2_1_30.txt
#
#echo "Evaluating model Lstm_16_32_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_2_1 --model_name Lstm_16_32_2_1_30 > ../Analysis/logs/eval/step3/Lstm_16_32_2_1_30.txt
#
#echo "Training model Lstm_16_32_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_5_1 --model_name Lstm_16_32_5_1_30 > ../Analysis/logs/train/step3/Lstm_16_32_5_1_30.txt
#
#echo "Evaluating model Lstm_16_32_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_5_1 --model_name Lstm_16_32_5_1_30 > ../Analysis/logs/eval/step3/Lstm_16_32_5_1_30.txt
#
#echo "Training model Lstm_16_32_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_10_1 --model_name Lstm_16_32_10_1_30 > ../Analysis/logs/train/step3/Lstm_16_32_10_1_30.txt
#
#echo "Evaluating model Lstm_16_32_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_10_1 --model_name Lstm_16_32_10_1_30 > ../Analysis/logs/eval/step3/Lstm_16_32_10_1_30.txt
#
#echo "Training model Lstm_16_32_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_20_1 --model_name Lstm_16_32_20_1_30 > ../Analysis/logs/train/step3/Lstm_16_32_20_1_30.txt
#
#echo "Evaluating model Lstm_16_32_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_20_1 --model_name Lstm_16_32_20_1_30 > ../Analysis/logs/eval/step3/Lstm_16_32_20_1_30.txt
#
## 1 Capa, 16-64
#
#echo "Training model Lstm_16_64_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_1_1 --model_name Lstm_16_64_1_1_30 > ../Analysis/logs/train/step3/Lstm_16_64_1_1_30.txt
#
#echo "Evaluating model Lstm_16_64_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_1_1 --model_name Lstm_16_64_1_1_30 > ../Analysis/logs/eval/step3/Lstm_16_64_1_1_30.txt
#
#echo "Training model Lstm_16_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_2_1 --model_name Lstm_16_64_2_1_30 > ../Analysis/logs/train/step3/Lstm_16_64_2_1_30.txt
#
#echo "Evaluating model Lstm_16_64_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_2_1 --model_name Lstm_16_64_2_1_30 > ../Analysis/logs/eval/step3/Lstm_16_64_2_1_30.txt
#
#echo "Training model Lstm_16_64_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_5_1 --model_name Lstm_16_64_5_1_30 > ../Analysis/logs/train/step3/Lstm_16_64_5_1_30.txt
#
#echo "Evaluating model Lstm_16_64_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_5_1 --model_name Lstm_16_64_5_1_30 > ../Analysis/logs/eval/step3/Lstm_16_64_5_1_30.txt
#
#echo "Training model Lstm_16_64_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_10_1 --model_name Lstm_16_64_10_1_30 > ../Analysis/logs/train/step3/Lstm_16_64_10_1_30.txt
#
#echo "Evaluating model Lstm_16_64_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_10_1 --model_name Lstm_16_64_10_1_30 > ../Analysis/logs/eval/step3/Lstm_16_64_10_1_30.txt
#
#echo "Training model Lstm_16_64_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_20_1 --model_name Lstm_16_64_20_1_30 > ../Analysis/logs/train/step3/Lstm_16_64_20_1_30.txt
#
#echo "Evaluating model Lstm_16_64_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_20_1 --model_name Lstm_16_64_20_1_30 > ../Analysis/logs/eval/step3/Lstm_16_64_20_1_30.txt
#
## 1 Capa, 16-128
#
#echo "Training model Lstm_16_128_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_1_1 --model_name Lstm_16_128_1_1_30 > ../Analysis/logs/train/step3/Lstm_16_128_1_1_30.txt
#
#echo "Evaluating model Lstm_16_128_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_1_1 --model_name Lstm_16_128_1_1_30 > ../Analysis/logs/eval/step3/Lstm_16_128_1_1_30.txt
#
#echo "Training model Lstm_16_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_2_1 --model_name Lstm_16_128_2_1_30 > ../Analysis/logs/train/step3/Lstm_16_128_2_1_30.txt
#
#echo "Evaluating model Lstm_16_128_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_2_1 --model_name Lstm_16_128_2_1_30 > ../Analysis/logs/eval/step3/Lstm_16_128_2_1_30.txt
#
#echo "Training model Lstm_16_128_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_5_1 --model_name Lstm_16_128_5_1_30 > ../Analysis/logs/train/step3/Lstm_16_128_5_1_30.txt
#
#echo "Evaluating model Lstm_16_128_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_5_1 --model_name Lstm_16_128_5_1_30 > ../Analysis/logs/eval/step3/Lstm_16_128_5_1_30.txt
#
#echo "Training model Lstm_16_128_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_30 > ../Analysis/logs/train/step3/Lstm_16_128_10_1_30.txt
#
#echo "Evaluating model Lstm_16_128_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_30 > ../Analysis/logs/eval/step3/Lstm_16_128_10_1_30.txt
#
#echo "Training model Lstm_16_128_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_20_1 --model_name Lstm_16_128_20_1_30 > ../Analysis/logs/train/step3/Lstm_16_128_20_1_30.txt
#
#echo "Evaluating model Lstm_16_128_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_20_1 --model_name Lstm_16_128_20_1_30 > ../Analysis/logs/eval/step3/Lstm_16_128_20_1_30.txt
#
## 1 Capa, 16-256
#
#echo "Training model Lstm_16_256_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_1_1 --model_name Lstm_16_256_1_1_30 > ../Analysis/logs/train/step3/Lstm_16_256_1_1_30.txt
#
#echo "Evaluating model Lstm_16_256_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_1_1 --model_name Lstm_16_256_1_1_30 > ../Analysis/logs/eval/step3/Lstm_16_256_1_1_30.txt
#
#echo "Training model Lstm_16_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_2_1 --model_name Lstm_16_256_2_1_30 > ../Analysis/logs/train/step3/Lstm_16_256_2_1_30.txt
#
#echo "Evaluating model Lstm_16_256_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_2_1 --model_name Lstm_16_256_2_1_30 > ../Analysis/logs/eval/step3/Lstm_16_256_2_1_30.txt
#
#echo "Training model Lstm_16_256_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_5_1 --model_name Lstm_16_256_5_1_30 > ../Analysis/logs/train/step3/Lstm_16_256_5_1_30.txt
#
#echo "Evaluating model Lstm_16_256_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_5_1 --model_name Lstm_16_256_5_1_30 > ../Analysis/logs/eval/step3/Lstm_16_256_5_1_30.txt
#
#echo "Training model Lstm_16_256_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_10_1 --model_name Lstm_16_256_10_1_30 > ../Analysis/logs/train/step3/Lstm_16_256_10_1_30.txt
#
#echo "Evaluating model Lstm_16_256_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_10_1 --model_name Lstm_16_256_10_1_30 > ../Analysis/logs/eval/step3/Lstm_16_256_10_1_30.txt
#
#echo "Training model Lstm_16_256_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_20_1 --model_name Lstm_16_256_20_1_30 > ../Analysis/logs/train/step3/Lstm_16_256_20_1_30.txt
#
#echo "Evaluating model Lstm_16_256_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_20_1 --model_name Lstm_16_256_20_1_30 > ../Analysis/logs/eval/step3/Lstm_16_256_20_1_30.txt
#
## 1 Capa, 32-16
#
#echo "Training model Lstm_32_16_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_1_1 --model_name Lstm_32_16_1_1_30 > ../Analysis/logs/train/step3/Lstm_32_16_1_1_30.txt
#
#echo "Evaluating model Lstm_32_16_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_1_1 --model_name Lstm_32_16_1_1_30 > ../Analysis/logs/eval/step3/Lstm_32_16_1_1_30.txt
#
#echo "Training model Lstm_32_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_2_1 --model_name Lstm_32_16_2_1_30 > ../Analysis/logs/train/step3/Lstm_32_16_2_1_30.txt
#
#echo "Evaluating model Lstm_32_16_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_2_1 --model_name Lstm_32_16_2_1_30 > ../Analysis/logs/eval/step3/Lstm_32_16_2_1_30.txt
#
#echo "Training model Lstm_32_16_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_5_1 --model_name Lstm_32_16_5_1_30 > ../Analysis/logs/train/step3/Lstm_32_16_5_1_30.txt
#
#echo "Evaluating model Lstm_32_16_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_5_1 --model_name Lstm_32_16_5_1_30 > ../Analysis/logs/eval/step3/Lstm_32_16_5_1_30.txt
#
#echo "Training model Lstm_32_16_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_10_1 --model_name Lstm_32_16_10_1_30 > ../Analysis/logs/train/step3/Lstm_32_16_10_1_30.txt
#
#echo "Evaluating model Lstm_32_16_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_10_1 --model_name Lstm_32_16_10_1_30 > ../Analysis/logs/eval/step3/Lstm_32_16_10_1_30.txt
#
#echo "Training model Lstm_32_16_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_20_1 --model_name Lstm_32_16_20_1_30 > ../Analysis/logs/train/step3/Lstm_32_16_20_1_30.txt
#
#echo "Evaluating model Lstm_32_16_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_20_1 --model_name Lstm_32_16_20_1_30 > ../Analysis/logs/eval/step3/Lstm_32_16_20_1_30.txt
#
## 1 Capa, 32-32
#
#echo "Training model Lstm_32_32_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_1_1 --model_name Lstm_32_32_1_1_30 > ../Analysis/logs/train/step3/Lstm_32_32_1_1_30.txt
#
#echo "Evaluating model Lstm_32_32_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_1_1 --model_name Lstm_32_32_1_1_30 > ../Analysis/logs/eval/step3/Lstm_32_32_1_1_30.txt
#
#echo "Training model Lstm_32_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_30 > ../Analysis/logs/train/step3/Lstm_32_32_2_1_30.txt
#
#echo "Evaluating model Lstm_32_32_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_30 > ../Analysis/logs/eval/step3/Lstm_32_32_2_1_30.txt
#
#echo "Training model Lstm_32_32_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_5_1 --model_name Lstm_32_32_5_1_30 > ../Analysis/logs/train/step3/Lstm_32_32_5_1_30.txt
#
#echo "Evaluating model Lstm_32_32_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_5_1 --model_name Lstm_32_32_5_1_30 > ../Analysis/logs/eval/step3/Lstm_32_32_5_1_30.txt
#
#echo "Training model Lstm_32_32_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_10_1 --model_name Lstm_32_32_10_1_30 > ../Analysis/logs/train/step3/Lstm_32_32_10_1_30.txt
#
#echo "Evaluating model Lstm_32_32_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_10_1 --model_name Lstm_32_32_10_1_30 > ../Analysis/logs/eval/step3/Lstm_32_32_10_1_30.txt
#
#echo "Training model Lstm_32_32_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_20_1 --model_name Lstm_32_32_20_1_30 > ../Analysis/logs/train/step3/Lstm_32_32_20_1_30.txt
#
#echo "Evaluating model Lstm_32_32_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_20_1 --model_name Lstm_32_32_20_1_30 > ../Analysis/logs/eval/step3/Lstm_32_32_20_1_30.txt
#
## 1 Capa, 32-64
#
#echo "Training model Lstm_32_64_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_30 > ../Analysis/logs/train/step3/Lstm_32_64_1_1_30.txt
#
#echo "Evaluating model Lstm_32_64_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_30 > ../Analysis/logs/eval/step3/Lstm_32_64_1_1_30.txt
#
#echo "Training model Lstm_32_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_2_1 --model_name Lstm_32_64_2_1_30 > ../Analysis/logs/train/step3/Lstm_32_64_2_1_30.txt
#
#echo "Evaluating model Lstm_32_64_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_2_1 --model_name Lstm_32_64_2_1_30 > ../Analysis/logs/eval/step3/Lstm_32_64_2_1_30.txt
#
#echo "Training model Lstm_32_64_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_5_1 --model_name Lstm_32_64_5_1_30 > ../Analysis/logs/train/step3/Lstm_32_64_5_1_30.txt
#
#echo "Evaluating model Lstm_32_64_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_5_1 --model_name Lstm_32_64_5_1_30 > ../Analysis/logs/eval/step3/Lstm_32_64_5_1_30.txt
#
#echo "Training model Lstm_32_64_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_10_1 --model_name Lstm_32_64_10_1_30 > ../Analysis/logs/train/step3/Lstm_32_64_10_1_30.txt
#
#echo "Evaluating model Lstm_32_64_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_10_1 --model_name Lstm_32_64_10_1_30 > ../Analysis/logs/eval/step3/Lstm_32_64_10_1_30.txt
#
#echo "Training model Lstm_32_64_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_20_1 --model_name Lstm_32_64_20_1_30 > ../Analysis/logs/train/step3/Lstm_32_64_20_1_30.txt
#
#echo "Evaluating model Lstm_32_64_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_20_1 --model_name Lstm_32_64_20_1_30 > ../Analysis/logs/eval/step3/Lstm_32_64_20_1_30.txt
#
## 1 Capa, 32-128
#
#echo "Training model Lstm_32_128_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_1_1 --model_name Lstm_32_128_1_1_30 > ../Analysis/logs/train/step3/Lstm_32_128_1_1_30.txt
#
#echo "Evaluating model Lstm_32_128_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_1_1 --model_name Lstm_32_128_1_1_30 > ../Analysis/logs/eval/step3/Lstm_32_128_1_1_30.txt
#
#echo "Training model Lstm_32_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_2_1 --model_name Lstm_32_128_2_1_30 > ../Analysis/logs/train/step3/Lstm_32_128_2_1_30.txt
#
#echo "Evaluating model Lstm_32_128_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_2_1 --model_name Lstm_32_128_2_1_30 > ../Analysis/logs/eval/step3/Lstm_32_128_2_1_30.txt
#
#echo "Training model Lstm_32_128_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_5_1 --model_name Lstm_32_128_5_1_30 > ../Analysis/logs/train/step3/Lstm_32_128_5_1_30.txt
#
#echo "Evaluating model Lstm_32_128_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_5_1 --model_name Lstm_32_128_5_1_30 > ../Analysis/logs/eval/step3/Lstm_32_128_5_1_30.txt
#
#echo "Training model Lstm_32_128_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_10_1 --model_name Lstm_32_128_10_1_30 > ../Analysis/logs/train/step3/Lstm_32_128_10_1_30.txt
#
#echo "Evaluating model Lstm_32_128_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_10_1 --model_name Lstm_32_128_10_1_30 > ../Analysis/logs/eval/step3/Lstm_32_128_10_1_30.txt
#
#echo "Training model Lstm_32_128_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_20_1 --model_name Lstm_32_128_20_1_30 > ../Analysis/logs/train/step3/Lstm_32_128_20_1_30.txt
#
#echo "Evaluating model Lstm_32_128_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_20_1 --model_name Lstm_32_128_20_1_30 > ../Analysis/logs/eval/step3/Lstm_32_128_20_1_30.txt
#
## 1 Capa, 32-256
#
#echo "Training model Lstm_32_256_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_1_1 --model_name Lstm_32_256_1_1_30 > ../Analysis/logs/train/step3/Lstm_32_256_1_1_30.txt
#
#echo "Evaluating model Lstm_32_256_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_1_1 --model_name Lstm_32_256_1_1_30 > ../Analysis/logs/eval/step3/Lstm_32_256_1_1_30.txt
#
#echo "Training model Lstm_32_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_2_1 --model_name Lstm_32_256_2_1_30 > ../Analysis/logs/train/step3/Lstm_32_256_2_1_30.txt
#
#echo "Evaluating model Lstm_32_256_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_2_1 --model_name Lstm_32_256_2_1_30 > ../Analysis/logs/eval/step3/Lstm_32_256_2_1_30.txt
#
#echo "Training model Lstm_32_256_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_5_1 --model_name Lstm_32_256_5_1_30 > ../Analysis/logs/train/step3/Lstm_32_256_5_1_30.txt
#
#echo "Evaluating model Lstm_32_256_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_5_1 --model_name Lstm_32_256_5_1_30 > ../Analysis/logs/eval/step3/Lstm_32_256_5_1_30.txt
#
#echo "Training model Lstm_32_256_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_10_1 --model_name Lstm_32_256_10_1_30 > ../Analysis/logs/train/step3/Lstm_32_256_10_1_30.txt
#
#echo "Evaluating model Lstm_32_256_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_10_1 --model_name Lstm_32_256_10_1_30 > ../Analysis/logs/eval/step3/Lstm_32_256_10_1_30.txt
#
#echo "Training model Lstm_32_256_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_20_1 --model_name Lstm_32_256_20_1_30 > ../Analysis/logs/train/step3/Lstm_32_256_20_1_30.txt
#
#echo "Evaluating model Lstm_32_256_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_20_1 --model_name Lstm_32_256_20_1_30 > ../Analysis/logs/eval/step3/Lstm_32_256_20_1_30.txt
#
## 1 capa, 64-16
#
#echo "Training model Lstm_64_16_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_1_1 --model_name Lstm_64_16_1_1_30 > ../Analysis/logs/train/step3/Lstm_64_16_1_1_30.txt
#
#echo "Evaluating model Lstm_64_16_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_1_1 --model_name Lstm_64_16_1_1_30 > ../Analysis/logs/eval/step3/Lstm_64_16_1_1_30.txt
#
#echo "Training model Lstm_64_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_2_1 --model_name Lstm_64_16_2_1_30 > ../Analysis/logs/train/step3/Lstm_64_16_2_1_30.txt
#
#echo "Evaluating model Lstm_64_16_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_2_1 --model_name Lstm_64_16_2_1_30 > ../Analysis/logs/eval/step3/Lstm_64_16_2_1_30.txt
#
#echo "Training model Lstm_64_16_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_5_1 --model_name Lstm_64_16_5_1_30 > ../Analysis/logs/train/step3/Lstm_64_16_5_1_30.txt
#
#echo "Evaluating model Lstm_64_16_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_5_1 --model_name Lstm_64_16_5_1_30 > ../Analysis/logs/eval/step3/Lstm_64_16_5_1_30.txt
#
#echo "Training model Lstm_64_16_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_10_1 --model_name Lstm_64_16_10_1_30 > ../Analysis/logs/train/step3/Lstm_64_16_10_1_30.txt
#
#echo "Evaluating model Lstm_64_16_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_10_1 --model_name Lstm_64_16_10_1_30 > ../Analysis/logs/eval/step3/Lstm_64_16_10_1_30.txt
#
#echo "Training model Lstm_64_16_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_20_1 --model_name Lstm_64_16_20_1_30 > ../Analysis/logs/train/step3/Lstm_64_16_20_1_30.txt
#
#echo "Evaluating model Lstm_64_16_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_20_1 --model_name Lstm_64_16_20_1_30 > ../Analysis/logs/eval/step3/Lstm_64_16_20_1_30.txt
#
## 1 Capa, 64-32
#
#echo "Training model Lstm_64_32_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_1_1 --model_name Lstm_64_32_1_1_30 > ../Analysis/logs/train/step3/Lstm_64_32_1_1_30.txt
#
#echo "Evaluating model Lstm_64_32_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_1_1 --model_name Lstm_64_32_1_1_30 > ../Analysis/logs/eval/step3/Lstm_64_32_1_1_30.txt
#
#echo "Training model Lstm_64_32_2_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_30 > ../Analysis/logs/train/step3/Lstm_64_32_2_1_30.txt
#
#echo "Evaluating model Lstm_64_32_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_30 > ../Analysis/logs/eval/step3/Lstm_64_32_2_1_30.txt
#
#echo "Training model Lstm_64_32_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_5_1 --model_name Lstm_64_32_5_1_30 > ../Analysis/logs/train/step3/Lstm_64_32_5_1_30.txt
#
#echo "Evaluating model Lstm_64_32_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_5_1 --model_name Lstm_64_32_5_1_30 > ../Analysis/logs/eval/step3/Lstm_64_32_5_1_30.txt
#
#echo "Training model Lstm_64_32_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_10_1 --model_name Lstm_64_32_10_1_30 > ../Analysis/logs/train/step3/Lstm_64_32_10_1_30.txt
#
#echo "Evaluating model Lstm_64_32_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_10_1 --model_name Lstm_64_32_10_1_30 > ../Analysis/logs/eval/step3/Lstm_64_32_10_1_30.txt
#
#echo "Training model Lstm_64_32_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_20_1 --model_name Lstm_64_32_20_1_30 > ../Analysis/logs/train/step3/Lstm_64_32_20_1_30.txt
#
#echo "Evaluating model Lstm_64_32_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_20_1 --model_name Lstm_64_32_20_1_30 > ../Analysis/logs/eval/step3/Lstm_64_32_20_1_30.txt
#
## 1 Capa, 64-64
#
#echo "Training model Lstm_64_64_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_1_1 --model_name Lstm_64_64_1_1_30 > ../Analysis/logs/train/step3/Lstm_64_64_1_1_30.txt
#
#echo "Evaluating model Lstm_64_64_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_1_1 --model_name Lstm_64_64_1_1_30 > ../Analysis/logs/eval/step3/Lstm_64_64_1_1_30.txt
#
#echo "Training model Lstm_64_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_2_1 --model_name Lstm_64_64_2_1_30 > ../Analysis/logs/train/step3/Lstm_64_64_2_1_30.txt
#
#echo "Evaluating model Lstm_64_64_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_2_1 --model_name Lstm_64_64_2_1_30 > ../Analysis/logs/eval/step3/Lstm_64_64_2_1_30.txt
#
#echo "Training model Lstm_64_64_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_30 > ../Analysis/logs/train/step3/Lstm_64_64_5_1_30.txt
#
#echo "Evaluating model Lstm_64_64_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_30 > ../Analysis/logs/eval/step3/Lstm_64_64_5_1_30.txt
#
#echo "Training model Lstm_64_64_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_10_1 --model_name Lstm_64_64_10_1_30 > ../Analysis/logs/train/step3/Lstm_64_64_10_1_30.txt
#
#echo "Evaluating model Lstm_64_64_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_10_1 --model_name Lstm_64_64_10_1_30 > ../Analysis/logs/eval/step3/Lstm_64_64_10_1_30.txt

echo "Training model Lstm_64_64_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --patience 30 --model_folder step3 \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier Lstm_64_64_20_1 --model_name Lstm_64_64_20_1_30 > ../Analysis/logs/train/step3/Lstm_64_64_20_1_30.txt

echo "Evaluating model Lstm_64_64_20_1_30"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier Lstm_64_64_20_1 --model_name Lstm_64_64_20_1_30 > ../Analysis/logs/eval/step3/Lstm_64_64_20_1_30.txt

## 1 Capa, 64-128
#
#echo "Training model Lstm_64_128_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_1_1 --model_name Lstm_64_128_1_1_30 > ../Analysis/logs/train/step3/Lstm_64_128_1_1_30.txt
#
#echo "Evaluating model Lstm_64_128_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_1_1 --model_name Lstm_64_128_1_1_30 > ../Analysis/logs/eval/step3/Lstm_64_128_1_1_30.txt
#
#echo "Training model Lstm_64_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_2_1 --model_name Lstm_64_128_2_1_30 > ../Analysis/logs/train/step3/Lstm_64_128_2_1_30.txt
#
#echo "Evaluating model Lstm_64_128_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_2_1 --model_name Lstm_64_128_2_1_30 > ../Analysis/logs/eval/step3/Lstm_64_128_2_1_30.txt
#
#echo "Training model Lstm_64_128_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_5_1 --model_name Lstm_64_128_5_1_30 > ../Analysis/logs/train/step3/Lstm_64_128_5_1_30.txt
#
#echo "Evaluating model Lstm_64_128_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_5_1 --model_name Lstm_64_128_5_1_30 > ../Analysis/logs/eval/step3/Lstm_64_128_5_1_30.txt
#
#echo "Training model Lstm_64_128_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_10_1 --model_name Lstm_64_128_10_1_30 > ../Analysis/logs/train/step3/Lstm_64_128_10_1_30.txt
#
#echo "Evaluating model Lstm_64_128_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_10_1 --model_name Lstm_64_128_10_1_30 > ../Analysis/logs/eval/step3/Lstm_64_128_10_1_30.txt
#
#echo "Training model Lstm_64_128_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_20_1 --model_name Lstm_64_128_20_1_30 > ../Analysis/logs/train/step3/Lstm_64_128_20_1_30.txt
#
#echo "Evaluating model Lstm_64_128_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_20_1 --model_name Lstm_64_128_20_1_30 > ../Analysis/logs/eval/step3/Lstm_64_128_20_1_30.txt
#
## 1 Capa, 64-256
#
#echo "Training model Lstm_64_256_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_1_1 --model_name Lstm_64_256_1_1_30 > ../Analysis/logs/train/step3/Lstm_64_256_1_1_30.txt
#
#echo "Evaluating model Lstm_64_256_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_1_1 --model_name Lstm_64_256_1_1_30 > ../Analysis/logs/eval/step3/Lstm_64_256_1_1_30.txt
#
#echo "Training model Lstm_64_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_2_1 --model_name Lstm_64_256_2_1_30 > ../Analysis/logs/train/step3/Lstm_64_256_2_1_30.txt
#
#echo "Evaluating model Lstm_64_256_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_2_1 --model_name Lstm_64_256_2_1_30 > ../Analysis/logs/eval/step3/Lstm_64_256_2_1_30.txt
#
#echo "Training model Lstm_64_256_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_5_1 --model_name Lstm_64_256_5_1_30 > ../Analysis/logs/train/step3/Lstm_64_256_5_1_30.txt
#
#echo "Evaluating model Lstm_64_256_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_5_1 --model_name Lstm_64_256_5_1_30 > ../Analysis/logs/eval/step3/Lstm_64_256_5_1_30.txt
#
#echo "Training model Lstm_64_256_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_10_1 --model_name Lstm_64_256_10_1_30 > ../Analysis/logs/train/step3/Lstm_64_256_10_1_30.txt
#
#echo "Evaluating model Lstm_64_256_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_10_1 --model_name Lstm_64_256_10_1_30 > ../Analysis/logs/eval/step3/Lstm_64_256_10_1_30.txt
#
#echo "Training model Lstm_64_256_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_20_1 --model_name Lstm_64_256_20_1_30 > ../Analysis/logs/train/step3/Lstm_64_256_20_1_30.txt
#
#echo "Evaluating model Lstm_64_256_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_20_1 --model_name Lstm_64_256_20_1_30 > ../Analysis/logs/eval/step3/Lstm_64_256_20_1_30.txt
#
## 1 capa, 128-16
#
#echo "Training model Lstm_128_16_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_1_1 --model_name Lstm_128_16_1_1_30 > ../Analysis/logs/train/step3/Lstm_128_16_1_1_30.txt
#
#echo "Evaluating model Lstm_128_16_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_1_1 --model_name Lstm_128_16_1_1_30 > ../Analysis/logs/eval/step3/Lstm_128_16_1_1_30.txt
#
#echo "Training model Lstm_128_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_2_1 --model_name Lstm_128_16_2_1_30 > ../Analysis/logs/train/step3/Lstm_128_16_2_1_30.txt
#
#echo "Evaluating model Lstm_128_16_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_2_1 --model_name Lstm_128_16_2_1_30 > ../Analysis/logs/eval/step3/Lstm_128_16_2_1_30.txt
#
#echo "Training model Lstm_128_16_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_5_1 --model_name Lstm_128_16_5_1_30 > ../Analysis/logs/train/step3/Lstm_128_16_5_1_30.txt
#
#echo "Evaluating model Lstm_128_16_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_5_1 --model_name Lstm_128_16_5_1_30 > ../Analysis/logs/eval/step3/Lstm_128_16_5_1_30.txt
#
#echo "Training model Lstm_128_16_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_10_1 --model_name Lstm_128_16_10_1_30 > ../Analysis/logs/train/step3/Lstm_128_16_10_1_30.txt
#
#echo "Evaluating model Lstm_128_16_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_10_1 --model_name Lstm_128_16_10_1_30 > ../Analysis/logs/eval/step3/Lstm_128_16_10_1_30.txt
#
#echo "Training model Lstm_128_16_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_20_1 --model_name Lstm_128_16_20_1_30 > ../Analysis/logs/train/step3/Lstm_128_16_20_1_30.txt
#
#echo "Evaluating model Lstm_128_16_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_20_1 --model_name Lstm_128_16_20_1_30 > ../Analysis/logs/eval/step3/Lstm_128_16_20_1_30.txt
#
## 1 Capa, 128-32
#
#echo "Training model Lstm_128_32_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_30 > ../Analysis/logs/train/step3/Lstm_128_32_1_1_30.txt
#
#echo "Evaluating model Lstm_128_32_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_30 > ../Analysis/logs/eval/step3/Lstm_128_32_1_1_30.txt
#
#echo "Training model Lstm_128_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_30 > ../Analysis/logs/train/step3/Lstm_128_32_2_1_30.txt
#
#echo "Evaluating model Lstm_128_32_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_30 > ../Analysis/logs/eval/step3/Lstm_128_32_2_1_30.txt
#
#echo "Training model Lstm_128_32_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_5_1 --model_name Lstm_128_32_5_1_30 > ../Analysis/logs/train/step3/Lstm_128_32_5_1_30.txt
#
#echo "Evaluating model Lstm_128_32_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_5_1 --model_name Lstm_128_32_5_1_30 > ../Analysis/logs/eval/step3/Lstm_128_32_5_1_30.txt
#
#echo "Training model Lstm_128_32_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_10_1 --model_name Lstm_128_32_10_1_30 > ../Analysis/logs/train/step3/Lstm_128_32_10_1_30.txt
#
#echo "Evaluating model Lstm_128_32_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_10_1 --model_name Lstm_128_32_10_1_30 > ../Analysis/logs/eval/step3/Lstm_128_32_10_1_30.txt
#
#echo "Training model Lstm_128_32_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_20_1 --model_name Lstm_128_32_20_1_30 > ../Analysis/logs/train/step3/Lstm_128_32_20_1_30.txt
#
#echo "Evaluating model Lstm_128_32_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_20_1 --model_name Lstm_128_32_20_1_30 > ../Analysis/logs/eval/step3/Lstm_128_32_20_1_30.txt
#
## 1 Capa, 128-64
#
#echo "Training model Lstm_128_64_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_1_1 --model_name Lstm_128_64_1_1_30 > ../Analysis/logs/train/step3/Lstm_128_64_1_1_30.txt
#
#echo "Evaluating model Lstm_128_64_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_1_1 --model_name Lstm_128_64_1_1_30 > ../Analysis/logs/eval/step3/Lstm_128_64_1_1_30.txt
#
#echo "Training model Lstm_128_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_2_1 --model_name Lstm_128_64_2_1_30 > ../Analysis/logs/train/step3/Lstm_128_64_2_1_30.txt
#
#echo "Evaluating model Lstm_128_64_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_2_1 --model_name Lstm_128_64_2_1_30 > ../Analysis/logs/eval/step3/Lstm_128_64_2_1_30.txt
#
#echo "Training model Lstm_128_64_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_5_1 --model_name Lstm_128_64_5_1_30 > ../Analysis/logs/train/step3/Lstm_128_64_5_1_30.txt
#
#echo "Evaluating model Lstm_128_64_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_5_1 --model_name Lstm_128_64_5_1_30 > ../Analysis/logs/eval/step3/Lstm_128_64_5_1_30.txt
#
#echo "Training model Lstm_128_64_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_10_1 --model_name Lstm_128_64_10_1_30 > ../Analysis/logs/train/step3/Lstm_128_64_10_1_30.txt
#
#echo "Evaluating model Lstm_128_64_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_10_1 --model_name Lstm_128_64_10_1_30 > ../Analysis/logs/eval/step3/Lstm_128_64_10_1_30.txt
#
#echo "Training model Lstm_128_64_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_20_1 --model_name Lstm_128_64_20_1_30 > ../Analysis/logs/train/step3/Lstm_128_64_20_1_30.txt
#
#echo "Evaluating model Lstm_128_64_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_20_1 --model_name Lstm_128_64_20_1_30 > ../Analysis/logs/eval/step3/Lstm_128_64_20_1_30.txt
#
## 1 Capa, 128-128
#
#echo "Training model Lstm_128_128_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_1_1 --model_name Lstm_128_128_1_1_30 > ../Analysis/logs/train/step3/Lstm_128_128_1_1_30.txt
#
#echo "Evaluating model Lstm_128_128_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_1_1 --model_name Lstm_128_128_1_1_30 > ../Analysis/logs/eval/step3/Lstm_128_128_1_1_30.txt
#
#echo "Training model Lstm_128_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_2_1 --model_name Lstm_128_128_2_1_30 > ../Analysis/logs/train/step3/Lstm_128_128_2_1_30.txt
#
#echo "Evaluating model Lstm_128_128_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_2_1 --model_name Lstm_128_128_2_1_30 > ../Analysis/logs/eval/step3/Lstm_128_128_2_1_30.txt
#
#echo "Training model Lstm_128_128_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_5_1 --model_name Lstm_128_128_5_1_30 > ../Analysis/logs/train/step3/Lstm_128_128_5_1_30.txt
#
#echo "Evaluating model Lstm_128_128_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_5_1 --model_name Lstm_128_128_5_1_30 > ../Analysis/logs/eval/step3/Lstm_128_128_5_1_30.txt
#
#echo "Training model Lstm_128_128_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_10_1 --model_name Lstm_128_128_10_1_30 > ../Analysis/logs/train/step3/Lstm_128_128_10_1_30.txt
#
#echo "Evaluating model Lstm_128_128_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_10_1 --model_name Lstm_128_128_10_1_30 > ../Analysis/logs/eval/step3/Lstm_128_128_10_1_30.txt
#
#echo "Training model Lstm_128_128_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_20_1 --model_name Lstm_128_128_20_1_30 > ../Analysis/logs/train/step3/Lstm_128_128_20_1_30.txt
#
#echo "Evaluating model Lstm_128_128_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_20_1 --model_name Lstm_128_128_20_1_30 > ../Analysis/logs/eval/step3/Lstm_128_128_20_1_30.txt
#
## 1 Capa, 128-256
#
#echo "Training model Lstm_128_256_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_1_1 --model_name Lstm_128_256_1_1_30 > ../Analysis/logs/train/step3/Lstm_128_256_1_1_30.txt
#
#echo "Evaluating model Lstm_128_256_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_1_1 --model_name Lstm_128_256_1_1_30 > ../Analysis/logs/eval/step3/Lstm_128_256_1_1_30.txt
#
#echo "Training model Lstm_128_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_2_1 --model_name Lstm_128_256_2_1_30 > ../Analysis/logs/train/step3/Lstm_128_256_2_1_30.txt
#
#echo "Evaluating model Lstm_128_256_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_2_1 --model_name Lstm_128_256_2_1_30 > ../Analysis/logs/eval/step3/Lstm_128_256_2_1_30.txt
#
#echo "Training model Lstm_128_256_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_5_1 --model_name Lstm_128_256_5_1_30 > ../Analysis/logs/train/step3/Lstm_128_256_5_1_30.txt
#
#echo "Evaluating model Lstm_128_256_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_5_1 --model_name Lstm_128_256_5_1_30 > ../Analysis/logs/eval/step3/Lstm_128_256_5_1_30.txt
#
#echo "Training model Lstm_128_256_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_10_1 --model_name Lstm_128_256_10_1_30 > ../Analysis/logs/train/step3/Lstm_128_256_10_1_30.txt
#
#echo "Evaluating model Lstm_128_256_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_10_1 --model_name Lstm_128_256_10_1_30 > ../Analysis/logs/eval/step3/Lstm_128_256_10_1_30.txt
#
#echo "Training model Lstm_128_256_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_20_1 --model_name Lstm_128_256_20_1_30 > ../Analysis/logs/train/step3/Lstm_128_256_20_1_30.txt
#
#echo "Evaluating model Lstm_128_256_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_20_1 --model_name Lstm_128_256_20_1_30 > ../Analysis/logs/eval/step3/Lstm_128_256_20_1_30.txt
#
## 1 capa, 256-16
#
#echo "Training model Lstm_256_16_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_1_1 --model_name Lstm_256_16_1_1_30 > ../Analysis/logs/train/step3/Lstm_256_16_1_1_30.txt
#
#echo "Evaluating model Lstm_256_16_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_1_1 --model_name Lstm_256_16_1_1_30 > ../Analysis/logs/eval/step3/Lstm_256_16_1_1_30.txt
#
#echo "Training model Lstm_256_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_2_1 --model_name Lstm_256_16_2_1_30 > ../Analysis/logs/train/step3/Lstm_256_16_2_1_30.txt
#
#echo "Evaluating model Lstm_256_16_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_2_1 --model_name Lstm_256_16_2_1_30 > ../Analysis/logs/eval/step3/Lstm_256_16_2_1_30.txt
#
#echo "Training model Lstm_256_16_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_5_1 --model_name Lstm_256_16_5_1_30 > ../Analysis/logs/train/step3/Lstm_256_16_5_1_30.txt
#
#echo "Evaluating model Lstm_256_16_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_5_1 --model_name Lstm_256_16_5_1_30 > ../Analysis/logs/eval/step3/Lstm_256_16_5_1_30.txt
#
#echo "Training model Lstm_256_16_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_10_1 --model_name Lstm_256_16_10_1_30 > ../Analysis/logs/train/step3/Lstm_256_16_10_1_30.txt
#
#echo "Evaluating model Lstm_256_16_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_10_1 --model_name Lstm_256_16_10_1_30 > ../Analysis/logs/eval/step3/Lstm_256_16_10_1_30.txt
#
#echo "Training model Lstm_256_16_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_20_1 --model_name Lstm_256_16_20_1_30 > ../Analysis/logs/train/step3/Lstm_256_16_20_1_30.txt
#
#echo "Evaluating model Lstm_256_16_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_20_1 --model_name Lstm_256_16_20_1_30 > ../Analysis/logs/eval/step3/Lstm_256_16_20_1_30.txt
#
## 1 Capa, 256-32
#
#echo "Training model Lstm_256_32_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_1_1 --model_name Lstm_256_32_1_1_30 > ../Analysis/logs/train/step3/Lstm_256_32_1_1_30.txt
#
#echo "Evaluating model Lstm_256_32_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_1_1 --model_name Lstm_256_32_1_1_30 > ../Analysis/logs/eval/step3/Lstm_256_32_1_1_30.txt
#
#echo "Training model Lstm_256_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_2_1 --model_name Lstm_256_32_2_1_30 > ../Analysis/logs/train/step3/Lstm_256_32_2_1_30.txt
#
#echo "Evaluating model Lstm_256_32_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_2_1 --model_name Lstm_256_32_2_1_30 > ../Analysis/logs/eval/step3/Lstm_256_32_2_1_30.txt
#
#echo "Training model Lstm_256_32_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_5_1 --model_name Lstm_256_32_5_1_30 > ../Analysis/logs/train/step3/Lstm_256_32_5_1_30.txt
#
#echo "Evaluating model Lstm_256_32_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_5_1 --model_name Lstm_256_32_5_1_30 > ../Analysis/logs/eval/step3/Lstm_256_32_5_1_30.txt
#
#echo "Training model Lstm_256_32_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_10_1 --model_name Lstm_256_32_10_1_30 > ../Analysis/logs/train/step3/Lstm_256_32_10_1_30.txt
#
#echo "Evaluating model Lstm_256_32_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_10_1 --model_name Lstm_256_32_10_1_30 > ../Analysis/logs/eval/step3/Lstm_256_32_10_1_30.txt
#
#echo "Training model Lstm_256_32_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_20_1 --model_name Lstm_256_32_20_1_30 > ../Analysis/logs/train/step3/Lstm_256_32_20_1_30.txt
#
#echo "Evaluating model Lstm_256_32_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_20_1 --model_name Lstm_256_32_20_1_30 > ../Analysis/logs/eval/step3/Lstm_256_32_20_1_30.txt
#
## 1 Capa, 256-64
#
#echo "Training model Lstm_256_64_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_1_1 --model_name Lstm_256_64_1_1_30 > ../Analysis/logs/train/step3/Lstm_256_64_1_1_30.txt
#
#echo "Evaluating model Lstm_256_64_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_1_1 --model_name Lstm_256_64_1_1_30 > ../Analysis/logs/eval/step3/Lstm_256_64_1_1_30.txt
#
#echo "Training model Lstm_256_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_2_1 --model_name Lstm_256_64_2_1_30 > ../Analysis/logs/train/step3/Lstm_256_64_2_1_30.txt
#
#echo "Evaluating model Lstm_256_64_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_2_1 --model_name Lstm_256_64_2_1_30 > ../Analysis/logs/eval/step3/Lstm_256_64_2_1_30.txt
#
#echo "Training model Lstm_256_64_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_5_1 --model_name Lstm_256_64_5_1_30 > ../Analysis/logs/train/step3/Lstm_256_64_5_1_30.txt
#
#echo "Evaluating model Lstm_256_64_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_5_1 --model_name Lstm_256_64_5_1_30 > ../Analysis/logs/eval/step3/Lstm_256_64_5_1_30.txt
#
#echo "Training model Lstm_256_64_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_10_1 --model_name Lstm_256_64_10_1_30 > ../Analysis/logs/train/step3/Lstm_256_64_10_1_30.txt
#
#echo "Evaluating model Lstm_256_64_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_10_1 --model_name Lstm_256_64_10_1_30 > ../Analysis/logs/eval/step3/Lstm_256_64_10_1_30.txt
#
#echo "Training model Lstm_256_64_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_20_1 --model_name Lstm_256_64_20_1_30 > ../Analysis/logs/train/step3/Lstm_256_64_20_1_30.txt
#
#echo "Evaluating model Lstm_256_64_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_20_1 --model_name Lstm_256_64_20_1_30 > ../Analysis/logs/eval/step3/Lstm_256_64_20_1_30.txt
#
## 1 Capa, 256-128
#
#echo "Training model Lstm_256_128_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_1_1 --model_name Lstm_256_128_1_1_30 > ../Analysis/logs/train/step3/Lstm_256_128_1_1_30.txt
#
#echo "Evaluating model Lstm_256_128_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_1_1 --model_name Lstm_256_128_1_1_30 > ../Analysis/logs/eval/step3/Lstm_256_128_1_1_30.txt
#
#echo "Training model Lstm_256_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_2_1 --model_name Lstm_256_128_2_1_30 > ../Analysis/logs/train/step3/Lstm_256_128_2_1_30.txt
#
#echo "Evaluating model Lstm_256_128_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_2_1 --model_name Lstm_256_128_2_1_30 > ../Analysis/logs/eval/step3/Lstm_256_128_2_1_30.txt
#
#echo "Training model Lstm_256_128_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_5_1 --model_name Lstm_256_128_5_1_30 > ../Analysis/logs/train/step3/Lstm_256_128_5_1_30.txt
#
#echo "Evaluating model Lstm_256_128_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_5_1 --model_name Lstm_256_128_5_1_30 > ../Analysis/logs/eval/step3/Lstm_256_128_5_1_30.txt
#
#echo "Training model Lstm_256_128_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_30 > ../Analysis/logs/train/step3/Lstm_256_128_10_1_30.txt
#
#echo "Evaluating model Lstm_256_128_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_30 > ../Analysis/logs/eval/step3/Lstm_256_128_10_1_30.txt
#
#echo "Training model Lstm_256_128_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_20_1 --model_name Lstm_256_128_20_1_30 > ../Analysis/logs/train/step3/Lstm_256_128_20_1_30.txt
#
#echo "Evaluating model Lstm_256_128_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_20_1 --model_name Lstm_256_128_20_1_30 > ../Analysis/logs/eval/step3/Lstm_256_128_20_1_30.txt
#
## 1 Capa, 256-256
#
#echo "Training model Lstm_256_256_1_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_1_1 --model_name Lstm_256_256_1_1_30 > ../Analysis/logs/train/step3/Lstm_256_256_1_1_30.txt
#
#echo "Evaluating model Lstm_256_256_1_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_1_1 --model_name Lstm_256_256_1_1_30 > ../Analysis/logs/eval/step3/Lstm_256_256_1_1_30.txt
#
#echo "Training model Lstm_256_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_2_1 --model_name Lstm_256_256_2_1_30 > ../Analysis/logs/train/step3/Lstm_256_256_2_1_30.txt
#
#echo "Evaluating model Lstm_256_256_2_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_2_1 --model_name Lstm_256_256_2_1_30 > ../Analysis/logs/eval/step3/Lstm_256_256_2_1_30.txt
#
#echo "Training model Lstm_256_256_5_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_5_1 --model_name Lstm_256_256_5_1_30 > ../Analysis/logs/train/step3/Lstm_256_256_5_1_30.txt
#
#echo "Evaluating model Lstm_256_256_5_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_5_1 --model_name Lstm_256_256_5_1_30 > ../Analysis/logs/eval/step3/Lstm_256_256_5_1_30.txt
#
#echo "Training model Lstm_256_256_10_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_10_1 --model_name Lstm_256_256_10_1_30 > ../Analysis/logs/train/step3/Lstm_256_256_10_1_30.txt
#
#echo "Evaluating model Lstm_256_256_10_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_10_1 --model_name Lstm_256_256_10_1_30 > ../Analysis/logs/eval/step3/Lstm_256_256_10_1_30.txt
#
#echo "Training model Lstm_256_256_20_1_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_20_1 --model_name Lstm_256_256_20_1_30 > ../Analysis/logs/train/step3/Lstm_256_256_20_1_30.txt
#
#echo "Evaluating model Lstm_256_256_20_1_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_20_1 --model_name Lstm_256_256_20_1_30 > ../Analysis/logs/eval/step3/Lstm_256_256_20_1_30.txt
#
## 2 capas, 16-16
#
#echo "Training model Lstm_16_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_30 > ../Analysis/logs/train/step3/Lstm_16_16_1_2_30.txt
#
#echo "Evaluating model Lstm_16_16_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_30 > ../Analysis/logs/eval/step3/Lstm_16_16_1_2_30.txt
#
#echo "Training model Lstm_16_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_2_2 --model_name Lstm_16_16_2_2_30 > ../Analysis/logs/train/step3/Lstm_16_16_2_2_30.txt
#
#echo "Evaluating model Lstm_16_16_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_2_2 --model_name Lstm_16_16_2_2_30 > ../Analysis/logs/eval/step3/Lstm_16_16_2_2_30.txt
#
#echo "Training model Lstm_16_16_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_5_2 --model_name Lstm_16_16_5_2_30 > ../Analysis/logs/train/step3/Lstm_16_16_5_2_30.txt
#
#echo "Evaluating model Lstm_16_16_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_5_2 --model_name Lstm_16_16_5_2_30 > ../Analysis/logs/eval/step3/Lstm_16_16_5_2_30.txt
#
#echo "Training model Lstm_16_16_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_10_2 --model_name Lstm_16_16_10_2_30 > ../Analysis/logs/train/step3/Lstm_16_16_10_2_30.txt
#
#echo "Evaluating model Lstm_16_16_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_10_2 --model_name Lstm_16_16_10_2_30 > ../Analysis/logs/eval/step3/Lstm_16_16_10_2_30.txt
#
#echo "Training model Lstm_16_16_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_16_20_2 --model_name Lstm_16_16_20_2_30 > ../Analysis/logs/train/step3/Lstm_16_16_20_2_30.txt
#
#echo "Evaluating model Lstm_16_16_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_16_20_2 --model_name Lstm_16_16_20_2_30 > ../Analysis/logs/eval/step3/Lstm_16_16_20_2_30.txt
#
## 2 capas, 16-32
#
#echo "Training model Lstm_16_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_1_2 --model_name Lstm_16_32_1_2_30 > ../Analysis/logs/train/step3/Lstm_16_32_1_2_30.txt
#
#echo "Evaluating model Lstm_16_32_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_1_2 --model_name Lstm_16_32_1_2_30 > ../Analysis/logs/eval/step3/Lstm_16_32_1_2_30.txt
#
#echo "Training model Lstm_16_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_2_2 --model_name Lstm_16_32_2_2_30 > ../Analysis/logs/train/step3/Lstm_16_32_2_2_30.txt
#
#echo "Evaluating model Lstm_16_32_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_2_2 --model_name Lstm_16_32_2_2_30 > ../Analysis/logs/eval/step3/Lstm_16_32_2_2_30.txt
#
#echo "Training model Lstm_16_32_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_5_2 --model_name Lstm_16_32_5_2_30 > ../Analysis/logs/train/step3/Lstm_16_32_5_2_30.txt
#
#echo "Evaluating model Lstm_16_32_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_5_2 --model_name Lstm_16_32_5_2_30 > ../Analysis/logs/eval/step3/Lstm_16_32_5_2_30.txt
#
#echo "Training model Lstm_16_32_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_10_2 --model_name Lstm_16_32_10_2_30 > ../Analysis/logs/train/step3/Lstm_16_32_10_2_30.txt
#
#echo "Evaluating model Lstm_16_32_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_10_2 --model_name Lstm_16_32_10_2_30 > ../Analysis/logs/eval/step3/Lstm_16_32_10_2_30.txt
#
#echo "Training model Lstm_16_32_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_32_20_2 --model_name Lstm_16_32_20_2_30 > ../Analysis/logs/train/step3/Lstm_16_32_20_2_30.txt
#
#echo "Evaluating model Lstm_16_32_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_32_20_2 --model_name Lstm_16_32_20_2_30 > ../Analysis/logs/eval/step3/Lstm_16_32_20_2_30.txt
#
## 2 capas, 16-64
#
#echo "Training model Lstm_16_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_1_2 --model_name Lstm_16_64_1_2_30 > ../Analysis/logs/train/step3/Lstm_16_64_1_2_30.txt
#
#echo "Evaluating model Lstm_16_64_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_1_2 --model_name Lstm_16_64_1_2_30 > ../Analysis/logs/eval/step3/Lstm_16_64_1_2_30.txt
#
#echo "Training model Lstm_16_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_2_2 --model_name Lstm_16_64_2_2_30 > ../Analysis/logs/train/step3/Lstm_16_64_2_2_30.txt
#
#echo "Evaluating model Lstm_16_64_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_2_2 --model_name Lstm_16_64_2_2_30 > ../Analysis/logs/eval/step3/Lstm_16_64_2_2_30.txt
#
#echo "Training model Lstm_16_64_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_5_2 --model_name Lstm_16_64_5_2_30 > ../Analysis/logs/train/step3/Lstm_16_64_5_2_30.txt
#
#echo "Evaluating model Lstm_16_64_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_5_2 --model_name Lstm_16_64_5_2_30 > ../Analysis/logs/eval/step3/Lstm_16_64_5_2_30.txt
#
#echo "Training model Lstm_16_64_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_10_2 --model_name Lstm_16_64_10_2_30 > ../Analysis/logs/train/step3/Lstm_16_64_10_2_30.txt
#
#echo "Evaluating model Lstm_16_64_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_10_2 --model_name Lstm_16_64_10_2_30 > ../Analysis/logs/eval/step3/Lstm_16_64_10_2_30.txt
#
#echo "Training model Lstm_16_64_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_64_20_2 --model_name Lstm_16_64_20_2_30 > ../Analysis/logs/train/step3/Lstm_16_64_20_2_30.txt
#
#echo "Evaluating model Lstm_16_64_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_64_20_2 --model_name Lstm_16_64_20_2_30 > ../Analysis/logs/eval/step3/Lstm_16_64_20_2_30.txt
#
## 2 capas, 16-128
#
#echo "Training model Lstm_16_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_1_2 --model_name Lstm_16_128_1_2_30 > ../Analysis/logs/train/step3/Lstm_16_128_1_2_30.txt
#
#echo "Evaluating model Lstm_16_128_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_1_2 --model_name Lstm_16_128_1_2_30 > ../Analysis/logs/eval/step3/Lstm_16_128_1_2_30.txt
#
#echo "Training model Lstm_16_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_2_2 --model_name Lstm_16_128_2_2_30 > ../Analysis/logs/train/step3/Lstm_16_128_2_2_30.txt
#
#echo "Evaluating model Lstm_16_128_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_2_2 --model_name Lstm_16_128_2_2_30 > ../Analysis/logs/eval/step3/Lstm_16_128_2_2_30.txt
#
#echo "Training model Lstm_16_128_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_5_2 --model_name Lstm_16_128_5_2_30 > ../Analysis/logs/train/step3/Lstm_16_128_5_2_30.txt
#
#echo "Evaluating model Lstm_16_128_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_5_2 --model_name Lstm_16_128_5_2_30 > ../Analysis/logs/eval/step3/Lstm_16_128_5_2_30.txt
#
#echo "Training model Lstm_16_128_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_2 --model_name Lstm_16_128_10_2_30 > ../Analysis/logs/train/step3/Lstm_16_128_10_2_30.txt
#
#echo "Evaluating model Lstm_16_128_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_10_2 --model_name Lstm_16_128_10_2_30 > ../Analysis/logs/eval/step3/Lstm_16_128_10_2_30.txt
#
#echo "Training model Lstm_16_128_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_128_20_2 --model_name Lstm_16_128_20_2_30 > ../Analysis/logs/train/step3/Lstm_16_128_20_2_30.txt
#
#echo "Evaluating model Lstm_16_128_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_128_20_2 --model_name Lstm_16_128_20_2_30 > ../Analysis/logs/eval/step3/Lstm_16_128_20_2_30.txt
#
## 2 capas, 16-256
#
#echo "Training model Lstm_16_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_1_2 --model_name Lstm_16_256_1_2_30 > ../Analysis/logs/train/step3/Lstm_16_256_1_2_30.txt
#
#echo "Evaluating model Lstm_16_256_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_1_2 --model_name Lstm_16_256_1_2_30 > ../Analysis/logs/eval/step3/Lstm_16_256_1_2_30.txt
#
#echo "Training model Lstm_16_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_2_2 --model_name Lstm_16_256_2_2_30 > ../Analysis/logs/train/step3/Lstm_16_256_2_2_30.txt
#
#echo "Evaluating model Lstm_16_256_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_2_2 --model_name Lstm_16_256_2_2_30 > ../Analysis/logs/eval/step3/Lstm_16_256_2_2_30.txt
#
#echo "Training model Lstm_16_256_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_5_2 --model_name Lstm_16_256_5_2_30 > ../Analysis/logs/train/step3/Lstm_16_256_5_2_30.txt
#
#echo "Evaluating model Lstm_16_256_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_5_2 --model_name Lstm_16_256_5_2_30 > ../Analysis/logs/eval/step3/Lstm_16_256_5_2_30.txt
#
#echo "Training model Lstm_16_256_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_10_2 --model_name Lstm_16_256_10_2_30 > ../Analysis/logs/train/step3/Lstm_16_256_10_2_30.txt
#
#echo "Evaluating model Lstm_16_256_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_10_2 --model_name Lstm_16_256_10_2_30 > ../Analysis/logs/eval/step3/Lstm_16_256_10_2_30.txt
#
#echo "Training model Lstm_16_256_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_16_256_20_2 --model_name Lstm_16_256_20_2_30 > ../Analysis/logs/train/step3/Lstm_16_256_20_2_30.txt
#
#echo "Evaluating model Lstm_16_256_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_16_256_20_2 --model_name Lstm_16_256_20_2_30 > ../Analysis/logs/eval/step3/Lstm_16_256_20_2_30.txt
#
## 2 capas, 32-16
#
#echo "Training model Lstm_32_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_1_2 --model_name Lstm_32_16_1_2_30 > ../Analysis/logs/train/step3/Lstm_32_16_1_2_30.txt
#
#echo "Evaluating model Lstm_32_16_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_1_2 --model_name Lstm_32_16_1_2_30 > ../Analysis/logs/eval/step3/Lstm_32_16_1_2_30.txt
#
#echo "Training model Lstm_32_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_2_2 --model_name Lstm_32_16_2_2_30 > ../Analysis/logs/train/step3/Lstm_32_16_2_2_30.txt
#
#echo "Evaluating model Lstm_32_16_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_2_2 --model_name Lstm_32_16_2_2_30 > ../Analysis/logs/eval/step3/Lstm_32_16_2_2_30.txt
#
#echo "Training model Lstm_32_16_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_5_2 --model_name Lstm_32_16_5_2_30 > ../Analysis/logs/train/step3/Lstm_32_16_5_2_30.txt
#
#echo "Evaluating model Lstm_32_16_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_5_2 --model_name Lstm_32_16_5_2_30 > ../Analysis/logs/eval/step3/Lstm_32_16_5_2_30.txt
#
#echo "Training model Lstm_32_16_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_10_2 --model_name Lstm_32_16_10_2_30 > ../Analysis/logs/train/step3/Lstm_32_16_10_2_30.txt
#
#echo "Evaluating model Lstm_32_16_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_10_2 --model_name Lstm_32_16_10_2_30 > ../Analysis/logs/eval/step3/Lstm_32_16_10_2_30.txt
#
#echo "Training model Lstm_32_16_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_16_20_2 --model_name Lstm_32_16_20_2_30 > ../Analysis/logs/train/step3/Lstm_32_16_20_2_30.txt
#
#echo "Evaluating model Lstm_32_16_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_16_20_2 --model_name Lstm_32_16_20_2_30 > ../Analysis/logs/eval/step3/Lstm_32_16_20_2_30.txt
#
## 2 capas, 32-32
#
#echo "Training model Lstm_32_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_1_2 --model_name Lstm_32_32_1_2_30 > ../Analysis/logs/train/step3/Lstm_32_32_1_2_30.txt
#
#echo "Evaluating model Lstm_32_32_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_1_2 --model_name Lstm_32_32_1_2_30 > ../Analysis/logs/eval/step3/Lstm_32_32_1_2_30.txt
#
#echo "Training model Lstm_32_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_2 --model_name Lstm_32_32_2_2_30 > ../Analysis/logs/train/step3/Lstm_32_32_2_2_30.txt
#
#echo "Evaluating model Lstm_32_32_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_2_2 --model_name Lstm_32_32_2_2_30 > ../Analysis/logs/eval/step3/Lstm_32_32_2_2_30.txt
#
#echo "Training model Lstm_32_32_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_5_2 --model_name Lstm_32_32_5_2_30 > ../Analysis/logs/train/step3/Lstm_32_32_5_2_30.txt
#
#echo "Evaluating model Lstm_32_32_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_5_2 --model_name Lstm_32_32_5_2_30 > ../Analysis/logs/eval/step3/Lstm_32_32_5_2_30.txt
#
#echo "Training model Lstm_32_32_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_10_2 --model_name Lstm_32_32_10_2_30 > ../Analysis/logs/train/step3/Lstm_32_32_10_2_30.txt
#
#echo "Evaluating model Lstm_32_32_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_10_2 --model_name Lstm_32_32_10_2_30 > ../Analysis/logs/eval/step3/Lstm_32_32_10_2_30.txt
#
#echo "Training model Lstm_32_32_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_32_20_2 --model_name Lstm_32_32_20_2_30 > ../Analysis/logs/train/step3/Lstm_32_32_20_2_30.txt
#
#echo "Evaluating model Lstm_32_32_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_32_20_2 --model_name Lstm_32_32_20_2_30 > ../Analysis/logs/eval/step3/Lstm_32_32_20_2_30.txt
#
## 2 capas, 32-64
#
#echo "Training model Lstm_32_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_2 --model_name Lstm_32_64_1_2_30 > ../Analysis/logs/train/step3/Lstm_32_64_1_2_30.txt
#
#echo "Evaluating model Lstm_32_64_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_1_2 --model_name Lstm_32_64_1_2_30 > ../Analysis/logs/eval/step3/Lstm_32_64_1_2_30.txt
#
#echo "Training model Lstm_32_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_2_2 --model_name Lstm_32_64_2_2_30 > ../Analysis/logs/train/step3/Lstm_32_64_2_2_30.txt
#
#echo "Evaluating model Lstm_32_64_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_2_2 --model_name Lstm_32_64_2_2_30 > ../Analysis/logs/eval/step3/Lstm_32_64_2_2_30.txt
#
#echo "Training model Lstm_32_64_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_5_2 --model_name Lstm_32_64_5_2_30 > ../Analysis/logs/train/step3/Lstm_32_64_5_2_30.txt
#
#echo "Evaluating model Lstm_32_64_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_5_2 --model_name Lstm_32_64_5_2_30 > ../Analysis/logs/eval/step3/Lstm_32_64_5_2_30.txt
#
#echo "Training model Lstm_32_64_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_10_2 --model_name Lstm_32_64_10_2_30 > ../Analysis/logs/train/step3/Lstm_32_64_10_2_30.txt
#
#echo "Evaluating model Lstm_32_64_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_10_2 --model_name Lstm_32_64_10_2_30 > ../Analysis/logs/eval/step3/Lstm_32_64_10_2_30.txt
#
#echo "Training model Lstm_32_64_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_64_20_2 --model_name Lstm_32_64_20_2_30 > ../Analysis/logs/train/step3/Lstm_32_64_20_2_30.txt
#
#echo "Evaluating model Lstm_32_64_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_64_20_2 --model_name Lstm_32_64_20_2_30 > ../Analysis/logs/eval/step3/Lstm_32_64_20_2_30.txt
#
## 2 capas, 32-128
#
#echo "Training model Lstm_32_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_1_2 --model_name Lstm_32_128_1_2_30 > ../Analysis/logs/train/step3/Lstm_32_128_1_2_30.txt
#
#echo "Evaluating model Lstm_32_128_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_1_2 --model_name Lstm_32_128_1_2_30 > ../Analysis/logs/eval/step3/Lstm_32_128_1_2_30.txt
#
#echo "Training model Lstm_32_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_2_2 --model_name Lstm_32_128_2_2_30 > ../Analysis/logs/train/step3/Lstm_32_128_2_2_30.txt
#
#echo "Evaluating model Lstm_32_128_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_2_2 --model_name Lstm_32_128_2_2_30 > ../Analysis/logs/eval/step3/Lstm_32_128_2_2_30.txt
#
#echo "Training model Lstm_32_128_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_5_2 --model_name Lstm_32_128_5_2_30 > ../Analysis/logs/train/step3/Lstm_32_128_5_2_30.txt
#
#echo "Evaluating model Lstm_32_128_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_5_2 --model_name Lstm_32_128_5_2_30 > ../Analysis/logs/eval/step3/Lstm_32_128_5_2_30.txt
#
#echo "Training model Lstm_32_128_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_10_2 --model_name Lstm_32_128_10_2_30 > ../Analysis/logs/train/step3/Lstm_32_128_10_2_30.txt
#
#echo "Evaluating model Lstm_32_128_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_10_2 --model_name Lstm_32_128_10_2_30 > ../Analysis/logs/eval/step3/Lstm_32_128_10_2_30.txt
#
#echo "Training model Lstm_32_128_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_128_20_2 --model_name Lstm_32_128_20_2_30 > ../Analysis/logs/train/step3/Lstm_32_128_20_2_30.txt
#
#echo "Evaluating model Lstm_32_128_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_128_20_2 --model_name Lstm_32_128_20_2_30 > ../Analysis/logs/eval/step3/Lstm_32_128_20_2_30.txt
#
## 2 capas, 32-256
#
#echo "Training model Lstm_32_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_1_2 --model_name Lstm_32_256_1_2_30 > ../Analysis/logs/train/step3/Lstm_32_256_1_2_30.txt
#
#echo "Evaluating model Lstm_32_256_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_1_2 --model_name Lstm_32_256_1_2_30 > ../Analysis/logs/eval/step3/Lstm_32_256_1_2_30.txt
#
#echo "Training model Lstm_32_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_2_2 --model_name Lstm_32_256_2_2_30 > ../Analysis/logs/train/step3/Lstm_32_256_2_2_30.txt
#
#echo "Evaluating model Lstm_32_256_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_2_2 --model_name Lstm_32_256_2_2_30 > ../Analysis/logs/eval/step3/Lstm_32_256_2_2_30.txt
#
#echo "Training model Lstm_32_256_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_5_2 --model_name Lstm_32_256_5_2_30 > ../Analysis/logs/train/step3/Lstm_32_256_5_2_30.txt
#
#echo "Evaluating model Lstm_32_256_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_5_2 --model_name Lstm_32_256_5_2_30 > ../Analysis/logs/eval/step3/Lstm_32_256_5_2_30.txt
#
#echo "Training model Lstm_32_256_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_10_2 --model_name Lstm_32_256_10_2_30 > ../Analysis/logs/train/step3/Lstm_32_256_10_2_30.txt
#
#echo "Evaluating model Lstm_32_256_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_10_2 --model_name Lstm_32_256_10_2_30 > ../Analysis/logs/eval/step3/Lstm_32_256_10_2_30.txt
#
#echo "Training model Lstm_32_256_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_32_256_20_2 --model_name Lstm_32_256_20_2_30 > ../Analysis/logs/train/step3/Lstm_32_256_20_2_30.txt
#
#echo "Evaluating model Lstm_32_256_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_32_256_20_2 --model_name Lstm_32_256_20_2_30 > ../Analysis/logs/eval/step3/Lstm_32_256_20_2_30.txt
#
## 2 capas, 64-16
#
#echo "Training model Lstm_64_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_1_2 --model_name Lstm_64_16_1_2_30 > ../Analysis/logs/train/step3/Lstm_64_16_1_2_30.txt
#
#echo "Evaluating model Lstm_64_16_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_1_2 --model_name Lstm_64_16_1_2_30 > ../Analysis/logs/eval/step3/Lstm_64_16_1_2_30.txt
#
#echo "Training model Lstm_64_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_2_2 --model_name Lstm_64_16_2_2_30 > ../Analysis/logs/train/step3/Lstm_64_16_2_2_30.txt
#
#echo "Evaluating model Lstm_64_16_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_2_2 --model_name Lstm_64_16_2_2_30 > ../Analysis/logs/eval/step3/Lstm_64_16_2_2_30.txt
#
#echo "Training model Lstm_64_16_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_5_2 --model_name Lstm_64_16_5_2_30 > ../Analysis/logs/train/step3/Lstm_64_16_5_2_30.txt
#
#echo "Evaluating model Lstm_64_16_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_5_2 --model_name Lstm_64_16_5_2_30 > ../Analysis/logs/eval/step3/Lstm_64_16_5_2_30.txt
#
#echo "Training model Lstm_64_16_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_10_2 --model_name Lstm_64_16_10_2_30 > ../Analysis/logs/train/step3/Lstm_64_16_10_2_30.txt
#
#echo "Evaluating model Lstm_64_16_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_10_2 --model_name Lstm_64_16_10_2_30 > ../Analysis/logs/eval/step3/Lstm_64_16_10_2_30.txt
#
#echo "Training model Lstm_64_16_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_16_20_2 --model_name Lstm_64_16_20_2_30 > ../Analysis/logs/train/step3/Lstm_64_16_20_2_30.txt
#
#echo "Evaluating model Lstm_64_16_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_16_20_2 --model_name Lstm_64_16_20_2_30 > ../Analysis/logs/eval/step3/Lstm_64_16_20_2_30.txt
#
## 2 capas, 64-32
#
#echo "Training model Lstm_64_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_1_2 --model_name Lstm_64_32_1_2_30 > ../Analysis/logs/train/step3/Lstm_64_32_1_2_30.txt
#
#echo "Evaluating model Lstm_64_32_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_1_2 --model_name Lstm_64_32_1_2_30 > ../Analysis/logs/eval/step3/Lstm_64_32_1_2_30.txt
#
#echo "Training model Lstm_64_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_30 > ../Analysis/logs/train/step3/Lstm_64_32_2_2_30.txt
#
#echo "Evaluating model Lstm_64_32_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_30 > ../Analysis/logs/eval/step3/Lstm_64_32_2_2_30.txt
#
#echo "Training model Lstm_64_32_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_5_2 --model_name Lstm_64_32_5_2_30 > ../Analysis/logs/train/step3/Lstm_64_32_5_2_30.txt
#
#echo "Evaluating model Lstm_64_32_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_5_2 --model_name Lstm_64_32_5_2_30 > ../Analysis/logs/eval/step3/Lstm_64_32_5_2_30.txt
#
#echo "Training model Lstm_64_32_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_10_2 --model_name Lstm_64_32_10_2_30 > ../Analysis/logs/train/step3/Lstm_64_32_10_2_30.txt
#
#echo "Evaluating model Lstm_64_32_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_10_2 --model_name Lstm_64_32_10_2_30 > ../Analysis/logs/eval/step3/Lstm_64_32_10_2_30.txt
#
#echo "Training model Lstm_64_32_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_32_20_2 --model_name Lstm_64_32_20_2_30 > ../Analysis/logs/train/step3/Lstm_64_32_20_2_30.txt
#
#echo "Evaluating model Lstm_64_32_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_32_20_2 --model_name Lstm_64_32_20_2_30 > ../Analysis/logs/eval/step3/Lstm_64_32_20_2_30.txt
#
## 2 capas, 64-64
#
#echo "Training model Lstm_64_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_1_2 --model_name Lstm_64_64_1_2_30 > ../Analysis/logs/train/step3/Lstm_64_64_1_2_30.txt
#
#echo "Evaluating model Lstm_64_64_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_1_2 --model_name Lstm_64_64_1_2_30 > ../Analysis/logs/eval/step3/Lstm_64_64_1_2_30.txt
#
#echo "Training model Lstm_64_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_2_2 --model_name Lstm_64_64_2_2_30 > ../Analysis/logs/train/step3/Lstm_64_64_2_2_30.txt
#
#echo "Evaluating model Lstm_64_64_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_2_2 --model_name Lstm_64_64_2_2_30 > ../Analysis/logs/eval/step3/Lstm_64_64_2_2_30.txt
#
#echo "Training model Lstm_64_64_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_30 > ../Analysis/logs/train/step3/Lstm_64_64_5_2_30.txt
#
#echo "Evaluating model Lstm_64_64_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_30 > ../Analysis/logs/eval/step3/Lstm_64_64_5_2_30.txt
#
#echo "Training model Lstm_64_64_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_10_2 --model_name Lstm_64_64_10_2_30 > ../Analysis/logs/train/step3/Lstm_64_64_10_2_30.txt
#
#echo "Evaluating model Lstm_64_64_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_10_2 --model_name Lstm_64_64_10_2_30 > ../Analysis/logs/eval/step3/Lstm_64_64_10_2_30.txt
#
#echo "Training model Lstm_64_64_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_64_20_2 --model_name Lstm_64_64_20_2_30 > ../Analysis/logs/train/step3/Lstm_64_64_20_2_30.txt
#
#echo "Evaluating model Lstm_64_64_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_64_20_2 --model_name Lstm_64_64_20_2_30 > ../Analysis/logs/eval/step3/Lstm_64_64_20_2_30.txt
#
## 2 capas, 64-128
#
#echo "Training model Lstm_64_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_1_2 --model_name Lstm_64_128_1_2_30 > ../Analysis/logs/train/step3/Lstm_64_128_1_2_30.txt
#
#echo "Evaluating model Lstm_64_128_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_1_2 --model_name Lstm_64_128_1_2_30 > ../Analysis/logs/eval/step3/Lstm_64_128_1_2_30.txt
#
#echo "Training model Lstm_64_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_2_2 --model_name Lstm_64_128_2_2_30 > ../Analysis/logs/train/step3/Lstm_64_128_2_2_30.txt
#
#echo "Evaluating model Lstm_64_128_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_2_2 --model_name Lstm_64_128_2_2_30 > ../Analysis/logs/eval/step3/Lstm_64_128_2_2_30.txt
#
#echo "Training model Lstm_64_128_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_5_2 --model_name Lstm_64_128_5_2_30 > ../Analysis/logs/train/step3/Lstm_64_128_5_2_30.txt
#
#echo "Evaluating model Lstm_64_128_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_5_2 --model_name Lstm_64_128_5_2_30 > ../Analysis/logs/eval/step3/Lstm_64_128_5_2_30.txt
#
#echo "Training model Lstm_64_128_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_10_2 --model_name Lstm_64_128_10_2_30 > ../Analysis/logs/train/step3/Lstm_64_128_10_2_30.txt
#
#echo "Evaluating model Lstm_64_128_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_10_2 --model_name Lstm_64_128_10_2_30 > ../Analysis/logs/eval/step3/Lstm_64_128_10_2_30.txt
#
#echo "Training model Lstm_64_128_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_128_20_2 --model_name Lstm_64_128_20_2_30 > ../Analysis/logs/train/step3/Lstm_64_128_20_2_30.txt
#
#echo "Evaluating model Lstm_64_128_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_128_20_2 --model_name Lstm_64_128_20_2_30 > ../Analysis/logs/eval/step3/Lstm_64_128_20_2_30.txt
#
## 2 capas, 64-256
#
#echo "Training model Lstm_64_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_1_2 --model_name Lstm_64_256_1_2_30 > ../Analysis/logs/train/step3/Lstm_64_256_1_2_30.txt
#
#echo "Evaluating model Lstm_64_256_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_1_2 --model_name Lstm_64_256_1_2_30 > ../Analysis/logs/eval/step3/Lstm_64_256_1_2_30.txt
#
#echo "Training model Lstm_64_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_2_2 --model_name Lstm_64_256_2_2_30 > ../Analysis/logs/train/step3/Lstm_64_256_2_2_30.txt
#
#echo "Evaluating model Lstm_64_256_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_2_2 --model_name Lstm_64_256_2_2_30 > ../Analysis/logs/eval/step3/Lstm_64_256_2_2_30.txt
#
#echo "Training model Lstm_64_256_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_5_2 --model_name Lstm_64_256_5_2_30 > ../Analysis/logs/train/step3/Lstm_64_256_5_2_30.txt
#
#echo "Evaluating model Lstm_64_256_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_5_2 --model_name Lstm_64_256_5_2_30 > ../Analysis/logs/eval/step3/Lstm_64_256_5_2_30.txt
#
#echo "Training model Lstm_64_256_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_10_2 --model_name Lstm_64_256_10_2_30 > ../Analysis/logs/train/step3/Lstm_64_256_10_2_30.txt
#
#echo "Evaluating model Lstm_64_256_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_10_2 --model_name Lstm_64_256_10_2_30 > ../Analysis/logs/eval/step3/Lstm_64_256_10_2_30.txt
#
#echo "Training model Lstm_64_256_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_64_256_20_2 --model_name Lstm_64_256_20_2_30 > ../Analysis/logs/train/step3/Lstm_64_256_20_2_30.txt
#
#echo "Evaluating model Lstm_64_256_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_64_256_20_2 --model_name Lstm_64_256_20_2_30 > ../Analysis/logs/eval/step3/Lstm_64_256_20_2_30.txt
#
## 2 capas, 128-16
#
#echo "Training model Lstm_128_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_1_2 --model_name Lstm_128_16_1_2_30 > ../Analysis/logs/train/step3/Lstm_128_16_1_2_30.txt
#
#echo "Evaluating model Lstm_128_16_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_1_2 --model_name Lstm_128_16_1_2_30 > ../Analysis/logs/eval/step3/Lstm_128_16_1_2_30.txt
#
#echo "Training model Lstm_128_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_2_2 --model_name Lstm_128_16_2_2_30 > ../Analysis/logs/train/step3/Lstm_128_16_2_2_30.txt
#
#echo "Evaluating model Lstm_128_16_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_2_2 --model_name Lstm_128_16_2_2_30 > ../Analysis/logs/eval/step3/Lstm_128_16_2_2_30.txt
#
#echo "Training model Lstm_128_16_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_5_2 --model_name Lstm_128_16_5_2_30 > ../Analysis/logs/train/step3/Lstm_128_16_5_2_30.txt
#
#echo "Evaluating model Lstm_128_16_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_5_2 --model_name Lstm_128_16_5_2_30 > ../Analysis/logs/eval/step3/Lstm_128_16_5_2_30.txt
#
#echo "Training model Lstm_128_16_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_10_2 --model_name Lstm_128_16_10_2_30 > ../Analysis/logs/train/step3/Lstm_128_16_10_2_30.txt
#
#echo "Evaluating model Lstm_128_16_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_10_2 --model_name Lstm_128_16_10_2_30 > ../Analysis/logs/eval/step3/Lstm_128_16_10_2_30.txt
#
#echo "Training model Lstm_128_16_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_16_20_2 --model_name Lstm_128_16_20_2_30 > ../Analysis/logs/train/step3/Lstm_128_16_20_2_30.txt
#
#echo "Evaluating model Lstm_128_16_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_16_20_2 --model_name Lstm_128_16_20_2_30 > ../Analysis/logs/eval/step3/Lstm_128_16_20_2_30.txt
#
## 2 capas, 128-32
#
#echo "Training model Lstm_128_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_1_2 --model_name Lstm_128_32_1_2_30 > ../Analysis/logs/train/step3/Lstm_128_32_1_2_30.txt
#
#echo "Evaluating model Lstm_128_32_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_1_2 --model_name Lstm_128_32_1_2_30 > ../Analysis/logs/eval/step3/Lstm_128_32_1_2_30.txt
#
#echo "Training model Lstm_128_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_2_2 --model_name Lstm_128_32_2_2_30 > ../Analysis/logs/train/step3/Lstm_128_32_2_2_30.txt
#
#echo "Evaluating model Lstm_128_32_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_2_2 --model_name Lstm_128_32_2_2_30 > ../Analysis/logs/eval/step3/Lstm_128_32_2_2_30.txt
#
#echo "Training model Lstm_128_32_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_5_2 --model_name Lstm_128_32_5_2_30 > ../Analysis/logs/train/step3/Lstm_128_32_5_2_30.txt
#
#echo "Evaluating model Lstm_128_32_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_5_2 --model_name Lstm_128_32_5_2_30 > ../Analysis/logs/eval/step3/Lstm_128_32_5_2_30.txt
#
#echo "Training model Lstm_128_32_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_10_2 --model_name Lstm_128_32_10_2_30 > ../Analysis/logs/train/step3/Lstm_128_32_10_2_30.txt
#
#echo "Evaluating model Lstm_128_32_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_10_2 --model_name Lstm_128_32_10_2_30 > ../Analysis/logs/eval/step3/Lstm_128_32_10_2_30.txt
#
#echo "Training model Lstm_128_32_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_32_20_2 --model_name Lstm_128_32_20_2_30 > ../Analysis/logs/train/step3/Lstm_128_32_20_2_30.txt
#
#echo "Evaluating model Lstm_128_32_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_32_20_2 --model_name Lstm_128_32_20_2_30 > ../Analysis/logs/eval/step3/Lstm_128_32_20_2_30.txt
#
## 2 capas, 128-64
#
#echo "Training model Lstm_128_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_1_2 --model_name Lstm_128_64_1_2_30 > ../Analysis/logs/train/step3/Lstm_128_64_1_2_30.txt
#
#echo "Evaluating model Lstm_128_64_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_1_2 --model_name Lstm_128_64_1_2_30 > ../Analysis/logs/eval/step3/Lstm_128_64_1_2_30.txt
#
#echo "Training model Lstm_128_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_2_2 --model_name Lstm_128_64_2_2_30 > ../Analysis/logs/train/step3/Lstm_128_64_2_2_30.txt
#
#echo "Evaluating model Lstm_128_64_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_2_2 --model_name Lstm_128_64_2_2_30 > ../Analysis/logs/eval/step3/Lstm_128_64_2_2_30.txt
#
#echo "Training model Lstm_128_64_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_5_2 --model_name Lstm_128_64_5_2_30 > ../Analysis/logs/train/step3/Lstm_128_64_5_2_30.txt
#
#echo "Evaluating model Lstm_128_64_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_5_2 --model_name Lstm_128_64_5_2_30 > ../Analysis/logs/eval/step3/Lstm_128_64_5_2_30.txt
#
#echo "Training model Lstm_128_64_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_10_2 --model_name Lstm_128_64_10_2_30 > ../Analysis/logs/train/step3/Lstm_128_64_10_2_30.txt
#
#echo "Evaluating model Lstm_128_64_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_10_2 --model_name Lstm_128_64_10_2_30 > ../Analysis/logs/eval/step3/Lstm_128_64_10_2_30.txt
#
#echo "Training model Lstm_128_64_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_64_20_2 --model_name Lstm_128_64_20_2_30 > ../Analysis/logs/train/step3/Lstm_128_64_20_2_30.txt
#
#echo "Evaluating model Lstm_128_64_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_64_20_2 --model_name Lstm_128_64_20_2_30 > ../Analysis/logs/eval/step3/Lstm_128_64_20_2_30.txt
#
## 2 capas, 128-128
#
#echo "Training model Lstm_128_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_1_2 --model_name Lstm_128_128_1_2_30 > ../Analysis/logs/train/step3/Lstm_128_128_1_2_30.txt
#
#echo "Evaluating model Lstm_128_128_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_1_2 --model_name Lstm_128_128_1_2_30 > ../Analysis/logs/eval/step3/Lstm_128_128_1_2_30.txt
#
#echo "Training model Lstm_128_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_2_2 --model_name Lstm_128_128_2_2_30 > ../Analysis/logs/train/step3/Lstm_128_128_2_2_30.txt
#
#echo "Evaluating model Lstm_128_128_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_2_2 --model_name Lstm_128_128_2_2_30 > ../Analysis/logs/eval/step3/Lstm_128_128_2_2_30.txt
#
#echo "Training model Lstm_128_128_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_5_2 --model_name Lstm_128_128_5_2_30 > ../Analysis/logs/train/step3/Lstm_128_128_5_2_30.txt
#
#echo "Evaluating model Lstm_128_128_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_5_2 --model_name Lstm_128_128_5_2_30 > ../Analysis/logs/eval/step3/Lstm_128_128_5_2_30.txt
#
#echo "Training model Lstm_128_128_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_10_2 --model_name Lstm_128_128_10_2_30 > ../Analysis/logs/train/step3/Lstm_128_128_10_2_30.txt
#
#echo "Evaluating model Lstm_128_128_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_10_2 --model_name Lstm_128_128_10_2_30 > ../Analysis/logs/eval/step3/Lstm_128_128_10_2_30.txt
#
#echo "Training model Lstm_128_128_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_128_20_2 --model_name Lstm_128_128_20_2_30 > ../Analysis/logs/train/step3/Lstm_128_128_20_2_30.txt
#
#echo "Evaluating model Lstm_128_128_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_128_20_2 --model_name Lstm_128_128_20_2_30 > ../Analysis/logs/eval/step3/Lstm_128_128_20_2_30.txt
#
## 2 capas, 128-256
#
#echo "Training model Lstm_128_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_1_2 --model_name Lstm_128_256_1_2_30 > ../Analysis/logs/train/step3/Lstm_128_256_1_2_30.txt
#
#echo "Evaluating model Lstm_128_256_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_1_2 --model_name Lstm_128_256_1_2_30 > ../Analysis/logs/eval/step3/Lstm_128_256_1_2_30.txt
#
#echo "Training model Lstm_128_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_2_2 --model_name Lstm_128_256_2_2_30 > ../Analysis/logs/train/step3/Lstm_128_256_2_2_30.txt
#
#echo "Evaluating model Lstm_128_256_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_2_2 --model_name Lstm_128_256_2_2_30 > ../Analysis/logs/eval/step3/Lstm_128_256_2_2_30.txt
#
#echo "Training model Lstm_128_256_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_5_2 --model_name Lstm_128_256_5_2_30 > ../Analysis/logs/train/step3/Lstm_128_256_5_2_30.txt
#
#echo "Evaluating model Lstm_128_256_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_5_2 --model_name Lstm_128_256_5_2_30 > ../Analysis/logs/eval/step3/Lstm_128_256_5_2_30.txt
#
#echo "Training model Lstm_128_256_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_10_2 --model_name Lstm_128_256_10_2_30 > ../Analysis/logs/train/step3/Lstm_128_256_10_2_30.txt
#
#echo "Evaluating model Lstm_128_256_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_10_2 --model_name Lstm_128_256_10_2_30 > ../Analysis/logs/eval/step3/Lstm_128_256_10_2_30.txt
#
#echo "Training model Lstm_128_256_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_128_256_20_2 --model_name Lstm_128_256_20_2_30 > ../Analysis/logs/train/step3/Lstm_128_256_20_2_30.txt
#
#echo "Evaluating model Lstm_128_256_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_128_256_20_2 --model_name Lstm_128_256_20_2_30 > ../Analysis/logs/eval/step3/Lstm_128_256_20_2_30.txt
#
## 2 capas, 256-16
#
#echo "Training model Lstm_256_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_1_2 --model_name Lstm_256_16_1_2_30 > ../Analysis/logs/train/step3/Lstm_256_16_1_2_30.txt
#
#echo "Evaluating model Lstm_256_16_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_1_2 --model_name Lstm_256_16_1_2_30 > ../Analysis/logs/eval/step3/Lstm_256_16_1_2_30.txt
#
#echo "Training model Lstm_256_16_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_2_2 --model_name Lstm_256_16_2_2_30 > ../Analysis/logs/train/step3/Lstm_256_16_2_2_30.txt
#
#echo "Evaluating model Lstm_256_16_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_2_2 --model_name Lstm_256_16_2_2_30 > ../Analysis/logs/eval/step3/Lstm_256_16_2_2_30.txt
#
#echo "Training model Lstm_256_16_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_5_2 --model_name Lstm_256_16_5_2_30 > ../Analysis/logs/train/step3/Lstm_256_16_5_2_30.txt
#
#echo "Evaluating model Lstm_256_16_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_5_2 --model_name Lstm_256_16_5_2_30 > ../Analysis/logs/eval/step3/Lstm_256_16_5_2_30.txt
#
#echo "Training model Lstm_256_16_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_10_2 --model_name Lstm_256_16_10_2_30 > ../Analysis/logs/train/step3/Lstm_256_16_10_2_30.txt
#
#echo "Evaluating model Lstm_256_16_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_10_2 --model_name Lstm_256_16_10_2_30 > ../Analysis/logs/eval/step3/Lstm_256_16_10_2_30.txt
#
#echo "Training model Lstm_256_16_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_16_20_2 --model_name Lstm_256_16_20_2_30 > ../Analysis/logs/train/step3/Lstm_256_16_20_2_30.txt
#
#echo "Evaluating model Lstm_256_16_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_16_20_2 --model_name Lstm_256_16_20_2_30 > ../Analysis/logs/eval/step3/Lstm_256_16_20_2_30.txt
#
## 2 capas, 256-32
#
#echo "Training model Lstm_256_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_1_2 --model_name Lstm_256_32_1_2_30 > ../Analysis/logs/train/step3/Lstm_256_32_1_2_30.txt
#
#echo "Evaluating model Lstm_256_32_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_1_2 --model_name Lstm_256_32_1_2_30 > ../Analysis/logs/eval/step3/Lstm_256_32_1_2_30.txt
#
#echo "Training model Lstm_256_32_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_2_2 --model_name Lstm_256_32_2_2_30 > ../Analysis/logs/train/step3/Lstm_256_32_2_2_30.txt
#
#echo "Evaluating model Lstm_256_32_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_2_2 --model_name Lstm_256_32_2_2_30 > ../Analysis/logs/eval/step3/Lstm_256_32_2_2_30.txt
#
#echo "Training model Lstm_256_32_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_5_2 --model_name Lstm_256_32_5_2_30 > ../Analysis/logs/train/step3/Lstm_256_32_5_2_30.txt
#
#echo "Evaluating model Lstm_256_32_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_5_2 --model_name Lstm_256_32_5_2_30 > ../Analysis/logs/eval/step3/Lstm_256_32_5_2_30.txt
#
#echo "Training model Lstm_256_32_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_10_2 --model_name Lstm_256_32_10_2_30 > ../Analysis/logs/train/step3/Lstm_256_32_10_2_30.txt
#
#echo "Evaluating model Lstm_256_32_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_10_2 --model_name Lstm_256_32_10_2_30 > ../Analysis/logs/eval/step3/Lstm_256_32_10_2_30.txt
#
#echo "Training model Lstm_256_32_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_32_20_2 --model_name Lstm_256_32_20_2_30 > ../Analysis/logs/train/step3/Lstm_256_32_20_2_30.txt
#
#echo "Evaluating model Lstm_256_32_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_32_20_2 --model_name Lstm_256_32_20_2_30 > ../Analysis/logs/eval/step3/Lstm_256_32_20_2_30.txt
#
## 2 capas, 256-64
#
#echo "Training model Lstm_256_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_1_2 --model_name Lstm_256_64_1_2_30 > ../Analysis/logs/train/step3/Lstm_256_64_1_2_30.txt
#
#echo "Evaluating model Lstm_256_64_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_1_2 --model_name Lstm_256_64_1_2_30 > ../Analysis/logs/eval/step3/Lstm_256_64_1_2_30.txt
#
#echo "Training model Lstm_256_64_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_2_2 --model_name Lstm_256_64_2_2_30 > ../Analysis/logs/train/step3/Lstm_256_64_2_2_30.txt
#
#echo "Evaluating model Lstm_256_64_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_2_2 --model_name Lstm_256_64_2_2_30 > ../Analysis/logs/eval/step3/Lstm_256_64_2_2_30.txt
#
#echo "Training model Lstm_256_64_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_5_2 --model_name Lstm_256_64_5_2_30 > ../Analysis/logs/train/step3/Lstm_256_64_5_2_30.txt
#
#echo "Evaluating model Lstm_256_64_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_5_2 --model_name Lstm_256_64_5_2_30 > ../Analysis/logs/eval/step3/Lstm_256_64_5_2_30.txt
#
#echo "Training model Lstm_256_64_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_10_2 --model_name Lstm_256_64_10_2_30 > ../Analysis/logs/train/step3/Lstm_256_64_10_2_30.txt
#
#echo "Evaluating model Lstm_256_64_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_10_2 --model_name Lstm_256_64_10_2_30 > ../Analysis/logs/eval/step3/Lstm_256_64_10_2_30.txt
#
#echo "Training model Lstm_256_64_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_64_20_2 --model_name Lstm_256_64_20_2_30 > ../Analysis/logs/train/step3/Lstm_256_64_20_2_30.txt
#
#echo "Evaluating model Lstm_256_64_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_64_20_2 --model_name Lstm_256_64_20_2_30 > ../Analysis/logs/eval/step3/Lstm_256_64_20_2_30.txt
#
## 2 capas, 256-128
#
#echo "Training model Lstm_256_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_1_2 --model_name Lstm_256_128_1_2_30 > ../Analysis/logs/train/step3/Lstm_256_128_1_2_30.txt
#
#echo "Evaluating model Lstm_256_128_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_1_2 --model_name Lstm_256_128_1_2_30 > ../Analysis/logs/eval/step3/Lstm_256_128_1_2_30.txt
#
#echo "Training model Lstm_256_128_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_2_2 --model_name Lstm_256_128_2_2_30 > ../Analysis/logs/train/step3/Lstm_256_128_2_2_30.txt
#
#echo "Evaluating model Lstm_256_128_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_2_2 --model_name Lstm_256_128_2_2_30 > ../Analysis/logs/eval/step3/Lstm_256_128_2_2_30.txt
#
#echo "Training model Lstm_256_128_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_5_2 --model_name Lstm_256_128_5_2_30 > ../Analysis/logs/train/step3/Lstm_256_128_5_2_30.txt
#
#echo "Evaluating model Lstm_256_128_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_5_2 --model_name Lstm_256_128_5_2_30 > ../Analysis/logs/eval/step3/Lstm_256_128_5_2_30.txt
#
#echo "Training model Lstm_256_128_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_10_2 --model_name Lstm_256_128_10_2_30 > ../Analysis/logs/train/step3/Lstm_256_128_10_2_30.txt
#
#echo "Evaluating model Lstm_256_128_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_10_2 --model_name Lstm_256_128_10_2_30 > ../Analysis/logs/eval/step3/Lstm_256_128_10_2_30.txt
#
#echo "Training model Lstm_256_128_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_128_20_2 --model_name Lstm_256_128_20_2_30 > ../Analysis/logs/train/step3/Lstm_256_128_20_2_30.txt
#
#echo "Evaluating model Lstm_256_128_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_128_20_2 --model_name Lstm_256_128_20_2_30 > ../Analysis/logs/eval/step3/Lstm_256_128_20_2_30.txt
#
## 2 capas, 256-256
#
#echo "Training model Lstm_256_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_1_2 --model_name Lstm_256_256_1_2_30 > ../Analysis/logs/train/step3/Lstm_256_256_1_2_30.txt
#
#echo "Evaluating model Lstm_256_256_1_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_1_2 --model_name Lstm_256_256_1_2_30 > ../Analysis/logs/eval/step3/Lstm_256_256_1_2_30.txt
#
#echo "Training model Lstm_256_256_1_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_2_2 --model_name Lstm_256_256_2_2_30 > ../Analysis/logs/train/step3/Lstm_256_256_2_2_30.txt
#
#echo "Evaluating model Lstm_256_256_2_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_2_2 --model_name Lstm_256_256_2_2_30 > ../Analysis/logs/eval/step3/Lstm_256_256_2_2_30.txt
#
#echo "Training model Lstm_256_256_5_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_5_2 --model_name Lstm_256_256_5_2_30 > ../Analysis/logs/train/step3/Lstm_256_256_5_2_30.txt
#
#echo "Evaluating model Lstm_256_256_5_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_5_2 --model_name Lstm_256_256_5_2_30 > ../Analysis/logs/eval/step3/Lstm_256_256_5_2_30.txt
#
#echo "Training model Lstm_256_256_10_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_10_2 --model_name Lstm_256_256_10_2_30 > ../Analysis/logs/train/step3/Lstm_256_256_10_2_30.txt
#
#echo "Evaluating model Lstm_256_256_10_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_10_2 --model_name Lstm_256_256_10_2_30 > ../Analysis/logs/eval/step3/Lstm_256_256_10_2_30.txt
#
#echo "Training model Lstm_256_256_20_2_30, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Lstm_256_256_20_2 --model_name Lstm_256_256_20_2_30 > ../Analysis/logs/train/step3/Lstm_256_256_20_2_30.txt
#
#echo "Evaluating model Lstm_256_256_20_2_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Lstm_256_256_20_2 --model_name Lstm_256_256_20_2_30 > ../Analysis/logs/eval/step3/Lstm_256_256_20_2_30.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../traineval2excel.py --xls_name 'LSTM_step3' --archives_folder 'step3'