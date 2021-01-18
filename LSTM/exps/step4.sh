#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

### Lstm_64_64_1_1
##
### Learning rate 1e-2
##
###echo "Training model Lstm_16_16_1_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_16_16_1_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e2_256_20.txt
###
###echo "Training model Lstm_16_16_1_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_16_16_1_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e2_256_25.txt
###
###echo "Training model Lstm_16_16_1_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_16_16_1_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e2_256_30.txt
###
###echo "Training model Lstm_16_16_1_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_16_16_1_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e2_256_35.txt
###
###echo "Training model Lstm_16_16_1_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_16_16_1_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e2_256_40.txt
###
### Learning rate 5e-3
##
##echo "Training model Lstm_16_16_1_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e3_256_20.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e3_256_20.txt
##
##echo "Training model Lstm_16_16_1_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e3_256_25.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e3_256_25.txt
##
##echo "Training model Lstm_16_16_1_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e3_256_30.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e3_256_30.txt
##
##echo "Training model Lstm_16_16_1_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e3_256_35.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e3_256_35.txt
##
##echo "Training model Lstm_16_16_1_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e3_256_40.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_16_16_1_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e3_256_20.txt
##
##echo "Training model Lstm_16_16_1_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e3_256_25.txt
##
##echo "Training model Lstm_16_16_1_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e3_256_30.txt
##
##echo "Training model Lstm_16_16_1_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e3_256_35.txt
##
##echo "Training model Lstm_16_16_1_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_16_16_1_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e4_256_20.txt
##
##echo "Training model Lstm_16_16_1_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e4_256_25.txt
##
##echo "Training model Lstm_16_16_1_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e4_256_30.txt
##
##echo "Training model Lstm_16_16_1_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e4_256_35.txt
##
##echo "Training model Lstm_16_16_1_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e4_256_40.txt
#
### Learning rate 1e-4
##
##echo "Training model Lstm_16_16_1_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e4_256_20.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e4_256_20.txt
##
##echo "Training model Lstm_16_16_1_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e4_256_25.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e4_256_25.txt
##
##echo "Training model Lstm_16_16_1_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e4_256_30.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e4_256_30.txt
##
##echo "Training model Lstm_16_16_1_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e4_256_35.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e4_256_35.txt
##
##echo "Training model Lstm_16_16_1_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_1e4_256_40.txt
##
##echo "Evaluating model Lstm_16_16_1_1_1e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model Lstm_16_16_1_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e5_256_20.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e5_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e5_256_20.txt
##
##echo "Training model Lstm_16_16_1_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e5_256_25.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e5_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e5_256_25.txt
##
##echo "Training model Lstm_16_16_1_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e5_256_30.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e5_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e5_256_30.txt
##
##echo "Training model Lstm_16_16_1_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e5_256_35.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e5_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e5_256_35.txt
##
##echo "Training model Lstm_16_16_1_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_16_16_1_1_5e5_256_40.txt
##
##echo "Evaluating model Lstm_16_16_1_1_5e5_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_16_16_1_1_5e5_256_40.txt
#
### Lstm_128_32_1_1
##
### Learning rate 1e-2
##
###echo "Training model Lstm_128_32_1_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_128_32_1_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e2_256_20.txt
###
###echo "Training model Lstm_128_32_1_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_128_32_1_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e2_256_25.txt
###
###echo "Training model Lstm_128_32_1_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_128_32_1_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e2_256_30.txt
###
###echo "Training model Lstm_128_32_1_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_128_32_1_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e2_256_35.txt
###
###echo "Training model Lstm_128_32_1_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_128_32_1_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e2_256_40.txt
###
### Learning rate 5e-3
##
##echo "Training model Lstm_128_32_1_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e3_256_20.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e3_256_20.txt
##
##echo "Training model Lstm_128_32_1_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e3_256_25.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e3_256_25.txt
##
##echo "Training model Lstm_128_32_1_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e3_256_30.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e3_256_30.txt
##
##echo "Training model Lstm_128_32_1_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e3_256_35.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e3_256_35.txt
##
##echo "Training model Lstm_128_32_1_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e3_256_40.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e3_256_40.txt
#
#### Learning rate 1e-3
##
##echo "Training model Lstm_128_32_1_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e3_256_20.txt
##
##echo "Training model Lstm_128_32_1_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e3_256_25.txt
##
##echo "Training model Lstm_128_32_1_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e3_256_30.txt
##
##echo "Training model Lstm_128_32_1_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e3_256_35.txt
##
##echo "Training model Lstm_128_32_1_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_128_32_1_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e4_256_20.txt
##
##echo "Training model Lstm_128_32_1_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e4_256_25.txt
##
##echo "Training model Lstm_128_32_1_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e4_256_30.txt
##
##echo "Training model Lstm_128_32_1_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e4_256_35.txt
##
##echo "Training model Lstm_128_32_1_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e4_256_40.txt
#
### Learning rate 1e-4
##
##echo "Training model Lstm_128_32_1_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e4_256_20.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e4_256_20.txt
##
##echo "Training model Lstm_128_32_1_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e4_256_25.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e4_256_25.txt
##
##echo "Training model Lstm_128_32_1_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e4_256_30.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e4_256_30.txt
##
##echo "Training model Lstm_128_32_1_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e4_256_35.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e4_256_35.txt
##
##echo "Training model Lstm_128_32_1_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_1e4_256_40.txt
##
##echo "Evaluating model Lstm_128_32_1_1_1e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model Lstm_128_32_1_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e5_256_20.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e5_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e5_256_20.txt
##
##echo "Training model Lstm_128_32_1_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e5_256_25.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e5_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e5_256_25.txt
##
##echo "Training model Lstm_128_32_1_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e5_256_30.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e5_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e5_256_30.txt
##
##echo "Training model Lstm_128_32_1_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e5_256_35.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e5_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e5_256_35.txt
##
##echo "Training model Lstm_128_32_1_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_1_1_5e5_256_40.txt
##
##echo "Evaluating model Lstm_128_32_1_1_5e5_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_1_1 --model_name Lstm_128_32_1_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_1_1_5e5_256_40.txt
#
### Lstm_64_64_5_1
##
### Learning rate 1e-2
##
###echo "Training model Lstm_64_64_5_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_64_64_5_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e2_256_20.txt
###
###echo "Training model Lstm_64_64_5_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_64_64_5_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e2_256_25.txt
###
###echo "Training model Lstm_64_64_5_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_64_64_5_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e2_256_30.txt
###
###echo "Training model Lstm_64_64_5_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_64_64_5_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e2_256_35.txt
###
###echo "Training model Lstm_64_64_5_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_64_64_5_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e2_256_40.txt
###
### Learning rate 5e-3
##
##echo "Training model Lstm_64_64_5_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e3_256_20.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e3_256_20.txt
##
##echo "Training model Lstm_64_64_5_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e3_256_25.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e3_256_25.txt
##
##echo "Training model Lstm_64_64_5_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e3_256_30.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e3_256_30.txt
##
##echo "Training model Lstm_64_64_5_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e3_256_35.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e3_256_35.txt
##
##echo "Training model Lstm_64_64_5_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e3_256_40.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_64_64_5_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e3_256_20.txt
##
##echo "Training model Lstm_64_64_5_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e3_256_25.txt
##
##echo "Training model Lstm_64_64_5_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e3_256_30.txt
##
##echo "Training model Lstm_64_64_5_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e3_256_35.txt
##
##echo "Training model Lstm_64_64_5_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_64_64_5_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e4_256_20.txt
##
##echo "Training model Lstm_64_64_5_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e4_256_25.txt
##
##echo "Training model Lstm_64_64_5_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e4_256_30.txt
##
##echo "Training model Lstm_64_64_5_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e4_256_35.txt
##
##echo "Training model Lstm_64_64_5_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e4_256_40.txt
#
### Learning rate 1e-4
##
##echo "Training model Lstm_64_64_5_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e4_256_20.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e4_256_20.txt
##
##echo "Training model Lstm_64_64_5_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e4_256_25.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e4_256_25.txt
##
##echo "Training model Lstm_64_64_5_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e4_256_30.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e4_256_30.txt
##
##echo "Training model Lstm_64_64_5_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e4_256_35.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e4_256_35.txt
##
##echo "Training model Lstm_64_64_5_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_1e4_256_40.txt
##
##echo "Evaluating model Lstm_64_64_5_1_1e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model Lstm_64_64_5_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e5_256_20.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e5_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e5_256_20.txt
##
##echo "Training model Lstm_64_64_5_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e5_256_25.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e5_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e5_256_25.txt
##
##echo "Training model Lstm_64_64_5_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e5_256_30.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e5_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e5_256_30.txt
##
##echo "Training model Lstm_64_64_5_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e5_256_35.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e5_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e5_256_35.txt
##
##echo "Training model Lstm_64_64_5_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_64_64_5_1_5e5_256_40.txt
##
##echo "Evaluating model Lstm_64_64_5_1_5e5_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_64_64_5_1_5e5_256_40.txt
#
### Lstm_64_32_2_2
##
### Learning rate 1e-2
##
###echo "Training model Lstm_64_32_2_2_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e2_256_20.txt
###
###echo "Evaluating model Lstm_64_32_2_2_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e2_256_20.txt
###
###echo "Training model Lstm_64_32_2_2_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e2_256_25.txt
###
###echo "Evaluating model Lstm_64_32_2_2_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e2_256_25.txt
###
###echo "Training model Lstm_64_32_2_2_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e2_256_30.txt
###
###echo "Evaluating model Lstm_64_32_2_2_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e2_256_30.txt
###
###echo "Training model Lstm_64_32_2_2_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e2_256_35.txt
###
###echo "Evaluating model Lstm_64_32_2_2_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e2_256_35.txt
###
###echo "Training model Lstm_64_32_2_2_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e2_256_40.txt
###
###echo "Evaluating model Lstm_64_32_2_2_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e2_256_40.txt
###
### Learning rate 5e-3
##
##echo "Training model Lstm_64_32_2_2_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e3_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e3_256_20.txt
##
##echo "Training model Lstm_64_32_2_2_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e3_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e3_256_25.txt
##
##echo "Training model Lstm_64_32_2_2_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e3_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e3_256_30.txt
##
##echo "Training model Lstm_64_32_2_2_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e3_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e3_256_35.txt
##
##echo "Training model Lstm_64_32_2_2_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e3_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_64_32_2_2_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e3_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e3_256_20.txt
##
##echo "Training model Lstm_64_32_2_2_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e3_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e3_256_25.txt
##
##echo "Training model Lstm_64_32_2_2_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e3_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e3_256_30.txt
##
##echo "Training model Lstm_64_32_2_2_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e3_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e3_256_35.txt
##
##echo "Training model Lstm_64_32_2_2_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e3_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_64_32_2_2_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e4_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e4_256_20.txt
##
##echo "Training model Lstm_64_32_2_2_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e4_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e4_256_25.txt
##
##echo "Training model Lstm_64_32_2_2_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e4_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e4_256_30.txt
##
##echo "Training model Lstm_64_32_2_2_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e4_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e4_256_35.txt
##
##echo "Training model Lstm_64_32_2_2_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e4_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e4_256_40.txt
#
### Learning rate 1e-4
##
##echo "Training model Lstm_64_32_2_2_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e4_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e4_256_20.txt
##
##echo "Training model Lstm_64_32_2_2_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e4_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e4_256_25.txt
##
##echo "Training model Lstm_64_32_2_2_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e4_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e4_256_30.txt
##
##echo "Training model Lstm_64_32_2_2_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e4_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e4_256_35.txt
##
##echo "Training model Lstm_64_32_2_2_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_1e4_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_2_1e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model Lstm_64_32_2_2_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e5_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e5_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e5_256_20.txt
##
##echo "Training model Lstm_64_32_2_2_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e5_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e5_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e5_256_25.txt
##
##echo "Training model Lstm_64_32_2_2_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e5_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e5_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e5_256_30.txt
##
##echo "Training model Lstm_64_32_2_2_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e5_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e5_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e5_256_35.txt
##
##echo "Training model Lstm_64_32_2_2_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_2_5e5_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_2_5e5_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_2 --model_name Lstm_64_32_2_2_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_2_5e5_256_40.txt
#
### Lstm_128_32_2_1
##
### Learning rate 1e-2
##
###echo "Training model Lstm_128_32_2_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_128_32_2_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e2_256_20.txt
###
###echo "Training model Lstm_128_32_2_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_128_32_2_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e2_256_25.txt
###
###echo "Training model Lstm_128_32_2_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_128_32_2_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e2_256_30.txt
###
###echo "Training model Lstm_128_32_2_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_128_32_2_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e2_256_35.txt
###
###echo "Training model Lstm_128_32_2_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_128_32_2_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e2_256_40.txt
###
### Learning rate 5e-3
##
##echo "Training model Lstm_128_32_2_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e3_256_20.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e3_256_20.txt
##
##echo "Training model Lstm_128_32_2_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e3_256_25.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e3_256_25.txt
##
##echo "Training model Lstm_128_32_2_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e3_256_30.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e3_256_30.txt
##
##echo "Training model Lstm_128_32_2_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e3_256_35.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e3_256_35.txt
##
##echo "Training model Lstm_128_32_2_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e3_256_40.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_128_32_2_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e3_256_20.txt
##
##echo "Training model Lstm_128_32_2_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e3_256_25.txt
##
##echo "Training model Lstm_128_32_2_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e3_256_30.txt
##
##echo "Training model Lstm_128_32_2_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e3_256_35.txt
##
##echo "Training model Lstm_128_32_2_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_128_32_2_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e4_256_20.txt
##
##echo "Training model Lstm_128_32_2_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e4_256_25.txt
##
##echo "Training model Lstm_128_32_2_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e4_256_30.txt
##
##echo "Training model Lstm_128_32_2_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e4_256_35.txt
##
##echo "Training model Lstm_128_32_2_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e4_256_40.txt
#
### Learning rate 1e-4
##
##echo "Training model Lstm_128_32_2_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e4_256_20.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e4_256_20.txt
##
##echo "Training model Lstm_128_32_2_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e4_256_25.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e4_256_25.txt
##
##echo "Training model Lstm_128_32_2_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e4_256_30.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e4_256_30.txt
##
##echo "Training model Lstm_128_32_2_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e4_256_35.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e4_256_35.txt
##
##echo "Training model Lstm_128_32_2_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_1e4_256_40.txt
##
##echo "Evaluating model Lstm_128_32_2_1_1e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model Lstm_128_32_2_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e5_256_20.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e5_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e5_256_20.txt
##
##echo "Training model Lstm_128_32_2_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e5_256_25.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e5_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e5_256_25.txt
##
##echo "Training model Lstm_128_32_2_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e5_256_30.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e5_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e5_256_30.txt
##
##echo "Training model Lstm_128_32_2_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e5_256_35.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e5_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e5_256_35.txt
##
##echo "Training model Lstm_128_32_2_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_128_32_2_1_5e5_256_40.txt
##
##echo "Evaluating model Lstm_128_32_2_1_5e5_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_128_32_2_1 --model_name Lstm_128_32_2_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_128_32_2_1_5e5_256_40.txt
#
### Lstm_256_128_10_1
##
### Learning rate 1e-2
##
###echo "Training model Lstm_256_128_10_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_256_128_10_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e2_256_20.txt
###
###echo "Training model Lstm_256_128_10_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_256_128_10_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e2_256_25.txt
###
###echo "Training model Lstm_256_128_10_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_256_128_10_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e2_256_30.txt
###
###echo "Training model Lstm_256_128_10_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_256_128_10_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e2_256_35.txt
###
###echo "Training model Lstm_256_128_10_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_256_128_10_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e2_256_40.txt
###
### Learning rate 5e-3
##
##echo "Training model Lstm_256_128_10_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e3_256_20.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e3_256_20.txt
##
##echo "Training model Lstm_256_128_10_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e3_256_25.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e3_256_25.txt
##
##echo "Training model Lstm_256_128_10_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e3_256_30.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e3_256_30.txt
##
##echo "Training model Lstm_256_128_10_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e3_256_35.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e3_256_35.txt
##
##echo "Training model Lstm_256_128_10_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e3_256_40.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_256_128_10_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e3_256_20.txt
##
##echo "Training model Lstm_256_128_10_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e3_256_25.txt
##
##echo "Training model Lstm_256_128_10_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e3_256_30.txt
##
##echo "Training model Lstm_256_128_10_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e3_256_35.txt
##
##echo "Training model Lstm_256_128_10_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_256_128_10_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e4_256_20.txt
##
##echo "Training model Lstm_256_128_10_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e4_256_25.txt
##
##echo "Training model Lstm_256_128_10_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e4_256_30.txt
##
##echo "Training model Lstm_256_128_10_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e4_256_35.txt
##
##echo "Training model Lstm_256_128_10_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e4_256_40.txt
#
### Learning rate 1e-4
##
##echo "Training model Lstm_256_128_10_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e4_256_20.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e4_256_20.txt
##
##echo "Training model Lstm_256_128_10_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e4_256_25.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e4_256_25.txt
##
##echo "Training model Lstm_256_128_10_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e4_256_30.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e4_256_30.txt
##
##echo "Training model Lstm_256_128_10_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e4_256_35.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e4_256_35.txt
##
##echo "Training model Lstm_256_128_10_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_1e4_256_40.txt
##
##echo "Evaluating model Lstm_256_128_10_1_1e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model Lstm_256_128_10_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e5_256_20.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e5_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e5_256_20.txt
##
##echo "Training model Lstm_256_128_10_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e5_256_25.txt
##
##echo "Evaluating model Lstm_256_128_10_1_5e5_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e5_256_25.txt
#
#echo "Training model Lstm_256_128_10_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e5_256_30.txt
#
#echo "Evaluating model Lstm_256_128_10_1_5e5_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e5_256_30.txt
#
#echo "Training model Lstm_256_128_10_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e5_256_35.txt
#
#echo "Evaluating model Lstm_256_128_10_1_5e5_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e5_256_35.txt
#
#echo "Training model Lstm_256_128_10_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_256_128_10_1_5e5_256_40.txt
#
#echo "Evaluating model Lstm_256_128_10_1_5e5_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_256_128_10_1 --model_name Lstm_256_128_10_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_256_128_10_1_5e5_256_40.txt
#
### Lstm_64_32_2_1
##
### Learning rate 1e-2
##
###echo "Training model Lstm_64_32_2_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_64_32_2_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e2_256_20.txt
###
###echo "Training model Lstm_64_32_2_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_64_32_2_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e2_256_25.txt
###
###echo "Training model Lstm_64_32_2_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_64_32_2_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e2_256_30.txt
###
###echo "Training model Lstm_64_32_2_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_64_32_2_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e2_256_35.txt
###
###echo "Training model Lstm_64_32_2_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_64_32_2_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e2_256_40.txt
###
## Learning rate 5e-3
#
#echo "Training model Lstm_64_32_2_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e3_256_20.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e3_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e3_256_20.txt
#
#echo "Training model Lstm_64_32_2_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e3_256_25.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e3_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e3_256_25.txt
#
#echo "Training model Lstm_64_32_2_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e3_256_30.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e3_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e3_256_30.txt
#
#echo "Training model Lstm_64_32_2_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e3_256_35.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e3_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e3_256_35.txt
#
#echo "Training model Lstm_64_32_2_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e3_256_40.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e3_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_64_32_2_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e3_256_20.txt
##
##echo "Training model Lstm_64_32_2_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e3_256_25.txt
##
##echo "Training model Lstm_64_32_2_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e3_256_30.txt
##
##echo "Training model Lstm_64_32_2_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e3_256_35.txt
##
##echo "Training model Lstm_64_32_2_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_64_32_2_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_64_32_2_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e4_256_20.txt
##
##echo "Training model Lstm_64_32_2_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_64_32_2_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e4_256_25.txt
##
##echo "Training model Lstm_64_32_2_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_64_32_2_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e4_256_30.txt
##
##echo "Training model Lstm_64_32_2_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_64_32_2_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e4_256_35.txt
##
##echo "Training model Lstm_64_32_2_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_64_32_2_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Lstm_64_32_2_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e4_256_20.txt
#
#echo "Evaluating model Lstm_64_32_2_1_1e4_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e4_256_20.txt
#
#echo "Training model Lstm_64_32_2_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e4_256_25.txt
#
#echo "Evaluating model Lstm_64_32_2_1_1e4_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e4_256_25.txt
#
#echo "Training model Lstm_64_32_2_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e4_256_30.txt
#
#echo "Evaluating model Lstm_64_32_2_1_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e4_256_30.txt
#
#echo "Training model Lstm_64_32_2_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e4_256_35.txt
#
#echo "Evaluating model Lstm_64_32_2_1_1e4_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e4_256_35.txt
#
#echo "Training model Lstm_64_32_2_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_1e4_256_40.txt
#
#echo "Evaluating model Lstm_64_32_2_1_1e4_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Lstm_64_32_2_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e5_256_20.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e5_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e5_256_20.txt
#
#echo "Training model Lstm_64_32_2_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e5_256_25.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e5_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e5_256_25.txt
#
#echo "Training model Lstm_64_32_2_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e5_256_30.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e5_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e5_256_30.txt
#
#echo "Training model Lstm_64_32_2_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e5_256_35.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e5_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e5_256_35.txt
#
#echo "Training model Lstm_64_32_2_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_64_32_2_1_5e5_256_40.txt
#
#echo "Evaluating model Lstm_64_32_2_1_5e5_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_64_32_2_1 --model_name Lstm_64_32_2_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_64_32_2_1_5e5_256_40.txt
#
### Lstm_32_32_2_1
##
#### Learning rate 1e-2
###
###echo "Training model Lstm_32_32_2_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_32_32_2_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e2_256_20.txt
###
###echo "Training model Lstm_32_32_2_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_32_32_2_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e2_256_25.txt
###
###echo "Training model Lstm_32_32_2_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_32_32_2_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e2_256_30.txt
###
###echo "Training model Lstm_32_32_2_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_32_32_2_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e2_256_35.txt
###
###echo "Training model Lstm_32_32_2_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_32_32_2_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e2_256_40.txt
###
## Learning rate 5e-3
#
#echo "Training model Lstm_32_32_2_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e3_256_20.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e3_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e3_256_20.txt
#
#echo "Training model Lstm_32_32_2_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e3_256_25.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e3_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e3_256_25.txt
#
#echo "Training model Lstm_32_32_2_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e3_256_30.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e3_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e3_256_30.txt
#
#echo "Training model Lstm_32_32_2_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e3_256_35.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e3_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e3_256_35.txt
#
#echo "Training model Lstm_32_32_2_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e3_256_40.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e3_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_32_32_2_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_32_32_2_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e3_256_20.txt
##
##echo "Training model Lstm_32_32_2_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_32_32_2_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e3_256_25.txt
##
##echo "Training model Lstm_32_32_2_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_32_32_2_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e3_256_30.txt
##
##echo "Training model Lstm_32_32_2_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_32_32_2_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e3_256_35.txt
##
##echo "Training model Lstm_32_32_2_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_32_32_2_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_32_32_2_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_32_32_2_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e4_256_20.txt
##
##echo "Training model Lstm_32_32_2_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_32_32_2_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e4_256_25.txt
##
##echo "Training model Lstm_32_32_2_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_32_32_2_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e4_256_30.txt
##
##echo "Training model Lstm_32_32_2_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_32_32_2_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e4_256_35.txt
##
##echo "Training model Lstm_32_32_2_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_32_32_2_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Lstm_32_32_2_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e4_256_20.txt
#
#echo "Evaluating model Lstm_32_32_2_1_1e4_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e4_256_20.txt
#
#echo "Training model Lstm_32_32_2_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e4_256_25.txt
#
#echo "Evaluating model Lstm_32_32_2_1_1e4_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e4_256_25.txt
#
#echo "Training model Lstm_32_32_2_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e4_256_30.txt
#
#echo "Evaluating model Lstm_32_32_2_1_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e4_256_30.txt
#
#echo "Training model Lstm_32_32_2_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e4_256_35.txt
#
#echo "Evaluating model Lstm_32_32_2_1_1e4_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e4_256_35.txt
#
#echo "Training model Lstm_32_32_2_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_1e4_256_40.txt
#
#echo "Evaluating model Lstm_32_32_2_1_1e4_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Lstm_32_32_2_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e5_256_20.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e5_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e5_256_20.txt
#
#echo "Training model Lstm_32_32_2_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e5_256_25.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e5_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e5_256_25.txt
#
#echo "Training model Lstm_32_32_2_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e5_256_30.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e5_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e5_256_30.txt
#
#echo "Training model Lstm_32_32_2_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e5_256_35.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e5_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e5_256_35.txt
#
#echo "Training model Lstm_32_32_2_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_32_32_2_1_5e5_256_40.txt
#
#echo "Evaluating model Lstm_32_32_2_1_5e5_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_32_2_1 --model_name Lstm_32_32_2_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_32_32_2_1_5e5_256_40.txt
#
### Lstm_32_64_1_1
##
#### Learning rate 1e-2
###
###echo "Training model Lstm_32_64_1_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_32_64_1_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e2_256_20.txt
###
###echo "Training model Lstm_32_64_1_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_32_64_1_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e2_256_25.txt
###
###echo "Training model Lstm_32_64_1_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_32_64_1_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e2_256_30.txt
###
###echo "Training model Lstm_32_64_1_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_32_64_1_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e2_256_35.txt
###
###echo "Training model Lstm_32_64_1_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_32_64_1_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e2_256_40.txt
###
## Learning rate 5e-3
#
#echo "Training model Lstm_32_64_1_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e3_256_20.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e3_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e3_256_20.txt
#
#echo "Training model Lstm_32_64_1_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e3_256_25.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e3_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e3_256_25.txt
#
#echo "Training model Lstm_32_64_1_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e3_256_30.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e3_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e3_256_30.txt
#
#echo "Training model Lstm_32_64_1_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e3_256_35.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e3_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e3_256_35.txt
#
#echo "Training model Lstm_32_64_1_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e3_256_40.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e3_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_32_64_1_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_32_64_1_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e3_256_20.txt
##
##echo "Training model Lstm_32_64_1_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_32_64_1_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e3_256_25.txt
##
##echo "Training model Lstm_32_64_1_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_32_64_1_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e3_256_30.txt
##
##echo "Training model Lstm_32_64_1_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_32_64_1_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e3_256_35.txt
##
##echo "Training model Lstm_32_64_1_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_32_64_1_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_32_64_1_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_32_64_1_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e4_256_20.txt
##
##echo "Training model Lstm_32_64_1_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_32_64_1_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e4_256_25.txt
##
##echo "Training model Lstm_32_64_1_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_32_64_1_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e4_256_30.txt
##
##echo "Training model Lstm_32_64_1_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_32_64_1_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e4_256_35.txt
##
##echo "Training model Lstm_32_64_1_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e4_256_40.txt
##
##echo "Evaluating model Lstm_32_64_1_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Lstm_32_64_1_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e4_256_20.txt
#
#echo "Evaluating model Lstm_32_64_1_1_1e4_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e4_256_20.txt
#
#echo "Training model Lstm_32_64_1_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e4_256_25.txt
#
#echo "Evaluating model Lstm_32_64_1_1_1e4_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e4_256_25.txt
#
#echo "Training model Lstm_32_64_1_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e4_256_30.txt
#
#echo "Evaluating model Lstm_32_64_1_1_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e4_256_30.txt
#
#echo "Training model Lstm_32_64_1_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e4_256_35.txt
#
#echo "Evaluating model Lstm_32_64_1_1_1e4_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e4_256_35.txt
#
#echo "Training model Lstm_32_64_1_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_1e4_256_40.txt
#
#echo "Evaluating model Lstm_32_64_1_1_1e4_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Lstm_32_64_1_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e5_256_20.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e5_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e5_256_20.txt
#
#echo "Training model Lstm_32_64_1_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e5_256_25.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e5_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e5_256_25.txt
#
#echo "Training model Lstm_32_64_1_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e5_256_30.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e5_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e5_256_30.txt
#
#echo "Training model Lstm_32_64_1_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e5_256_35.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e5_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e5_256_35.txt
#
#echo "Training model Lstm_32_64_1_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_32_64_1_1_5e5_256_40.txt
#
#echo "Evaluating model Lstm_32_64_1_1_5e5_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_32_64_1_1 --model_name Lstm_32_64_1_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_32_64_1_1_5e5_256_40.txt
#
### Lstm_16_128_10_1
##
#### Learning rate 1e-2
###
###echo "Training model Lstm_16_128_10_1_1e2_256_20, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 20 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_20 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e2_256_20.txt
###
###echo "Evaluating model Lstm_16_128_10_1_1e2_256_20"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_20 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e2_256_20.txt
###
###echo "Training model Lstm_16_128_10_1_1e2_256_25, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 25 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_25 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e2_256_25.txt
###
###echo "Evaluating model Lstm_16_128_10_1_1e2_256_25"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_25 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e2_256_25.txt
###
###echo "Training model Lstm_16_128_10_1_1e2_256_30, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 30 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_30 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e2_256_30.txt
###
###echo "Evaluating model Lstm_16_128_10_1_1e2_256_30"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_30 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e2_256_30.txt
###
###echo "Training model Lstm_16_128_10_1_1e2_256_35, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 35 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_35 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e2_256_35.txt
###
###echo "Evaluating model Lstm_16_128_10_1_1e2_256_35"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_35 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e2_256_35.txt
###
###echo "Training model Lstm_16_128_10_1_1e2_256_40, lr = 1e-2, epochs = 5, batch_size = 256"
###python ../train_validation.py \
###              --model_folder step4 --patience 40 \
###              --train_path $trn --val_path $val      \
###              --n_epochs 5 --lr 1e-2 --batch_size 256 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_40 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e2_256_40.txt
###
###echo "Evaluating model Lstm_16_128_10_1_1e2_256_40"
###python ../eval_curves.py --test_path $tst \
###              --model_folder step4 \
###              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e2_256_40 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e2_256_40.txt
###
## Learning rate 5e-3
#
#echo "Training model Lstm_16_128_10_1_5e3_256_20, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_20 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e3_256_20.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e3_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_20 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e3_256_20.txt
#
#echo "Training model Lstm_16_128_10_1_5e3_256_25, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_25 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e3_256_25.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e3_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_25 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e3_256_25.txt
#
#echo "Training model Lstm_16_128_10_1_5e3_256_30, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_30 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e3_256_30.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e3_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_30 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e3_256_30.txt
#
#echo "Training model Lstm_16_128_10_1_5e3_256_35, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_35 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e3_256_35.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e3_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_35 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e3_256_35.txt
#
#echo "Training model Lstm_16_128_10_1_5e3_256_40, lr = 5e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-3 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_40 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e3_256_40.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e3_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e3_256_40 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e3_256_40.txt
#
### Learning rate 1e-3
##
##echo "Training model Lstm_16_128_10_1_1e3_256_20, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_20 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e3_256_20.txt
##
##echo "Evaluating model Lstm_16_128_10_1_1e3_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_20 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e3_256_20.txt
##
##echo "Training model Lstm_16_128_10_1_1e3_256_25, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_25 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e3_256_25.txt
##
##echo "Evaluating model Lstm_16_128_10_1_1e3_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_25 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e3_256_25.txt
##
##echo "Training model Lstm_16_128_10_1_1e3_256_30, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_30 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e3_256_30.txt
##
##echo "Evaluating model Lstm_16_128_10_1_1e3_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_30 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e3_256_30.txt
##
##echo "Training model Lstm_16_128_10_1_1e3_256_35, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_35 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e3_256_35.txt
##
##echo "Evaluating model Lstm_16_128_10_1_1e3_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_35 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e3_256_35.txt
##
##echo "Training model Lstm_16_128_10_1_1e3_256_40, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_40 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e3_256_40.txt
##
##echo "Evaluating model Lstm_16_128_10_1_1e3_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e3_256_40 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model Lstm_16_128_10_1_5e4_256_20, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_20 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e4_256_20.txt
##
##echo "Evaluating model Lstm_16_128_10_1_5e4_256_20"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_20 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e4_256_20.txt
##
##echo "Training model Lstm_16_128_10_1_5e4_256_25, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_25 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e4_256_25.txt
##
##echo "Evaluating model Lstm_16_128_10_1_5e4_256_25"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_25 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e4_256_25.txt
##
##echo "Training model Lstm_16_128_10_1_5e4_256_30, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_30 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e4_256_30.txt
##
##echo "Evaluating model Lstm_16_128_10_1_5e4_256_30"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_30 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e4_256_30.txt
##
##echo "Training model Lstm_16_128_10_1_5e4_256_35, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_35 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e4_256_35.txt
##
##echo "Evaluating model Lstm_16_128_10_1_5e4_256_35"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_35 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e4_256_35.txt
#
##echo "Training model Lstm_16_128_10_1_5e4_256_40, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_40 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e4_256_40.txt
#
##echo "Evaluating model Lstm_16_128_10_1_5e4_256_40"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e4_256_40 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Lstm_16_128_10_1_1e4_256_20, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_20 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e4_256_20.txt
#
#echo "Evaluating model Lstm_16_128_10_1_1e4_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_20 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e4_256_20.txt
#
#echo "Training model Lstm_16_128_10_1_1e4_256_25, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_25 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e4_256_25.txt
#
#echo "Evaluating model Lstm_16_128_10_1_1e4_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_25 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e4_256_25.txt
#
#echo "Training model Lstm_16_128_10_1_1e4_256_30, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_30 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e4_256_30.txt
#
#echo "Evaluating model Lstm_16_128_10_1_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_30 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e4_256_30.txt
#
#echo "Training model Lstm_16_128_10_1_1e4_256_35, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_35 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e4_256_35.txt
#
#echo "Evaluating model Lstm_16_128_10_1_1e4_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_35 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e4_256_35.txt
#
#echo "Training model Lstm_16_128_10_1_1e4_256_40, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_40 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_1e4_256_40.txt
#
#echo "Evaluating model Lstm_16_128_10_1_1e4_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_1e4_256_40 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Lstm_16_128_10_1_5e5_256_20, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_20 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e5_256_20.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e5_256_20"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_20 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e5_256_20.txt
#
#echo "Training model Lstm_16_128_10_1_5e5_256_25, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_25 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e5_256_25.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e5_256_25"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_25 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e5_256_25.txt
#
#echo "Training model Lstm_16_128_10_1_5e5_256_30, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_30 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e5_256_30.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e5_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_30 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e5_256_30.txt
#
#echo "Training model Lstm_16_128_10_1_5e5_256_35, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_35 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e5_256_35.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e5_256_35"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_35 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e5_256_35.txt
#
#echo "Training model Lstm_16_128_10_1_5e5_256_40, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_40 > ../Analysis/logs/train/step4/Lstm_16_128_10_1_5e5_256_40.txt
#
#echo "Evaluating model Lstm_16_128_10_1_5e5_256_40"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Lstm_16_128_10_1 --model_name Lstm_16_128_10_1_5e5_256_40 > ../Analysis/logs/eval/step4/Lstm_16_128_10_1_5e5_256_40.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../trainevalcurves2excel.py --xls_name 'LSTM_step4' --archives_folder 'step4' --best 40