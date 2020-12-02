#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

## Cnn1_2k_1h
#
## Learning rate 1e-3
#echo "Training model Cnn1_2k_1h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e3_256_20.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e3_256_25.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e3_256_30.txt
#
#
#echo "Training model Cnn1_2k_1h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e3_256_35.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_2k_1h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e4_256_20.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e4_256_25.txt
#
#echo "Training model Cnn1_2k_1h, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e4_256_30.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e4_256_35.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e4_256_20.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e4_256_25.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e4_256_30.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e4_256_35.txt
#
#echo "Training model Cnn1_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_2k_1h_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_2k_1h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e5_256_20.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e5_256_25.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e5_256_30.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e5_256_35.txt
#
#echo "Training model Cnn1_2k_1h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_2k_1h_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_2k_1h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_2k_1h_5e5_256_40.txt
#
## Cnn1_3k
#
## Learning rate 1e-3
#echo "Training model Cnn1_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_1e3_256_20.txt
#
#echo "Training model Cnn1_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_1e3_256_25.txt
#
#echo "Training model Cnn1_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_1e3_256_30.txt
#
#
#echo "Training model Cnn1_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_1e3_256_35.txt
#
#echo "Training model Cnn1_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_5e4_256_20.txt
#
#echo "Training model Cnn1_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_5e4_256_25.txt
#
#echo "Training model Cnn1_3k, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_5e4_256_30.txt
#
#echo "Training model Cnn1_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_5e4_256_35.txt
#
#echo "Training model Cnn1_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_1e4_256_20.txt
#
#echo "Training model Cnn1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_1e4_256_25.txt
#
#echo "Training model Cnn1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_1e4_256_30.txt
#
#echo "Training model Cnn1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_1e4_256_35.txt
#
#echo "Training model Cnn1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_5e5_256_20.txt
#
#echo "Training model Cnn1_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_5e5_256_25.txt
#
#echo "Training model Cnn1_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_5e5_256_30.txt
#
#echo "Training model Cnn1_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_5e5_256_35.txt
#
#echo "Training model Cnn1_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_5e5_256_40.txt
#
## Cnn1_1k_2k
#
## Learning rate 1e-3
#echo "Training model Cnn1_1k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e3_256_20.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e3_256_25.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e3_256_30.txt
#
#
#echo "Training model Cnn1_1k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e3_256_35.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_1k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e4_256_20.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e4_256_25.txt
#
#echo "Training model Cnn1_1k_2k, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e4_256_30.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e4_256_35.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e4_256_20.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e4_256_25.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e4_256_30.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e4_256_35.txt
#
#echo "Training model Cnn1_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_2k_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_1k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e5_256_20.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e5_256_25.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e5_256_30.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e5_256_35.txt
#
#echo "Training model Cnn1_1k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_2k_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_1k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_2k_5e5_256_40.txt
#
## Cnn1_5k_5h
#
## Learning rate 1e-3
#echo "Training model Cnn1_5k_5h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e3_256_20.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e3_256_25.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e3_256_30.txt
#
#
#echo "Training model Cnn1_5k_5h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e3_256_35.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_5k_5h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e4_256_20.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e4_256_25.txt
#
#echo "Training model Cnn1_5k_5h, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e4_256_30.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e4_256_35.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e4_256_20.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e4_256_25.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e4_256_30.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e4_256_35.txt
#
#echo "Training model Cnn1_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_5h_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_5k_5h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e5_256_20.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e5_256_25.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e5_256_30.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e5_256_35.txt
#
#echo "Training model Cnn1_5k_5h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_5h_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_5k_5h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_5h_5e5_256_40.txt
#
## Cnn1_5k_3k
#
## Learning rate 1e-3
#echo "Training model Cnn1_5k_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e3_256_20.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e3_256_25.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e3_256_30.txt
#
#
#echo "Training model Cnn1_5k_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e3_256_35.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_5k_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e4_256_20.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e4_256_25.txt
#
#echo "Training model Cnn1_5k_3k, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e4_256_30.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e4_256_35.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e4_256_20.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e4_256_25.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e4_256_30.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e4_256_35.txt
#
#echo "Training model Cnn1_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_3k_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_5k_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e5_256_20.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e5_256_25.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e5_256_30.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e5_256_35.txt
#
#echo "Training model Cnn1_5k_3k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_3k_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_5k_3k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_3k_5e5_256_40.txt
#
## Cnn1_6k_5k
#
## Learning rate 1e-3
#echo "Training model Cnn1_6k_5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e3_256_20.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e3_256_25.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e3_256_30.txt
#
#
#echo "Training model Cnn1_6k_5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e3_256_35.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_6k_5k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e4_256_20.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e4_256_25.txt
#
#echo "Training model Cnn1_6k_5k, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e4_256_30.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e4_256_35.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e4_256_20.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e4_256_25.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e4_256_30.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e4_256_35.txt
#
#echo "Training model Cnn1_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_5k_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_6k_5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e5_256_20.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e5_256_25.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e5_256_30.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e5_256_35.txt
#
#echo "Training model Cnn1_6k_5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_5k_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_6k_5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_5k_5e5_256_40.txt
#
## Cnn1_5k_2k
#
## Learning rate 1e-3
echo "Training model Cnn1_5k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --model_folder step4 --patience 20 \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e3_256_20.txt

echo "Evaluating model Cnn1_5k_2k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --model_folder step4 \
              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e3_256_20.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e3_256_25.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e3_256_30.txt
#
#
#echo "Training model Cnn1_5k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e3_256_35.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_5k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e4_256_20.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e4_256_25.txt
#
#echo "Training model Cnn1_5k_2k, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e4_256_30.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e4_256_35.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e4_256_20.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e4_256_25.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e4_256_30.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e4_256_35.txt
#
#echo "Training model Cnn1_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_2k_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_5k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e5_256_20.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e5_256_25.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e5_256_30.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e5_256_35.txt
#
#echo "Training model Cnn1_5k_2k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_5k_2k_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_5k_2k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_5k_2k_5e5_256_40.txt
#
## Cnn1_3k_10
#
## Learning rate 1e-3
#echo "Training model Cnn1_3k_10, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_3k_10_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e3_256_20.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_3k_10_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e3_256_25.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_3k_10_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e3_256_30.txt
#
#
#echo "Training model Cnn1_3k_10, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_3k_10_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e3_256_35.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_3k_10_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_3k_10, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_3k_10_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e4_256_20.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_3k_10_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e4_256_25.txt
#
#echo "Training model Cnn1_3k_10, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_10_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e4_256_30.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_3k_10_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e4_256_35.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_3k_10_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_3k_10_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e4_256_20.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_3k_10_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e4_256_25.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_10_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e4_256_30.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_3k_10_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e4_256_35.txt
#
#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_10_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_3k_10_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_10_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_3k_10, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_3k_10_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e5_256_20.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_3k_10_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e5_256_25.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_3k_10_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e5_256_30.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_3k_10_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e5_256_35.txt
#
#echo "Training model Cnn1_3k_10, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_3k_10_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_3k_10_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_3k_10_5e5_256_40.txt
#
## Cnn1_6k_2h
#
## Learning rate 1e-3
#echo "Training model Cnn1_6k_2h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e3_256_20.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e3_256_25.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e3_256_30.txt
#
#
#echo "Training model Cnn1_6k_2h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e3_256_35.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_6k_2h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e4_256_20.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e4_256_25.txt
#
#echo "Training model Cnn1_6k_2h, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e4_256_30.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e4_256_35.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e4_256_20.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e4_256_25.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e4_256_30.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e4_256_35.txt
#
#echo "Training model Cnn1_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_2h_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_6k_2h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e5_256_20.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e5_256_25.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e5_256_30.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e5_256_35.txt
#
#echo "Training model Cnn1_6k_2h, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_6k_2h_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_6k_2h_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_6k_2h_5e5_256_40.txt
#
## Cnn1_1k_1k
#
## Learning rate 1e-3
#echo "Training model Cnn1_1k_1k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e3_256_20.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e3_256_20.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e3_256_25.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e3_256_25.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e3_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e3_256_30.txt
#
#
#echo "Training model Cnn1_1k_1k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e3_256_35.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e3_256_35.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e3_256_40.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e3_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e3_256_40.txt
#
## Learning rate 5e-4
#
#echo "Training model Cnn1_1k_1k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e4_256_20.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e4_256_20.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e4_256_25.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e4_256_25.txt
#
#echo "Training model Cnn1_1k_1k, lr = 4e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e4_256_30.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e4_256_35.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e4_256_35.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e4_256_40.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model Cnn1_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e4_256_20.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e4_256_20.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e4_256_25.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e4_256_25.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e4_256_30.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e4_256_35.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e4_256_35.txt
#
#echo "Training model Cnn1_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_1k_1e4_256_40.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model Cnn1_1k_1k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_20 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e5_256_20.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_20 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e5_256_20.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_25 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e5_256_25.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_25 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e5_256_25.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_30 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e5_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_30 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e5_256_30.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_35 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e5_256_35.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_35 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e5_256_35.txt
#
#echo "Training model Cnn1_1k_1k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_40 > ../Analysis/logs/train/step4/Cnn1_1k_1k_5e5_256_40.txt
#
#echo "Evaluating model Cnn1_1k_1k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_5e5_256_40 > ../Analysis/logs/eval/step4/Cnn1_1k_1k_5e5_256_40.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../trainevalcurves2excel.py --xls_name 'CNN_step4' --archives_folder 'step4' --best 30
