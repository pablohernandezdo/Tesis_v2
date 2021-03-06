#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

## 2h5h5k
#
## Learning rate 1e-3
##echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_20.txt
##
##echo "Evaluating model 2h5h5k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_20.txt
##
##echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_25 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_25.txt
##
##echo "Evaluating model 2h5h5k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_25 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_25.txt
##
##echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_30 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_30.txt
##
##echo "Evaluating model 2h5h5k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_30 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_30.txt
##
##
##echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_35 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_35.txt
##
##echo "Evaluating model 2h5h5k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_35 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_35.txt
##
##echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_40 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_40.txt
##
##echo "Evaluating model 2h5h5k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_40 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model 2h5h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_20 > ../Analysis/logs/train/step4/2h5h5k_5e4_256_20.txt
##
##echo "Evaluating model 2h5h5k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_20 > ../Analysis/logs/eval/step4/2h5h5k_5e4_256_20.txt
##
##echo "Training model 2h5h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_25 > ../Analysis/logs/train/step4/2h5h5k_5e4_256_25.txt
##
##echo "Evaluating model 2h5h5k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_25 > ../Analysis/logs/eval/step4/2h5h5k_5e4_256_25.txt
##
##echo "Training model 2h5h5k, lr = 4e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_30 > ../Analysis/logs/train/step4/2h5h5k_5e4_256_30.txt
##
##echo "Evaluating model 2h5h5k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_30 > ../Analysis/logs/eval/step4/2h5h5k_5e4_256_30.txt
###
##echo "Training model 2h5h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_35 > ../Analysis/logs/train/step4/2h5h5k_5e4_256_35.txt
##
##echo "Evaluating model 2h5h5k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_35 > ../Analysis/logs/eval/step4/2h5h5k_5e4_256_35.txt
##
##echo "Training model 2h5h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_40 > ../Analysis/logs/train/step4/2h5h5k_5e4_256_40.txt
##
##echo "Evaluating model 2h5h5k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e4_256_40 > ../Analysis/logs/eval/step4/2h5h5k_5e4_256_40.txt
##
### Learning rate 1e-4
##
##echo "Training model 2h5h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_20 > ../Analysis/logs/train/step4/2h5h5k_1e4_256_20.txt
##
##echo "Evaluating model 2h5h5k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_20 > ../Analysis/logs/eval/step4/2h5h5k_1e4_256_20.txt
##
##echo "Training model 2h5h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_25 > ../Analysis/logs/train/step4/2h5h5k_1e4_256_25.txt
##
##echo "Evaluating model 2h5h5k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_25 > ../Analysis/logs/eval/step4/2h5h5k_1e4_256_25.txt
##
##echo "Training model 2h5h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_30 > ../Analysis/logs/train/step4/2h5h5k_1e4_256_30.txt
##
##echo "Evaluating model 2h5h5k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_30 > ../Analysis/logs/eval/step4/2h5h5k_1e4_256_30.txt
##
##echo "Training model 2h5h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_35 > ../Analysis/logs/train/step4/2h5h5k_1e4_256_35.txt
##
##echo "Evaluating model 2h5h5k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_35 > ../Analysis/logs/eval/step4/2h5h5k_1e4_256_35.txt
##
##echo "Training model 2h5h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_40 > ../Analysis/logs/train/step4/2h5h5k_1e4_256_40.txt
##
##echo "Evaluating model 2h5h5k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_40 > ../Analysis/logs/eval/step4/2h5h5k_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model 2h5h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_20 > ../Analysis/logs/train/step4/2h5h5k_5e5_256_20.txt
##
##echo "Evaluating model 2h5h5k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_20 > ../Analysis/logs/eval/step4/2h5h5k_5e5_256_20.txt
##
##echo "Training model 2h5h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_25 > ../Analysis/logs/train/step4/2h5h5k_5e5_256_25.txt
##
##echo "Evaluating model 2h5h5k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_25 > ../Analysis/logs/eval/step4/2h5h5k_5e5_256_25.txt
##
##echo "Training model 2h5h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_30 > ../Analysis/logs/train/step4/2h5h5k_5e5_256_30.txt
##
##echo "Evaluating model 2h5h5k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_30 > ../Analysis/logs/eval/step4/2h5h5k_5e5_256_30.txt
##
##echo "Training model 2h5h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_35 > ../Analysis/logs/train/step4/2h5h5k_5e5_256_35.txt
##
##echo "Evaluating model 2h5h5k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_35 > ../Analysis/logs/eval/step4/2h5h5k_5e5_256_35.txt
##
##echo "Training model 2h5h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_40 > ../Analysis/logs/train/step4/2h5h5k_5e5_256_40.txt
##
##echo "Evaluating model 2h5h5k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h5k --model_name 2h5h5k_5e5_256_40 > ../Analysis/logs/eval/step4/2h5h5k_5e5_256_40.txt
##
### 2h5h6k
### Learning rate 1e-3
##
##echo "Training model 2h5h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_20 > ../Analysis/logs/train/step4/2h5h6k_1e3_256_20.txt
##
##echo "Evaluating model 2h5h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_20 > ../Analysis/logs/eval/step4/2h5h6k_1e3_256_20.txt
##
##echo "Training model 2h5h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_25 > ../Analysis/logs/train/step4/2h5h6k_1e3_256_25.txt
##
##echo "Evaluating model 2h5h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_25 > ../Analysis/logs/eval/step4/2h5h6k_1e3_256_25.txt
##
##echo "Training model 2h5h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_30 > ../Analysis/logs/train/step4/2h5h6k_1e3_256_30.txt
##
##echo "Evaluating model 2h5h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_30 > ../Analysis/logs/eval/step4/2h5h6k_1e3_256_30.txt
##
##echo "Training model 2h5h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_35 > ../Analysis/logs/train/step4/2h5h6k_1e3_256_35.txt
##
##echo "Evaluating model 2h5h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_35 > ../Analysis/logs/eval/step4/2h5h6k_1e3_256_35.txt
##
##echo "Training model 2h5h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_40 > ../Analysis/logs/train/step4/2h5h6k_1e3_256_40.txt
##
##echo "Evaluating model 2h5h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e3_256_40 > ../Analysis/logs/eval/step4/2h5h6k_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model 2h5h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_20 > ../Analysis/logs/train/step4/2h5h6k_5e4_256_20.txt
##
##echo "Evaluating model 2h5h6k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_20 > ../Analysis/logs/eval/step4/2h5h6k_5e4_256_20.txt
##
##echo "Training model 2h5h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_25 > ../Analysis/logs/train/step4/2h5h6k_5e4_256_25.txt
##
##echo "Evaluating model 2h5h6k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_25 > ../Analysis/logs/eval/step4/2h5h6k_5e4_256_25.txt
##
##echo "Training model 2h5h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_30 > ../Analysis/logs/train/step4/2h5h6k_5e4_256_30.txt
##
##echo "Evaluating model 2h5h6k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_30 > ../Analysis/logs/eval/step4/2h5h6k_5e4_256_30.txt
##
##echo "Training model 2h5h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_35 > ../Analysis/logs/train/step4/2h5h6k_5e4_256_35.txt
##
##echo "Evaluating model 2h5h6k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_35 > ../Analysis/logs/eval/step4/2h5h6k_5e4_256_35.txt
##
##echo "Training model 2h5h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_40 > ../Analysis/logs/train/step4/2h5h6k_5e4_256_40.txt
##
##echo "Evaluating model 2h5h6k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e4_256_40 > ../Analysis/logs/eval/step4/2h5h6k_5e4_256_40.txt
##
### Learning rate 1e-4
##
##echo "Training model 2h5h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_20 > ../Analysis/logs/train/step4/2h5h6k_1e4_256_20.txt
##
##echo "Evaluating model 2h5h6k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_20 > ../Analysis/logs/eval/step4/2h5h6k_1e4_256_20.txt
##
##echo "Training model 2h5h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_25 > ../Analysis/logs/train/step4/2h5h6k_1e4_256_25.txt
##
##echo "Evaluating model 2h5h6k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_25 > ../Analysis/logs/eval/step4/2h5h6k_1e4_256_25.txt
##
##echo "Training model 2h5h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_30 > ../Analysis/logs/train/step4/2h5h6k_1e4_256_30.txt
##
##echo "Evaluating model 2h5h6k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_30 > ../Analysis/logs/eval/step4/2h5h6k_1e4_256_30.txt
##
##echo "Training model 2h5h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_35 > ../Analysis/logs/train/step4/2h5h6k_1e4_256_35.txt
##
##echo "Evaluating model 2h5h6k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_35 > ../Analysis/logs/eval/step4/2h5h6k_1e4_256_35.txt
##
##echo "Training model 2h5h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_40 > ../Analysis/logs/train/step4/2h5h6k_1e4_256_40.txt
##
##echo "Evaluating model 2h5h6k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_40 > ../Analysis/logs/eval/step4/2h5h6k_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model 2h5h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_20 > ../Analysis/logs/train/step4/2h5h6k_5e5_256_20.txt
##
##echo "Evaluating model 2h5h6k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_20 > ../Analysis/logs/eval/step4/2h5h6k_5e5_256_20.txt
##
#echo "Training model 2h5h6k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_25 > ../Analysis/logs/train/step4/2h5h6k_5e5_256_25.txt
#
#echo "Evaluating model 2h5h6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_25 > ../Analysis/logs/eval/step4/2h5h6k_5e5_256_25.txt
##
##echo "Training model 2h5h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_30 > ../Analysis/logs/train/step4/2h5h6k_5e5_256_30.txt
##
##echo "Evaluating model 2h5h6k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_30 > ../Analysis/logs/eval/step4/2h5h6k_5e5_256_30.txt
##
##echo "Training model 2h5h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_35 > ../Analysis/logs/train/step4/2h5h6k_5e5_256_35.txt
##
##echo "Evaluating model 2h5h6k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_35 > ../Analysis/logs/eval/step4/2h5h6k_5e5_256_35.txt
##
##echo "Training model 2h5h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_40 > ../Analysis/logs/train/step4/2h5h6k_5e5_256_40.txt
##
##echo "Evaluating model 2h5h6k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_40 > ../Analysis/logs/eval/step4/2h5h6k_5e5_256_40.txt
##
### 2h5h3k
### Learning rate 1e-3
##echo "Training model 2h5h3k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_20 > ../Analysis/logs/train/step4/2h5h3k_1e3_256_20.txt
##
##echo "Evaluating model 2h5h3k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_20 > ../Analysis/logs/eval/step4/2h5h3k_1e3_256_20.txt
##
##echo "Training model 2h5h3k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_25 > ../Analysis/logs/train/step4/2h5h3k_1e3_256_25.txt
##
##echo "Evaluating model 2h5h3k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_25 > ../Analysis/logs/eval/step4/2h5h3k_1e3_256_25.txt
##
##echo "Training model 2h5h3k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_30 > ../Analysis/logs/train/step4/2h5h3k_1e3_256_30.txt
##
##echo "Evaluating model 2h5h3k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_30 > ../Analysis/logs/eval/step4/2h5h3k_1e3_256_30.txt
##
##echo "Training model 2h5h3k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_35 > ../Analysis/logs/train/step4/2h5h3k_1e3_256_35.txt
##
##echo "Evaluating model 2h5h3k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_35 > ../Analysis/logs/eval/step4/2h5h3k_1e3_256_35.txt
##
##echo "Training model 2h5h3k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_40 > ../Analysis/logs/train/step4/2h5h3k_1e3_256_40.txt
##
##echo "Evaluating model 2h5h3k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e3_256_40 > ../Analysis/logs/eval/step4/2h5h3k_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model 2h5h3k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_20 > ../Analysis/logs/train/step4/2h5h3k_5e4_256_20.txt
##
##echo "Evaluating model 2h5h3k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_20 > ../Analysis/logs/eval/step4/2h5h3k_5e4_256_20.txt
##
##echo "Training model 2h5h3k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_25 > ../Analysis/logs/train/step4/2h5h3k_5e4_256_25.txt
##
##echo "Evaluating model 2h5h3k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_25 > ../Analysis/logs/eval/step4/2h5h3k_5e4_256_25.txt
##
##echo "Training model 2h5h3k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_30 > ../Analysis/logs/train/step4/2h5h3k_5e4_256_30.txt
##
##echo "Evaluating model 2h5h3k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_30 > ../Analysis/logs/eval/step4/2h5h3k_5e4_256_30.txt
##
##echo "Training model 2h5h3k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_35 > ../Analysis/logs/train/step4/2h5h3k_5e4_256_35.txt
##
##echo "Evaluating model 2h5h3k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_35 > ../Analysis/logs/eval/step4/2h5h3k_5e4_256_35.txt
##
##echo "Training model 2h5h3k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_40 > ../Analysis/logs/train/step4/2h5h3k_5e4_256_40.txt
##
##echo "Evaluating model 2h5h3k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e4_256_40 > ../Analysis/logs/eval/step4/2h5h3k_5e4_256_40.txt
##
### Learning rate 1e-4
##
##echo "Training model 2h5h3k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_20 > ../Analysis/logs/train/step4/2h5h3k_1e4_256_20.txt
##
##echo "Evaluating model 2h5h3k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_20 > ../Analysis/logs/eval/step4/2h5h3k_1e4_256_20.txt
##
##echo "Training model 2h5h3k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_25 > ../Analysis/logs/train/step4/2h5h3k_1e4_256_25.txt
##
##echo "Evaluating model 2h5h3k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_25 > ../Analysis/logs/eval/step4/2h5h3k_1e4_256_25.txt
##
##echo "Training model 2h5h3k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_30 > ../Analysis/logs/train/step4/2h5h3k_1e4_256_30.txt
##
##echo "Evaluating model 2h5h3k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_30 > ../Analysis/logs/eval/step4/2h5h3k_1e4_256_30.txt
##
##echo "Training model 2h5h3k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_35 > ../Analysis/logs/train/step4/2h5h3k_1e4_256_35.txt
##
##echo "Evaluating model 2h5h3k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_35 > ../Analysis/logs/eval/step4/2h5h3k_1e4_256_35.txt
##
##echo "Training model 2h5h3k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_40 > ../Analysis/logs/train/step4/2h5h3k_1e4_256_40.txt
##
##echo "Evaluating model 2h5h3k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_40 > ../Analysis/logs/eval/step4/2h5h3k_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model 2h5h3k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_20 > ../Analysis/logs/train/step4/2h5h3k_5e5_256_20.txt
##
##echo "Evaluating model 2h5h3k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_20 > ../Analysis/logs/eval/step4/2h5h3k_5e5_256_20.txt
##
##echo "Training model 2h5h3k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_25 > ../Analysis/logs/train/step4/2h5h3k_5e5_256_25.txt
##
##echo "Evaluating model 2h5h3k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_25 > ../Analysis/logs/eval/step4/2h5h3k_5e5_256_25.txt
##
##echo "Training model 2h5h3k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_30 > ../Analysis/logs/train/step4/2h5h3k_5e5_256_30.txt
##
##echo "Evaluating model 2h5h3k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_30 > ../Analysis/logs/eval/step4/2h5h3k_5e5_256_30.txt
##
##echo "Training model 2h5h3k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_35 > ../Analysis/logs/train/step4/2h5h3k_5e5_256_35.txt
##
##echo "Evaluating model 2h5h3k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_35 > ../Analysis/logs/eval/step4/2h5h3k_5e5_256_35.txt
##
##echo "Training model 2h5h3k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_40 > ../Analysis/logs/train/step4/2h5h3k_5e5_256_40.txt
##
##echo "Evaluating model 2h5h3k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h3k --model_name 2h5h3k_5e5_256_40 > ../Analysis/logs/eval/step4/2h5h3k_5e5_256_40.txt
##
### 2h5h4k
### Learning rate 1e-3
##
##echo "Training model 2h5h4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_20 > ../Analysis/logs/train/step4/2h5h4k_1e3_256_20.txt
##
##echo "Evaluating model 2h5h4k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_20 > ../Analysis/logs/eval/step4/2h5h4k_1e3_256_20.txt
##
##echo "Training model 2h5h4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_25 > ../Analysis/logs/train/step4/2h5h4k_1e3_256_25.txt
##
##echo "Evaluating model 2h5h4k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_25 > ../Analysis/logs/eval/step4/2h5h4k_1e3_256_25.txt
##
##echo "Training model 2h5h4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_30 > ../Analysis/logs/train/step4/2h5h4k_1e3_256_30.txt
##
##echo "Evaluating model 2h5h4k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_30 > ../Analysis/logs/eval/step4/2h5h4k_1e3_256_30.txt
##
##echo "Training model 2h5h4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_35 > ../Analysis/logs/train/step4/2h5h4k_1e3_256_35.txt
##
##echo "Evaluating model 2h5h4k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_35 > ../Analysis/logs/eval/step4/2h5h4k_1e3_256_35.txt
##
##echo "Training model 2h5h4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_40 > ../Analysis/logs/train/step4/2h5h4k_1e3_256_40.txt
##
##echo "Evaluating model 2h5h4k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e3_256_40 > ../Analysis/logs/eval/step4/2h5h4k_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model 2h5h4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_20 > ../Analysis/logs/train/step4/2h5h4k_5e4_256_20.txt
##
##echo "Evaluating model 2h5h4k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_20 > ../Analysis/logs/eval/step4/2h5h4k_5e4_256_20.txt
##
##echo "Training model 2h5h4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_25 > ../Analysis/logs/train/step4/2h5h4k_5e4_256_25.txt
##
##echo "Evaluating model 2h5h4k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_25 > ../Analysis/logs/eval/step4/2h5h4k_5e4_256_25.txt
##
##echo "Training model 2h5h4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_30 > ../Analysis/logs/train/step4/2h5h4k_5e4_256_30.txt
##
##echo "Evaluating model 2h5h4k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_30 > ../Analysis/logs/eval/step4/2h5h4k_5e4_256_30.txt
##
##echo "Training model 2h5h4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_35 > ../Analysis/logs/train/step4/2h5h4k_5e4_256_35.txt
##
##echo "Evaluating model 2h5h4k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_35 > ../Analysis/logs/eval/step4/2h5h4k_5e4_256_35.txt
##
##echo "Training model 2h5h4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_40 > ../Analysis/logs/train/step4/2h5h4k_5e4_256_40.txt
##
##echo "Evaluating model 2h5h4k_5e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e4_256_40 > ../Analysis/logs/eval/step4/2h5h4k_5e4_256_40.txt
##
### Learning rate 1e-4
##
##echo "Training model 2h5h4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_20 > ../Analysis/logs/train/step4/2h5h4k_1e4_256_20.txt
##
##echo "Evaluating model 2h5h4k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_20 > ../Analysis/logs/eval/step4/2h5h4k_1e4_256_20.txt
##
##echo "Training model 2h5h4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_25 > ../Analysis/logs/train/step4/2h5h4k_1e4_256_25.txt
##
##echo "Evaluating model 2h5h4k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_25 > ../Analysis/logs/eval/step4/2h5h4k_1e4_256_25.txt
##
##echo "Training model 2h5h4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_30 > ../Analysis/logs/train/step4/2h5h4k_1e4_256_30.txt
##
##echo "Evaluating model 2h5h4k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_30 > ../Analysis/logs/eval/step4/2h5h4k_1e4_256_30.txt
##
##echo "Training model 2h5h4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_35 > ../Analysis/logs/train/step4/2h5h4k_1e4_256_35.txt
##
##echo "Evaluating model 2h5h4k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_35 > ../Analysis/logs/eval/step4/2h5h4k_1e4_256_35.txt
##
##echo "Training model 2h5h4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_40 > ../Analysis/logs/train/step4/2h5h4k_1e4_256_40.txt
##
##echo "Evaluating model 2h5h4k_1e4_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_40 > ../Analysis/logs/eval/step4/2h5h4k_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model 2h5h4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_20 > ../Analysis/logs/train/step4/2h5h4k_5e5_256_20.txt
##
##echo "Evaluating model 2h5h4k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_20 > ../Analysis/logs/eval/step4/2h5h4k_5e5_256_20.txt
##
##echo "Training model 2h5h4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_25 > ../Analysis/logs/train/step4/2h5h4k_5e5_256_25.txt
##
##echo "Evaluating model 2h5h4k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_25 > ../Analysis/logs/eval/step4/2h5h4k_5e5_256_25.txt
##
##echo "Training model 2h5h4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_30 > ../Analysis/logs/train/step4/2h5h4k_5e5_256_30.txt
##
##echo "Evaluating model 2h5h4k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_30 > ../Analysis/logs/eval/step4/2h5h4k_5e5_256_30.txt
##
#echo "Training model 2h5h4k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_35 > ../Analysis/logs/train/step4/2h5h4k_5e5_256_35.txt
#
#echo "Evaluating model 2h5h4k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_35 > ../Analysis/logs/eval/step4/2h5h4k_5e5_256_35.txt
##
##echo "Training model 2h5h4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_40 > ../Analysis/logs/train/step4/2h5h4k_5e5_256_40.txt
##
##echo "Evaluating model 2h5h4k_5e5_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_40 > ../Analysis/logs/eval/step4/2h5h4k_5e5_256_40.txt
##
### 2h1h6k
### Learning rate 1e-3
##
##echo "Training model 2h1h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_20 > ../Analysis/logs/train/step4/2h1h6k_1e3_256_20.txt
##
##echo "Evaluating model 2h1h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_20 > ../Analysis/logs/eval/step4/2h1h6k_1e3_256_20.txt
##
##echo "Training model 2h1h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_25 > ../Analysis/logs/train/step4/2h1h6k_1e3_256_25.txt
##
##echo "Evaluating model 2h1h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_25 > ../Analysis/logs/eval/step4/2h1h6k_1e3_256_25.txt
##
##echo "Training model 2h1h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_30 > ../Analysis/logs/train/step4/2h1h6k_1e3_256_30.txt
##
##echo "Evaluating model 2h1h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_30 > ../Analysis/logs/eval/step4/2h1h6k_1e3_256_30.txt
##
##echo "Training model 2h1h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_35 > ../Analysis/logs/train/step4/2h1h6k_1e3_256_35.txt
##
##echo "Evaluating model 2h1h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_35 > ../Analysis/logs/eval/step4/2h1h6k_1e3_256_35.txt
##
##echo "Training model 2h1h6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_40 > ../Analysis/logs/train/step4/2h1h6k_1e3_256_40.txt
##
##echo "Evaluating model 2h1h6k_1e3_256"
##python ../eval_curves.py --test_path $tst \
##              --model_folder step4 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e3_256_40 > ../Analysis/logs/eval/step4/2h1h6k_1e3_256_40.txt
##
### Learning rate 5e-4
##
##echo "Training model 2h1h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_20 > ../Analysis/logs/train/step4/2h1h6k_5e4_256_20.txt
##
#
#### AQUI TENGO QUE PONERLE PA QUE EMPIECE DE NUEVO
#
#echo "Evaluating model 2h1h6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_20 > ../Analysis/logs/eval/step4/2h1h6k_5e4_256_20.txt
##
##echo "Training model 2h1h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_25 > ../Analysis/logs/train/step4/2h1h6k_5e4_256_25.txt
##
#echo "Evaluating model 2h1h6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_25 > ../Analysis/logs/eval/step4/2h1h6k_5e4_256_25.txt
##
##echo "Training model 2h1h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_30 > ../Analysis/logs/train/step4/2h1h6k_5e4_256_30.txt
##
#echo "Evaluating model 2h1h6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_30 > ../Analysis/logs/eval/step4/2h1h6k_5e4_256_30.txt
##
##echo "Training model 2h1h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_35 > ../Analysis/logs/train/step4/2h1h6k_5e4_256_35.txt
##
#echo "Evaluating model 2h1h6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_35 > ../Analysis/logs/eval/step4/2h1h6k_5e4_256_35.txt
##
##echo "Training model 2h1h6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_40 > ../Analysis/logs/train/step4/2h1h6k_5e4_256_40.txt
##
#echo "Evaluating model 2h1h6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e4_256_40 > ../Analysis/logs/eval/step4/2h1h6k_5e4_256_40.txt
##
### Learning rate 1e-4
##
##echo "Training model 2h1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_20 > ../Analysis/logs/train/step4/2h1h6k_1e4_256_20.txt
##
#echo "Evaluating model 2h1h6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_20 > ../Analysis/logs/eval/step4/2h1h6k_1e4_256_20.txt
##
##echo "Training model 2h1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_25 > ../Analysis/logs/train/step4/2h1h6k_1e4_256_25.txt
##
#echo "Evaluating model 2h1h6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_25 > ../Analysis/logs/eval/step4/2h1h6k_1e4_256_25.txt
##
##echo "Training model 2h1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_30 > ../Analysis/logs/train/step4/2h1h6k_1e4_256_30.txt
##
#echo "Evaluating model 2h1h6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_30 > ../Analysis/logs/eval/step4/2h1h6k_1e4_256_30.txt
##
##echo "Training model 2h1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_35 > ../Analysis/logs/train/step4/2h1h6k_1e4_256_35.txt
##
#echo "Evaluating model 2h1h6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_35 > ../Analysis/logs/eval/step4/2h1h6k_1e4_256_35.txt
##
##echo "Training model 2h1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_40 > ../Analysis/logs/train/step4/2h1h6k_1e4_256_40.txt
##
#echo "Evaluating model 2h1h6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_40 > ../Analysis/logs/eval/step4/2h1h6k_1e4_256_40.txt
##
### Learning rate 5e-5
##
##echo "Training model 2h1h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_20 > ../Analysis/logs/train/step4/2h1h6k_5e5_256_20.txt
##
#echo "Evaluating model 2h1h6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_20 > ../Analysis/logs/eval/step4/2h1h6k_5e5_256_20.txt
##
##echo "Training model 2h1h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_25 > ../Analysis/logs/train/step4/2h1h6k_5e5_256_25.txt
##
#echo "Evaluating model 2h1h6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_25 > ../Analysis/logs/eval/step4/2h1h6k_5e5_256_25.txt
##
##echo "Training model 2h1h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_30 > ../Analysis/logs/train/step4/2h1h6k_5e5_256_30.txt
##
#echo "Evaluating model 2h1h6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_30 > ../Analysis/logs/eval/step4/2h1h6k_5e5_256_30.txt
##
##echo "Training model 2h1h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_35 > ../Analysis/logs/train/step4/2h1h6k_5e5_256_35.txt
##
#echo "Evaluating model 2h1h6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_35 > ../Analysis/logs/eval/step4/2h1h6k_5e5_256_35.txt
##
##echo "Training model 2h1h6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_40 > ../Analysis/logs/train/step4/2h1h6k_5e5_256_40.txt
##
#echo "Evaluating model 2h1h6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h6k --model_name 2h1h6k_5e5_256_40 > ../Analysis/logs/eval/step4/2h1h6k_5e5_256_40.txt
##
## 2h1h5k
## Learning rate 1e-3
##
##echo "Training model 2h1h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_20 > ../Analysis/logs/train/step4/2h1h5k_1e3_256_20.txt
#
#echo "Evaluating model 2h1h5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_20 > ../Analysis/logs/eval/step4/2h1h5k_1e3_256_20.txt
#
##echo "Training model 2h1h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_25 > ../Analysis/logs/train/step4/2h1h5k_1e3_256_25.txt
#
#echo "Evaluating model 2h1h5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_25 > ../Analysis/logs/eval/step4/2h1h5k_1e3_256_25.txt
#
##echo "Training model 2h1h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_30 > ../Analysis/logs/train/step4/2h1h5k_1e3_256_30.txt
#
#echo "Evaluating model 2h1h5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_30 > ../Analysis/logs/eval/step4/2h1h5k_1e3_256_30.txt
#
##echo "Training model 2h1h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_35 > ../Analysis/logs/train/step4/2h1h5k_1e3_256_35.txt
#
#echo "Evaluating model 2h1h5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_35 > ../Analysis/logs/eval/step4/2h1h5k_1e3_256_35.txt
#
##echo "Training model 2h1h5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_40 > ../Analysis/logs/train/step4/2h1h5k_1e3_256_40.txt
#
#echo "Evaluating model 2h1h5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e3_256_40 > ../Analysis/logs/eval/step4/2h1h5k_1e3_256_40.txt
#
## Learning rate 5e-4
#
##echo "Training model 2h1h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_20 > ../Analysis/logs/train/step4/2h1h5k_5e4_256_20.txt
#
#echo "Evaluating model 2h1h5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_20 > ../Analysis/logs/eval/step4/2h1h5k_5e4_256_20.txt
#
##echo "Training model 2h1h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_25 > ../Analysis/logs/train/step4/2h1h5k_5e4_256_25.txt
#
#echo "Evaluating model 2h1h5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_25 > ../Analysis/logs/eval/step4/2h1h5k_5e4_256_25.txt
#
##echo "Training model 2h1h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_30 > ../Analysis/logs/train/step4/2h1h5k_5e4_256_30.txt
#
#echo "Evaluating model 2h1h5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_30 > ../Analysis/logs/eval/step4/2h1h5k_5e4_256_30.txt
#
##echo "Training model 2h1h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_35 > ../Analysis/logs/train/step4/2h1h5k_5e4_256_35.txt
#
#echo "Evaluating model 2h1h5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_35 > ../Analysis/logs/eval/step4/2h1h5k_5e4_256_35.txt
#
##echo "Training model 2h1h5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_40 > ../Analysis/logs/train/step4/2h1h5k_5e4_256_40.txt
#
#echo "Evaluating model 2h1h5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e4_256_40 > ../Analysis/logs/eval/step4/2h1h5k_5e4_256_40.txt
#
## Learning rate 1e-4
#
##echo "Training model 2h1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_20 > ../Analysis/logs/train/step4/2h1h5k_1e4_256_20.txt
#
#echo "Evaluating model 2h1h5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_20 > ../Analysis/logs/eval/step4/2h1h5k_1e4_256_20.txt
#
##echo "Training model 2h1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_25 > ../Analysis/logs/train/step4/2h1h5k_1e4_256_25.txt
#
#echo "Evaluating model 2h1h5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_25 > ../Analysis/logs/eval/step4/2h1h5k_1e4_256_25.txt
#
##echo "Training model 2h1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_30 > ../Analysis/logs/train/step4/2h1h5k_1e4_256_30.txt
#
#echo "Evaluating model 2h1h5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_30 > ../Analysis/logs/eval/step4/2h1h5k_1e4_256_30.txt
#
##echo "Training model 2h1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_35 > ../Analysis/logs/train/step4/2h1h5k_1e4_256_35.txt
#
#echo "Evaluating model 2h1h5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_35 > ../Analysis/logs/eval/step4/2h1h5k_1e4_256_35.txt
#
##echo "Training model 2h1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_40 > ../Analysis/logs/train/step4/2h1h5k_1e4_256_40.txt
#
#echo "Evaluating model 2h1h5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_40 > ../Analysis/logs/eval/step4/2h1h5k_1e4_256_40.txt
#
## Learning rate 5e-5
#
##echo "Training model 2h1h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_20 > ../Analysis/logs/train/step4/2h1h5k_5e5_256_20.txt
#
#echo "Evaluating model 2h1h5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_20 > ../Analysis/logs/eval/step4/2h1h5k_5e5_256_20.txt
#
##echo "Training model 2h1h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_25 > ../Analysis/logs/train/step4/2h1h5k_5e5_256_25.txt
#
#echo "Evaluating model 2h1h5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_25 > ../Analysis/logs/eval/step4/2h1h5k_5e5_256_25.txt
#
##echo "Training model 2h1h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_30 > ../Analysis/logs/train/step4/2h1h5k_5e5_256_30.txt
#
#echo "Evaluating model 2h1h5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_30 > ../Analysis/logs/eval/step4/2h1h5k_5e5_256_30.txt
#
##echo "Training model 2h1h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_35 > ../Analysis/logs/train/step4/2h1h5k_5e5_256_35.txt
#
#echo "Evaluating model 2h1h5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_35 > ../Analysis/logs/eval/step4/2h1h5k_5e5_256_35.txt
#
##echo "Training model 2h1h5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_40 > ../Analysis/logs/train/step4/2h1h5k_5e5_256_40.txt
#
#echo "Evaluating model 2h1h5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1h5k --model_name 2h1h5k_5e5_256_40 > ../Analysis/logs/eval/step4/2h1h5k_5e5_256_40.txt
###
## 2h2k5k
## Learning rate 1e-3
#
##echo "Training model 2h2k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_20 > ../Analysis/logs/train/step4/2h2k5k_1e3_256_20.txt
#
#echo "Evaluating model 2h2k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_20 > ../Analysis/logs/eval/step4/2h2k5k_1e3_256_20.txt
#
##echo "Training model 2h2k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_25 > ../Analysis/logs/train/step4/2h2k5k_1e3_256_25.txt
#
#echo "Evaluating model 2h2k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_25 > ../Analysis/logs/eval/step4/2h2k5k_1e3_256_25.txt
#
##echo "Training model 2h2k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_30 > ../Analysis/logs/train/step4/2h2k5k_1e3_256_30.txt
#
#echo "Evaluating model 2h2k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_30 > ../Analysis/logs/eval/step4/2h2k5k_1e3_256_30.txt
#
##echo "Training model 2h2k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_35 > ../Analysis/logs/train/step4/2h2k5k_1e3_256_35.txt
#
#echo "Evaluating model 2h2k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_35 > ../Analysis/logs/eval/step4/2h2k5k_1e3_256_35.txt
#
##echo "Training model 2h2k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_40 > ../Analysis/logs/train/step4/2h2k5k_1e3_256_40.txt
#
#echo "Evaluating model 2h2k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e3_256_40 > ../Analysis/logs/eval/step4/2h2k5k_1e3_256_40.txt
#
## Learning rate 5e-4
#
##echo "Training model 2h2k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_20 > ../Analysis/logs/train/step4/2h2k5k_5e4_256_20.txt
#
#echo "Evaluating model 2h2k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_20 > ../Analysis/logs/eval/step4/2h2k5k_5e4_256_20.txt
#
##echo "Training model 2h2k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_25 > ../Analysis/logs/train/step4/2h2k5k_5e4_256_25.txt
#
#echo "Evaluating model 2h2k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_25 > ../Analysis/logs/eval/step4/2h2k5k_5e4_256_25.txt
#
##echo "Training model 2h2k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_30 > ../Analysis/logs/train/step4/2h2k5k_5e4_256_30.txt
#
#echo "Evaluating model 2h2k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_30 > ../Analysis/logs/eval/step4/2h2k5k_5e4_256_30.txt
#
##echo "Training model 2h2k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_35 > ../Analysis/logs/train/step4/2h2k5k_5e4_256_35.txt
#
#echo "Evaluating model 2h2k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_35 > ../Analysis/logs/eval/step4/2h2k5k_5e4_256_35.txt
#
##echo "Training model 2h2k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_40 > ../Analysis/logs/train/step4/2h2k5k_5e4_256_40.txt
#
#echo "Evaluating model 2h2k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e4_256_40 > ../Analysis/logs/eval/step4/2h2k5k_5e4_256_40.txt
#
## Learning rate 1e-4
#
##echo "Training model 2h2k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_20 > ../Analysis/logs/train/step4/2h2k5k_1e4_256_20.txt
#
#echo "Evaluating model 2h2k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_20 > ../Analysis/logs/eval/step4/2h2k5k_1e4_256_20.txt
#
##echo "Training model 2h2k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_25 > ../Analysis/logs/train/step4/2h2k5k_1e4_256_25.txt
#
#echo "Evaluating model 2h2k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_25 > ../Analysis/logs/eval/step4/2h2k5k_1e4_256_25.txt
#
##echo "Training model 2h2k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_30 > ../Analysis/logs/train/step4/2h2k5k_1e4_256_30.txt
#
#echo "Evaluating model 2h2k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_30 > ../Analysis/logs/eval/step4/2h2k5k_1e4_256_30.txt
#
##echo "Training model 2h2k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_35 > ../Analysis/logs/train/step4/2h2k5k_1e4_256_35.txt
#
#echo "Evaluating model 2h2k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_35 > ../Analysis/logs/eval/step4/2h2k5k_1e4_256_35.txt
#
##echo "Training model 2h2k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_40 > ../Analysis/logs/train/step4/2h2k5k_1e4_256_40.txt
#
#echo "Evaluating model 2h2k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_40 > ../Analysis/logs/eval/step4/2h2k5k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model 2h2k5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_20 > ../Analysis/logs/train/step4/2h2k5k_5e5_256_20.txt
#
#echo "Evaluating model 2h2k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_20 > ../Analysis/logs/eval/step4/2h2k5k_5e5_256_20.txt
#
#echo "Training model 2h2k5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_25 > ../Analysis/logs/train/step4/2h2k5k_5e5_256_25.txt
#
#echo "Evaluating model 2h2k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_25 > ../Analysis/logs/eval/step4/2h2k5k_5e5_256_25.txt
#
##echo "Training model 2h2k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_30 > ../Analysis/logs/train/step4/2h2k5k_5e5_256_30.txt
#
#echo "Evaluating model 2h2k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_30 > ../Analysis/logs/eval/step4/2h2k5k_5e5_256_30.txt
#
##echo "Training model 2h2k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_35 > ../Analysis/logs/train/step4/2h2k5k_5e5_256_35.txt
#
#echo "Evaluating model 2h2k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_35 > ../Analysis/logs/eval/step4/2h2k5k_5e5_256_35.txt
#
##echo "Training model 2h2k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_40 > ../Analysis/logs/train/step4/2h2k5k_5e5_256_40.txt
#
#echo "Evaluating model 2h2k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h2k5k --model_name 2h2k5k_5e5_256_40 > ../Analysis/logs/eval/step4/2h2k5k_5e5_256_40.txt
##
### 2h1k6k
## Learning rate 1e-3
#
##echo "Training model 2h1k6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_20 > ../Analysis/logs/train/step4/2h1k6k_1e3_256_20.txt
##
#echo "Evaluating model 2h1k6k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_20 > ../Analysis/logs/eval/step4/2h1k6k_1e3_256_20.txt
##
##echo "Training model 2h1k6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_25 > ../Analysis/logs/train/step4/2h1k6k_1e3_256_25.txt
##
#echo "Evaluating model 2h1k6k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_25 > ../Analysis/logs/eval/step4/2h1k6k_1e3_256_25.txt
##
##echo "Training model 2h1k6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_30 > ../Analysis/logs/train/step4/2h1k6k_1e3_256_30.txt
##
#echo "Evaluating model 2h1k6k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_30 > ../Analysis/logs/eval/step4/2h1k6k_1e3_256_30.txt
##
##echo "Training model 2h1k6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_35 > ../Analysis/logs/train/step4/2h1k6k_1e3_256_35.txt
##
#echo "Evaluating model 2h1k6k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_35 > ../Analysis/logs/eval/step4/2h1k6k_1e3_256_35.txt
##
##echo "Training model 2h1k6k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_40 > ../Analysis/logs/train/step4/2h1k6k_1e3_256_40.txt
##
#echo "Evaluating model 2h1k6k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e3_256_40 > ../Analysis/logs/eval/step4/2h1k6k_1e3_256_40.txt
##
## Learning rate 5e-4
#
##echo "Training model 2h1k6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_20 > ../Analysis/logs/train/step4/2h1k6k_5e4_256_20.txt
##
#echo "Evaluating model 2h1k6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_20 > ../Analysis/logs/eval/step4/2h1k6k_5e4_256_20.txt
##
##echo "Training model 2h1k6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_25 > ../Analysis/logs/train/step4/2h1k6k_5e4_256_25.txt
##
#echo "Evaluating model 2h1k6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_25 > ../Analysis/logs/eval/step4/2h1k6k_5e4_256_25.txt
##
##echo "Training model 2h1k6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_30 > ../Analysis/logs/train/step4/2h1k6k_5e4_256_30.txt
##
#echo "Evaluating model 2h1k6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_30 > ../Analysis/logs/eval/step4/2h1k6k_5e4_256_30.txt
##
##echo "Training model 2h1k6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_35 > ../Analysis/logs/train/step4/2h1k6k_5e4_256_35.txt
##
#echo "Evaluating model 2h1k6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_35 > ../Analysis/logs/eval/step4/2h1k6k_5e4_256_35.txt
##
##echo "Training model 2h1k6k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_40 > ../Analysis/logs/train/step4/2h1k6k_5e4_256_40.txt
##
#echo "Evaluating model 2h1k6k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e4_256_40 > ../Analysis/logs/eval/step4/2h1k6k_5e4_256_40.txt
#
## Learning rate 1e-4
#
#echo "Training model 2h1k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_20 > ../Analysis/logs/train/step4/2h1k6k_1e4_256_20.txt
##
#echo "Evaluating model 2h1k6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_20 > ../Analysis/logs/eval/step4/2h1k6k_1e4_256_20.txt
##
##echo "Training model 2h1k6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_25 > ../Analysis/logs/train/step4/2h1k6k_1e4_256_25.txt
##
#echo "Evaluating model 2h1k6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_25 > ../Analysis/logs/eval/step4/2h1k6k_1e4_256_25.txt
##
##echo "Training model 2h1k6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_30 > ../Analysis/logs/train/step4/2h1k6k_1e4_256_30.txt
##
#echo "Evaluating model 2h1k6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_30 > ../Analysis/logs/eval/step4/2h1k6k_1e4_256_30.txt
##
##echo "Training model 2h1k6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_35 > ../Analysis/logs/train/step4/2h1k6k_1e4_256_35.txt
##
#echo "Evaluating model 2h1k6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_35 > ../Analysis/logs/eval/step4/2h1k6k_1e4_256_35.txt
##
##echo "Training model 2h1k6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_40 > ../Analysis/logs/train/step4/2h1k6k_1e4_256_40.txt
##
#echo "Evaluating model 2h1k6k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_40 > ../Analysis/logs/eval/step4/2h1k6k_1e4_256_40.txt
#
## Learning rate 5e-5
#
#echo "Training model 2h1k6k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_20 > ../Analysis/logs/train/step4/2h1k6k_5e5_256_20.txt
##
#echo "Evaluating model 2h1k6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_20 > ../Analysis/logs/eval/step4/2h1k6k_5e5_256_20.txt
##
##echo "Training model 2h1k6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_25 > ../Analysis/logs/train/step4/2h1k6k_5e5_256_25.txt
##
#echo "Evaluating model 2h1k6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_25 > ../Analysis/logs/eval/step4/2h1k6k_5e5_256_25.txt
##
#echo "Training model 2h1k6k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_30 > ../Analysis/logs/train/step4/2h1k6k_5e5_256_30.txt
##
#echo "Evaluating model 2h1k6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_30 > ../Analysis/logs/eval/step4/2h1k6k_5e5_256_30.txt
##
##echo "Training model 2h1k6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_35 > ../Analysis/logs/train/step4/2h1k6k_5e5_256_35.txt
##
#echo "Evaluating model 2h1k6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_35 > ../Analysis/logs/eval/step4/2h1k6k_5e5_256_35.txt
##
##echo "Training model 2h1k6k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_40 > ../Analysis/logs/train/step4/2h1k6k_5e5_256_40.txt
##
#echo "Evaluating model 2h1k6k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k6k --model_name 2h1k6k_5e5_256_40 > ../Analysis/logs/eval/step4/2h1k6k_5e5_256_40.txt
##
## 2h1k4k
## Learning rate 1e-3
##
##echo "Training model 2h1k4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_20 > ../Analysis/logs/train/step4/2h1k4k_1e3_256_20.txt
#
#echo "Evaluating model 2h1k4k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_20 > ../Analysis/logs/eval/step4/2h1k4k_1e3_256_20.txt
#
##echo "Training model 2h1k4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_25 > ../Analysis/logs/train/step4/2h1k4k_1e3_256_25.txt
#
#echo "Evaluating model 2h1k4k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_25 > ../Analysis/logs/eval/step4/2h1k4k_1e3_256_25.txt
#
##echo "Training model 2h1k4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_30 > ../Analysis/logs/train/step4/2h1k4k_1e3_256_30.txt
#
#echo "Evaluating model 2h1k4k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_30 > ../Analysis/logs/eval/step4/2h1k4k_1e3_256_30.txt
#
##echo "Training model 2h1k4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_35 > ../Analysis/logs/train/step4/2h1k4k_1e3_256_35.txt
#
#echo "Evaluating model 2h1k4k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_35 > ../Analysis/logs/eval/step4/2h1k4k_1e3_256_35.txt
#
##echo "Training model 2h1k4k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_40 > ../Analysis/logs/train/step4/2h1k4k_1e3_256_40.txt
#
#echo "Evaluating model 2h1k4k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e3_256_40 > ../Analysis/logs/eval/step4/2h1k4k_1e3_256_40.txt
#
## Learning rate 5e-4
#
##echo "Training model 2h1k4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_20 > ../Analysis/logs/train/step4/2h1k4k_5e4_256_20.txt
#
#echo "Evaluating model 2h1k4k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_20 > ../Analysis/logs/eval/step4/2h1k4k_5e4_256_20.txt
#
##echo "Training model 2h1k4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_25 > ../Analysis/logs/train/step4/2h1k4k_5e4_256_25.txt
#
#echo "Evaluating model 2h1k4k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_25 > ../Analysis/logs/eval/step4/2h1k4k_5e4_256_25.txt
#
##echo "Training model 2h1k4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_30 > ../Analysis/logs/train/step4/2h1k4k_5e4_256_30.txt
#
#echo "Evaluating model 2h1k4k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_30 > ../Analysis/logs/eval/step4/2h1k4k_5e4_256_30.txt
#
##echo "Training model 2h1k4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_35 > ../Analysis/logs/train/step4/2h1k4k_5e4_256_35.txt
#
#echo "Evaluating model 2h1k4k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_35 > ../Analysis/logs/eval/step4/2h1k4k_5e4_256_35.txt
#
##echo "Training model 2h1k4k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_40 > ../Analysis/logs/train/step4/2h1k4k_5e4_256_40.txt
#
#echo "Evaluating model 2h1k4k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e4_256_40 > ../Analysis/logs/eval/step4/2h1k4k_5e4_256_40.txt
#
## Learning rate 1e-4
##
#echo "Training model 2h1k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_20 > ../Analysis/logs/train/step4/2h1k4k_1e4_256_20.txt
#
#echo "Evaluating model 2h1k4k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_20 > ../Analysis/logs/eval/step4/2h1k4k_1e4_256_20.txt
#
##echo "Training model 2h1k4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_25 > ../Analysis/logs/train/step4/2h1k4k_1e4_256_25.txt
#
#echo "Evaluating model 2h1k4k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_25 > ../Analysis/logs/eval/step4/2h1k4k_1e4_256_25.txt
#
##echo "Training model 2h1k4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_30 > ../Analysis/logs/train/step4/2h1k4k_1e4_256_30.txt
#
#echo "Evaluating model 2h1k4k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_30 > ../Analysis/logs/eval/step4/2h1k4k_1e4_256_30.txt
#
##echo "Training model 2h1k4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_35 > ../Analysis/logs/train/step4/2h1k4k_1e4_256_35.txt
#
#echo "Evaluating model 2h1k4k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_35 > ../Analysis/logs/eval/step4/2h1k4k_1e4_256_35.txt
#
##echo "Training model 2h1k4k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_40 > ../Analysis/logs/train/step4/2h1k4k_1e4_256_40.txt
#
#echo "Evaluating model 2h1k4k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_40 > ../Analysis/logs/eval/step4/2h1k4k_1e4_256_40.txt
#
## Learning rate 5e-5
##
##echo "Training model 2h1k4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_20 > ../Analysis/logs/train/step4/2h1k4k_5e5_256_20.txt
#
#echo "Evaluating model 2h1k4k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_20 > ../Analysis/logs/eval/step4/2h1k4k_5e5_256_20.txt
#
#echo "Training model 2h1k4k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_25 > ../Analysis/logs/train/step4/2h1k4k_5e5_256_25.txt
#
#echo "Evaluating model 2h1k4k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_25 > ../Analysis/logs/eval/step4/2h1k4k_5e5_256_25.txt
#
##echo "Training model 2h1k4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_30 > ../Analysis/logs/train/step4/2h1k4k_5e5_256_30.txt
#
#echo "Evaluating model 2h1k4k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_30 > ../Analysis/logs/eval/step4/2h1k4k_5e5_256_30.txt
#
##echo "Training model 2h1k4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_35 > ../Analysis/logs/train/step4/2h1k4k_5e5_256_35.txt
#
#echo "Evaluating model 2h1k4k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_35 > ../Analysis/logs/eval/step4/2h1k4k_5e5_256_35.txt
#
##echo "Training model 2h1k4k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_40 > ../Analysis/logs/train/step4/2h1k4k_5e5_256_40.txt
#
#echo "Evaluating model 2h1k4k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k4k --model_name 2h1k4k_5e5_256_40 > ../Analysis/logs/eval/step4/2h1k4k_5e5_256_40.txt
###
### 2h1k5k
### Learning rate 1e-3
##
##echo "Training model 2h1k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_20 > ../Analysis/logs/train/step4/2h1k5k_1e3_256_20.txt
##
#echo "Evaluating model 2h1k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_20 > ../Analysis/logs/eval/step4/2h1k5k_1e3_256_20.txt
##
##echo "Training model 2h1k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_25 > ../Analysis/logs/train/step4/2h1k5k_1e3_256_25.txt
##
#echo "Evaluating model 2h1k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_25 > ../Analysis/logs/eval/step4/2h1k5k_1e3_256_25.txt
##
##echo "Training model 2h1k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_30 > ../Analysis/logs/train/step4/2h1k5k_1e3_256_30.txt
##
#echo "Evaluating model 2h1k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_30 > ../Analysis/logs/eval/step4/2h1k5k_1e3_256_30.txt
##
##echo "Training model 2h1k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_35 > ../Analysis/logs/train/step4/2h1k5k_1e3_256_35.txt
##
#echo "Evaluating model 2h1k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_35 > ../Analysis/logs/eval/step4/2h1k5k_1e3_256_35.txt
##
##echo "Training model 2h1k5k, lr = 1e-3, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-3 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_40 > ../Analysis/logs/train/step4/2h1k5k_1e3_256_40.txt
##
#echo "Evaluating model 2h1k5k_1e3_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_40 > ../Analysis/logs/eval/step4/2h1k5k_1e3_256_40.txt
##
## Learning rate 5e-4
#
##echo "Training model 2h1k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_20 > ../Analysis/logs/train/step4/2h1k5k_5e4_256_20.txt
##
#echo "Evaluating model 2h1k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_20 > ../Analysis/logs/eval/step4/2h1k5k_5e4_256_20.txt
##
##echo "Training model 2h1k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_25 > ../Analysis/logs/train/step4/2h1k5k_5e4_256_25.txt
##
#echo "Evaluating model 2h1k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_25 > ../Analysis/logs/eval/step4/2h1k5k_5e4_256_25.txt
##
##echo "Training model 2h1k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_30 > ../Analysis/logs/train/step4/2h1k5k_5e4_256_30.txt
##
#echo "Evaluating model 2h1k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_30 > ../Analysis/logs/eval/step4/2h1k5k_5e4_256_30.txt
##
##echo "Training model 2h1k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_35 > ../Analysis/logs/train/step4/2h1k5k_5e4_256_35.txt
##
#echo "Evaluating model 2h1k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_35 > ../Analysis/logs/eval/step4/2h1k5k_5e4_256_35.txt
##
##echo "Training model 2h1k5k, lr = 5e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_40 > ../Analysis/logs/train/step4/2h1k5k_5e4_256_40.txt
##
#echo "Evaluating model 2h1k5k_5e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e4_256_40 > ../Analysis/logs/eval/step4/2h1k5k_5e4_256_40.txt
##
### Learning rate 1e-4
##
##echo "Training model 2h1k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 20 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_20 > ../Analysis/logs/train/step4/2h1k5k_1e4_256_20.txt
##
#echo "Evaluating model 2h1k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_20 > ../Analysis/logs/eval/step4/2h1k5k_1e4_256_20.txt
##
##echo "Training model 2h1k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_25 > ../Analysis/logs/train/step4/2h1k5k_1e4_256_25.txt
##
#echo "Evaluating model 2h1k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_25 > ../Analysis/logs/eval/step4/2h1k5k_1e4_256_25.txt
##
##echo "Training model 2h1k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_30 > ../Analysis/logs/train/step4/2h1k5k_1e4_256_30.txt
##
#echo "Evaluating model 2h1k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_30 > ../Analysis/logs/eval/step4/2h1k5k_1e4_256_30.txt
##
##echo "Training model 2h1k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_35 > ../Analysis/logs/train/step4/2h1k5k_1e4_256_35.txt
##
#echo "Evaluating model 2h1k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_35 > ../Analysis/logs/eval/step4/2h1k5k_1e4_256_35.txt
##
##echo "Training model 2h1k5k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 1e-4 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_40 > ../Analysis/logs/train/step4/2h1k5k_1e4_256_40.txt
##
#echo "Evaluating model 2h1k5k_1e4_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_40 > ../Analysis/logs/eval/step4/2h1k5k_1e4_256_40.txt
##
### Learning rate 5e-5
##
#echo "Training model 2h1k5k, lr = 5e-5, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 5e-5 --batch_size 256 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_20 > ../Analysis/logs/train/step4/2h1k5k_5e5_256_20.txt
##
#echo "Evaluating model 2h1k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_20 > ../Analysis/logs/eval/step4/2h1k5k_5e5_256_20.txt
##
##echo "Training model 2h1k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 25 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_25 > ../Analysis/logs/train/step4/2h1k5k_5e5_256_25.txt
##
#echo "Evaluating model 2h1k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_25 > ../Analysis/logs/eval/step4/2h1k5k_5e5_256_25.txt
##
##echo "Training model 2h1k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 30 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_30 > ../Analysis/logs/train/step4/2h1k5k_5e5_256_30.txt
##
#echo "Evaluating model 2h1k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_30 > ../Analysis/logs/eval/step4/2h1k5k_5e5_256_30.txt
##
##echo "Training model 2h1k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 35 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_35 > ../Analysis/logs/train/step4/2h1k5k_5e5_256_35.txt
##
#echo "Evaluating model 2h1k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_35 > ../Analysis/logs/eval/step4/2h1k5k_5e5_256_35.txt
##
##echo "Training model 2h1k5k, lr = 5e-5, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step4 --patience 40 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 5 --lr 5e-5 --batch_size 256 \
##              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_40 > ../Analysis/logs/train/step4/2h1k5k_5e5_256_40.txt
##
#echo "Evaluating model 2h1k5k_5e5_256"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_40 > ../Analysis/logs/eval/step4/2h1k5k_5e5_256_40.txt

# reports 2 excel

#echo "Creating summary of reports excel file"
#python ../trainevalcurves2excel.py --xls_name 'ANN_step4' --archives_folder 'step4'

#echo "Evaluating model 2h5h4k_5e5_256_35"
#python ../eval.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_35

#echo Graficando desde CSV
python ../metrics.py --csv_folder ../Analysis/OutputsCSV/step4/eval