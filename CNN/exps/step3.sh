#!/bin/bash

mkdir -p ../logs/train/step3
mkdir -p ../logs/eval/step3
mkdir -p ../models/step3

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

## 1c1h_1e6_256_30
#
#echo "Training model 1c1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1h --model_name 1c1h_1e6_256_30 > ../logs/train/step3/1c1h_1e6_256_30.txt
#
#echo "Evaluating model 1c1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1h --model_name 1c1h_1e6_256_30 > ../logs/eval/step3/1c1h_1e6_256_30.txt
#
## 1c2h_1e6_256_30
#
#echo "Training model 1c2h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2h --model_name 1c2h_1e6_256_30 > ../logs/train/step3/1c2h_1e6_256_30.txt
#
#echo "Evaluating model 1c2h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2h --model_name 1c2h_1e6_256_30 > ../logs/eval/step3/1c2h_1e6_256_30.txt
#
## 1c5h_1e6_256_30
#
#echo "Training model 1c5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5h --model_name 1c5h_1e6_256_30 > ../logs/train/step3/1c5h_1e6_256_30.txt
#
#echo "Evaluating model 1c5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5h --model_name 1c5h_1e6_256_30 > ../logs/eval/step3/1c5h_1e6_256_30.txt
#
## 1c1k_1e6_256_30
#
#echo "Training model 1c1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1k --model_name 1c1k_1e6_256_30 > ../logs/train/step3/1c1k_1e6_256_30.txt
#
#echo "Evaluating model 1c1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1k --model_name 1c1k_1e6_256_30 > ../logs/eval/step3/1c1k_1e6_256_30.txt
#
#
## 1c2k_1e6_256_30
#
#echo "Training model 1c2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2k --model_name 1c2k_1e6_256_30 > ../logs/train/step3/1c2k_1e6_256_30.txt
#
#echo "Evaluating model 1c2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2k --model_name 1c2k_1e6_256_30 > ../logs/eval/step3/1c2k_1e6_256_30.txt
#
## 1c3k_1e6_256_30
#
#echo "Training model 1c3k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c3k --model_name 1c3k_1e6_256_30 > ../logs/train/step3/1c3k_1e6_256_30.txt
#
#echo "Evaluating model 1c3k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c3k --model_name 1c3k_1e6_256_30 > ../logs/eval/step3/1c3k_1e6_256_30.txt
#
## 1c4k_1e6_256_30
#
#echo "Training model 1c4k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c4k --model_name 1c4k_1e6_256_30 > ../logs/train/step3/1c4k_1e6_256_30.txt
#
#echo "Evaluating model 1c4k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c4k --model_name 1c4k_1e6_256_30 > ../logs/eval/step3/1c4k_1e6_256_30.txt
#
## 1c5k_1e6_256_30
#
#echo "Training model 1c5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5k --model_name 1c5k_1e6_256_30 > ../logs/train/step3/1c5k_1e6_256_30.txt
#
#echo "Evaluating model 1c5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5k --model_name 1c5k_1e6_256_30 > ../logs/eval/step3/1c5k_1e6_256_30.txt
#
## 1c6k_1e6_256_30
#
#echo "Training model 1c6k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c6k --model_name 1c6k_1e6_256_30 > ../logs/train/step3/1c6k_1e6_256_30.txt
#
#echo "Evaluating model 1c6k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c6k --model_name 1c6k_1e6_256_30 > ../logs/eval/step3/1c6k_1e6_256_30.txt
#
## 1c10k_1e6_256_30
#
#echo "Training model 1c10k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c10k --model_name 1c10k_1e6_256_30 > ../logs/train/step3/1c10k_1e6_256_30.txt
#
#echo "Evaluating model 1c10k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c10k --model_name 1c10k_1e6_256_30 > ../logs/eval/step3/1c10k_1e6_256_30.txt
#
## 1c10k10k_1e6_256_30
#
#echo "Training model 1c10k10k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c10k10k --model_name 1c10k10k_1e6_256_30 > ../logs/train/step3/1c10k10k_1e6_256_30.txt
#
#echo "Evaluating model 1c10k10k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c10k10k --model_name 1c10k10k_1e6_256_30 > ../logs/eval/step3/1c10k10k_1e6_256_30.txt
#
## 1c10k5k_1e6_256_30
#
#echo "Training model 1c10k5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c10k5k --model_name 1c10k5k_1e6_256_30 > ../logs/train/step3/1c10k5k_1e6_256_30.txt
#
#echo "Evaluating model 1c10k5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c10k5k --model_name 1c10k5k_1e6_256_30 > ../logs/eval/step3/1c10k5k_1e6_256_30.txt
#
## 1c10k1k_1e6_256_30
#
#echo "Training model 1c10k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c10k1k --model_name 1c10k1k_1e6_256_30 > ../logs/train/step3/1c10k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c10k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c10k1k --model_name 1c10k1k_1e6_256_30 > ../logs/eval/step3/1c10k1k_1e6_256_30.txt
#
## 1c10k1h_1e6_256_30
#
#echo "Training model 1c10k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c10k1h --model_name 1c10k1h_1e6_256_30 > ../logs/train/step3/1c10k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c10k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c10k1h --model_name 1c10k1h_1e6_256_30 > ../logs/eval/step3/1c10k1h_1e6_256_30.txt
#
## 1c10k10_1e6_256_30
#
#echo "Training model 1c10k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c10k10 --model_name 1c10k10_1e6_256_30 > ../logs/train/step3/1c10k10_1e6_256_30.txt
#
#echo "Evaluating model 1c10k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c10k10 --model_name 1c10k10_1e6_256_30 > ../logs/eval/step3/1c10k10_1e6_256_30.txt
#
## 1c6k6k_1e6_256_30
#
#echo "Training model 1c6k6k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c6k6k --model_name 1c6k6k_1e6_256_30 > ../logs/train/step3/1c6k6k_1e6_256_30.txt
#
#echo "Evaluating model 1c6k6k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c6k6k --model_name 1c6k6k_1e6_256_30 > ../logs/eval/step3/1c6k6k_1e6_256_30.txt
#
#
## 1c6k1k_1e6_256_30
#
#echo "Training model 1c6k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c6k1k --model_name 1c6k1k_1e6_256_30 > ../logs/train/step3/1c6k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c6k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c6k1k --model_name 1c6k1k_1e6_256_30 > ../logs/eval/step3/1c6k1k_1e6_256_30.txt
#
## 1c6k1h_1e6_256_30
#
#echo "Training model 1c6k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c6k1h --model_name 1c6k1h_1e6_256_30 > ../logs/train/step3/1c6k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c6k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c6k1h --model_name 1c6k1h_1e6_256_30 > ../logs/eval/step3/1c6k1h_1e6_256_30.txt
#
## 1c6k10_1e6_256_30
#
#echo "Training model 1c6k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c6k10 --model_name 1c6k10_1e6_256_30 > ../logs/train/step3/1c6k10_1e6_256_30.txt
#
#echo "Evaluating model 1c6k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c6k10 --model_name 1c6k10_1e6_256_30 > ../logs/eval/step3/1c6k10_1e6_256_30.txt
#
#
## 1c5k5k_1e6_256_30
#
#echo "Training model 1c5k5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5k5k --model_name 1c5k5k_1e6_256_30 > ../logs/train/step3/1c5k5k_1e6_256_30.txt
#
#echo "Evaluating model 1c5k5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5k5k --model_name 1c5k5k_1e6_256_30 > ../logs/eval/step3/1c5k5k_1e6_256_30.txt
#
## 1c5k1k_1e6_256_30
#
#echo "Training model 1c5k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5k1k --model_name 1c5k1k_1e6_256_30 > ../logs/train/step3/1c5k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c5k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5k1k --model_name 1c5k1k_1e6_256_30 > ../logs/eval/step3/1c5k1k_1e6_256_30.txt
#
## 1c5k1h_1e6_256_30
#
#echo "Training model 1c5k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5k1h --model_name 1c5k1h_1e6_256_30 > ../logs/train/step3/1c5k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c5k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5k1h --model_name 1c5k1h_1e6_256_30 > ../logs/eval/step3/1c5k1h_1e6_256_30.txt
#
## 1c5k10_1e6_256_30
#
#echo "Training model 1c5k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5k10 --model_name 1c5k10_1e6_256_30 > ../logs/train/step3/1c5k10_1e6_256_30.txt
#
#echo "Evaluating model 1c5k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5k10 --model_name 1c5k10_1e6_256_30 > ../logs/eval/step3/1c5k10_1e6_256_30.txt
#
## 1c4k4k_1e6_256_30
#
#echo "Training model 1c4k4k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c4k4k --model_name 1c4k4k_1e6_256_30 > ../logs/train/step3/1c4k4k_1e6_256_30.txt
#
#echo "Evaluating model 1c4k4k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --classifier 1c4k4k --model_name 1c4k4k_1e6_256_30 > ../logs/eval/step3/1c4k4k_1e6_256_30.txt
#
## 1c4k1k_1e6_256_30
#
#echo "Training model 1c4k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c4k1k --model_name 1c4k1k_1e6_256_30 > ../logs/train/step3/1c4k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c4k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c4k1k --model_name 1c4k1k_1e6_256_30 > ../logs/eval/step3/1c4k1k_1e6_256_30.txt
#
## 1c4k1h_1e6_256_30
#
#echo "Training model 1c4k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c4k1h --model_name 1c4k1h_1e6_256_30 > ../logs/train/step3/1c4k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c4k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c4k1h --model_name 1c4k1h_1e6_256_30 > ../logs/eval/step3/1c4k1h_1e6_256_30.txt
#
## 1c4k10_1e6_256_30
#
#echo "Training model 1c4k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c4k10 --model_name 1c4k10_1e6_256_30 > ../logs/train/step3/1c4k10_1e6_256_30.txt
#
#echo "Evaluating model 1c4k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c4k1k0 --model_name 1c4k10_1e6_256_30 > ../logs/eval/step3/1c4k10_1e6_256_30.txt
#
## 1c3k3k_1e6_256_30
#
#echo "Training model 1c3k3k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c3k3k --model_name 1c3k3k_1e6_256_30 > ../logs/train/step3/1c3k3k_1e6_256_30.txt
#
#echo "Evaluating model 1c3k3k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c3k3k --model_name 1c3k3k_1e6_256_30 > ../logs/eval/step3/1c3k3k_1e6_256_30.txt
#
## 1c3k1k_1e6_256_30
#
#echo "Training model 1c3k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c3k1k --model_name 1c3k1k_1e6_256_30 > ../logs/train/step3/1c3k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c3k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c3k1k --model_name 1c3k1k_1e6_256_30 > ../logs/eval/step3/1c3k1k_1e6_256_30.txt
#
## 1c3k1h_1e6_256_30
#
#echo "Training model 1c3k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c3k1h --model_name 1c3k1h_1e6_256_30 > ../logs/train/step3/1c3k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c3k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c3k1h --model_name 1c3k1h_1e6_256_30 > ../logs/eval/step3/1c3k1h_1e6_256_30.txt
#
## 1c3k10_1e6_256_30
#
#echo "Training model 1c3k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c3k10 --model_name 1c3k10_1e6_256_30 > ../logs/train/step3/1c3k10_1e6_256_30.txt
#
#echo "Evaluating model 1c3k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c3k10 --model_name 1c3k10_1e6_256_30 > ../logs/eval/step3/1c3k10_1e6_256_30.txt
#
## 1c2k2k_1e6_256_30
#
#echo "Training model 1c2k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2k2k --model_name 1c2k2k_1e6_256_30 > ../logs/train/step3/1c2k2k_1e6_256_30.txt
#
#echo "Evaluating model 1c2k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2k2k --model_name 1c2k2k_1e6_256_30 > ../logs/eval/step3/1c2k2k_1e6_256_30.txt
#
## 1c2k1k_1e6_256_30
#
#echo "Training model 1c2k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2k1k --model_name 1c2k1k_1e6_256_30 > ../logs/train/step3/1c2k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c2k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2k1k --model_name 1c2k1k_1e6_256_30 > ../logs/eval/step3/1c2k1k_1e6_256_30.txt
#
## 1c2k1h_1e6_256_30
#
#echo "Training model 1c2k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2k1h --model_name 1c2k1h_1e6_256_30 > ../logs/train/step3/1c2k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c2k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2k1h --model_name 1c2k1h_1e6_256_30 > ../logs/eval/step3/1c2k1h_1e6_256_30.txt
#
## 1c2k10_1e6_256_30
#
#echo "Training model 1c2k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2k10 --model_name 1c2k10_1e6_256_30 > ../logs/train/step3/1c2k10_1e6_256_30.txt
#
#echo "Evaluating model 1c2k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2k10 --model_name 1c2k10_1e6_256_30 > ../logs/eval/step3/1c2k10_1e6_256_30.txt
#
## 1c1k1k_1e6_256_30
#
#echo "Training model 1c1k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1k1k --model_name 1c1k1k_1e6_256_30 > ../logs/train/step3/1c1k1k_1e6_256_30.txt
#
#echo "Evaluating model 1c1k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1k1k --model_name 1c1k1k_1e6_256_30 > ../logs/eval/step3/1c1k1k_1e6_256_30.txt
#
## 1c1k1h_1e6_256_30
#
#echo "Training model 1c1k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1k1h --model_name 1c1k1h_1e6_256_30 > ../logs/train/step3/1c1k1h_1e6_256_30.txt
#
#echo "Evaluating model 1c1k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1k1h --model_name 1c1k1h_1e6_256_30 > ../logs/eval/step3/1c1k1h_1e6_256_30.txt
#
## 1c1k10_1e6_256_30
#
#echo "Training model 1c1k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1k10 --model_name 1c1k10_1e6_256_30 > ../logs/train/step3/1c1k10_1e6_256_30.txt
#
#echo "Evaluating model 1c1k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1k10 --model_name 1c1k10_1e6_256_30 > ../logs/eval/step3/1c1k10_1e6_256_30.txt
#
## 1c5h5h_1e6_256_30
#
#echo "Training model 1c5h5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5h5h --model_name 1c5h5h_1e6_256_30 > ../logs/train/step3/1c5h5h_1e6_256_30.txt
#
#echo "Evaluating model 1c5h5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5h5h --model_name 1c5h5h_1e6_256_30 > ../logs/eval/step3/1c5h5h_1e6_256_30.txt
#
## 1c5h1h_1e6_256_30
#
#echo "Training model 1c5h1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5h1h --model_name 1c5h1h_1e6_256_30 > ../logs/train/step3/1c5h1h_1e6_256_30.txt
#
#echo "Evaluating model 1c5h1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5h1h --model_name 1c5h1h_1e6_256_30 > ../logs/eval/step3/1c5h1h_1e6_256_30.txt
#
## 1c5h10_1e6_256_30
#
#echo "Training model 1c5h10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c5h10 --model_name 1c5h10_1e6_256_30 > ../logs/train/step3/1c5h10_1e6_256_30.txt
#
#echo "Evaluating model 1c5h10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c5h10 --model_name 1c5h10_1e6_256_30 > ../logs/eval/step3/1c5h10_1e6_256_30.txt
#
## 1c2h2h_1e6_256_30
#
#echo "Training model 1c2h2h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2h2h --model_name 1c2h2h_1e6_256_30 > ../logs/train/step3/1c2h2h_1e6_256_30.txt
#
#echo "Evaluating model 1c2h2h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2h2h --model_name 1c2h2h_1e6_256_30 > ../logs/eval/step3/1c2h2h_1e6_256_30.txt
#
## 1c2h1h_1e6_256_30
#
#echo "Training model 1c2h1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2h1h --model_name 1c2h1h_1e6_256_30 > ../logs/train/step3/1c2h1h_1e6_256_30.txt
#
#echo "Evaluating model 1c2h1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2h1h --model_name 1c2h1h_1e6_256_30 > ../logs/eval/step3/1c2h1h_1e6_256_30.txt
#
## 1c2h10_1e6_256_30
#
#echo "Training model 1c2h10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c2h10 --model_name 1c2h10_1e6_256_30 > ../logs/train/step3/1c2h10_1e6_256_30.txt
#
#echo "Evaluating model 1c2h10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c2h10 --model_name 1c2h10_1e6_256_30 > ../logs/eval/step3/1c2h10_1e6_256_30.txt
#
## 1c1h1h_1e6_256_30
#
#echo "Training model 1c1h1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1h1h --model_name 1c1h1h_1e6_256_30 > ../logs/train/step3/1c1h1h_1e6_256_30.txt
#
#echo "Evaluating model 1c1h1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1h1h --model_name 1c1h1h_1e6_256_30 > ../logs/eval/step3/1c1h1h_1e6_256_30.txt
#
## 1c1h10_1e6_256_30
#
#echo "Training model 1c1h10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 1c1h10 --model_name 1c1h10_1e6_256_30 > ../logs/train/step3/1c1h10_1e6_256_30.txt
#
#echo "Evaluating model 1c1h10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 1c1h10 --model_name 1c1h10_1e6_256_30 > ../logs/eval/step3/1c1h10_1e6_256_30.txt
#
## 2c20k_1e6_256_30
#
#echo "Training model 2c20k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k --model_name 2c20k_1e6_256_30 > ../logs/train/step3/2c20k_1e6_256_30.txt
#
#echo "Evaluating model 2c20k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k --model_name 2c20k_1e6_256_30 > ../logs/eval/step3/2c20k_1e6_256_30.txt
#
## 2c15k_1e6_256_30
#
#echo "Training model 2c15k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k --model_name 2c15k_1e6_256_30 > ../logs/train/step3/2c15k_1e6_256_30.txt
#
#echo "Evaluating model 2c15k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k --model_name 2c15k_1e6_256_30 > ../logs/eval/step3/2c15k_1e6_256_30.txt
#
## 2c10k_1e6_256_30
#
#echo "Training model 2c10k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k --model_name 2c10k_1e6_256_30 > ../logs/train/step3/2c10k_1e6_256_30.txt
#
#echo "Evaluating model 2c10k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k --model_name 2c10k_1e6_256_30 > ../logs/eval/step3/2c10k_1e6_256_30.txt
#
## 2c5k_1e6_256_30
#
#echo "Training model 2c5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k --model_name 2c5k_1e6_256_30 > ../logs/train/step3/2c5k_1e6_256_30.txt
#
#echo "Evaluating model 2c5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k --model_name 2c5k_1e6_256_30 > ../logs/eval/step3/2c5k_1e6_256_30.txt
#
## 2c3k_1e6_256_30
#
#echo "Training model 2c3k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k --model_name 2c3k_1e6_256_30 > ../logs/train/step3/2c3k_1e6_256_30.txt
#
#echo "Evaluating model 2c3k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k --model_name 2c3k_1e6_256_30 > ../logs/eval/step3/2c3k_1e6_256_30.txt
#
## 2c2k_1e6_256_30
#
#echo "Training model 2c2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c2k --model_name 2c2k_1e6_256_30 > ../logs/train/step3/2c2k_1e6_256_30.txt
#
#echo "Evaluating model 2c2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c2k --model_name 2c2k_1e6_256_30 > ../logs/eval/step3/2c2k_1e6_256_30.txt
#
## 2c1k_1e6_256_30
#
#echo "Training model 2c1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c1k --model_name 2c1k_1e6_256_30 > ../logs/train/step3/2c1k_1e6_256_30.txt
#
#echo "Evaluating model 2c1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c1k --model_name 2c1k_1e6_256_30 > ../logs/eval/step3/2c1k_1e6_256_30.txt
#
## 2c20k20k_1e6_256_30
#
#echo "Training model 2c20k20k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k20k --model_name 2c20k20k_1e6_256_30 > ../logs/train/step3/2c20k20k_1e6_256_30.txt
#
#echo "Evaluating model 2c20k20k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k20k --model_name 2c20k20k_1e6_256_30 > ../logs/eval/step3/2c20k20k_1e6_256_30.txt
#
## 2c20k10k_1e6_256_30
#
#echo "Training model 2c20k10k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k10k --model_name 2c20k10k_1e6_256_30 > ../logs/train/step3/2c20k10k_1e6_256_30.txt
#
#echo "Evaluating model 2c20k10k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k10k --model_name 2c20k10k_1e6_256_30 > ../logs/eval/step3/2c20k10k_1e6_256_30.txt
#
## 2c20k5k_1e6_256_30
#
#echo "Training model 2c20k5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k5k --model_name 2c20k5k_1e6_256_30 > ../logs/train/step3/2c20k5k_1e6_256_30.txt
#
#echo "Evaluating model 2c20k5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k5k --model_name 2c20k5k_1e6_256_30 > ../logs/eval/step3/2c20k5k_1e6_256_30.txt
#
## 2c20k2k_1e6_256_30
#
#echo "Training model 2c20k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k2k --model_name 2c20k2k_1e6_256_30 > ../logs/train/step3/2c20k2k_1e6_256_30.txt
#
#echo "Evaluating model 2c20k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k2k --model_name 2c20k2k_1e6_256_30 > ../logs/eval/step3/2c20k2k_1e6_256_30.txt
#
## 2c20k1k_1e6_256_30
#
#echo "Training model 2c20k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k1k --model_name 2c20k1k_1e6_256_30 > ../logs/train/step3/2c20k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c20k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k1k --model_name 2c20k1k_1e6_256_30 > ../logs/eval/step3/2c20k1k_1e6_256_30.txt
#
## 2c20k5h_1e6_256_30
#
#echo "Training model 2c20k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k5h --model_name 2c20k5h_1e6_256_30 > ../logs/train/step3/2c20k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c20k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k5h --model_name 2c20k5h_1e6_256_30 > ../logs/eval/step3/2c20k5h_1e6_256_30.txt
#
## 2c20k1h_1e6_256_30
#
#echo "Training model 2c20k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k1h --model_name 2c20k1h_1e6_256_30 > ../logs/train/step3/2c20k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c20k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k1h --model_name 2c20k1h_1e6_256_30 > ../logs/eval/step3/2c20k1h_1e6_256_30.txt
#
## 2c20k10_1e6_256_30
#
#echo "Training model 2c20k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c20k10 --model_name 2c20k10_1e6_256_30 > ../logs/train/step3/2c20k10_1e6_256_30.txt
#
#echo "Evaluating model 2c20k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c20k10 --model_name 2c20k10_1e6_256_30 > ../logs/eval/step3/2c20k10_1e6_256_30.txt
#
## 2c15k15k_1e6_256_30
#
#echo "Training model 2c15k15k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k15k --model_name 2c15k15k_1e6_256_30 > ../logs/train/step3/2c15k15k_1e6_256_30.txt
#
#echo "Evaluating model 2c15k15k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k15k --model_name 2c15k15k_1e6_256_30 > ../logs/eval/step3/2c15k15k_1e6_256_30.txt
#
## 2c15k10k_1e6_256_30
#
#echo "Training model 2c15k10k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k10k --model_name 2c15k10k_1e6_256_30 > ../logs/train/step3/2c15k10k_1e6_256_30.txt
#
#echo "Evaluating model 2c15k10k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k10k --model_name 2c15k10k_1e6_256_30 > ../logs/eval/step3/2c15k10k_1e6_256_30.txt
#
## 2c15k5k_1e6_256_30
#
#echo "Training model 2c15k5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k5k --model_name 2c15k5k_1e6_256_30 > ../logs/train/step3/2c15k5k_1e6_256_30.txt
#
#echo "Evaluating model 2c15k5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k5k --model_name 2c15k5k_1e6_256_30 > ../logs/eval/step3/2c15k5k_1e6_256_30.txt
#
## 2c15k2k_1e6_256_30
#
#echo "Training model 2c15k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k2k --model_name 2c15k2k_1e6_256_30 > ../logs/train/step3/2c15k2k_1e6_256_30.txt
#
#echo "Evaluating model 2c15k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k2k --model_name 2c15k2k_1e6_256_30 > ../logs/eval/step3/2c15k2k_1e6_256_30.txt
#
## 2c15k1k_1e6_256_30
#
#echo "Training model 2c15k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k1k --model_name 2c15k1k_1e6_256_30 > ../logs/train/step3/2c15k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c15k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k1k --model_name 2c15k1k_1e6_256_30 > ../logs/eval/step3/2c15k1k_1e6_256_30.txt
#
## 2c15k5h_1e6_256_30
#
#echo "Training model 2c15k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k5h --model_name 2c15k5h_1e6_256_30 > ../logs/train/step3/2c15k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c15k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k5h --model_name 2c15k5h_1e6_256_30 > ../logs/eval/step3/2c15k5h_1e6_256_30.txt
#
## 2c15k1h_1e6_256_30
#
#echo "Training model 2c15k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k1h --model_name 2c15k1h_1e6_256_30 > ../logs/train/step3/2c15k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c15k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k1h --model_name 2c15k1h_1e6_256_30 > ../logs/eval/step3/2c15k1h_1e6_256_30.txt
#
## 2c15k10_1e6_256_30
#
#echo "Training model 2c15k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c15k10 --model_name 2c15k10_1e6_256_30 > ../logs/train/step3/2c15k10_1e6_256_30.txt
#
#echo "Evaluating model 2c15k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c15k10 --model_name 2c15k10_1e6_256_30 > ../logs/eval/step3/2c15k10_1e6_256_30.txt
#
## 2c10k10k_1e6_256_30
#
#echo "Training model 2c10k10k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k10k --model_name 2c10k10k_1e6_256_30 > ../logs/train/step3/2c10k10k_1e6_256_30.txt
#
#echo "Evaluating model 2c10k10k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k10k --model_name 2c10k10k_1e6_256_30 > ../logs/eval/step3/2c10k10k_1e6_256_30.txt
#
## 2c10k5k_1e6_256_30
#
#echo "Training model 2c10k5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k5k --model_name 2c10k5k_1e6_256_30 > ../logs/train/step3/2c10k5k_1e6_256_30.txt
#
#echo "Evaluating model 2c10k5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k5k --model_name 2c10k5k_1e6_256_30 > ../logs/eval/step3/2c10k5k_1e6_256_30.txt
#
## 2c10k2k_1e6_256_30
#
#echo "Training model 2c10k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k2k --model_name 2c10k2k_1e6_256_30 > ../logs/train/step3/2c10k2k_1e6_256_30.txt
#
#echo "Evaluating model 2c10k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k2k --model_name 2c10k2k_1e6_256_30 > ../logs/eval/step3/2c10k2k_1e6_256_30.txt
#
## 2c10k1k_1e6_256_30
#
#echo "Training model 2c10k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k1k --model_name 2c10k1k_1e6_256_30 > ../logs/train/step3/2c10k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c10k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k1k --model_name 2c10k1k_1e6_256_30 > ../logs/eval/step3/2c10k1k_1e6_256_30.txt
#
## 2c10k5h_1e6_256_30
#
#echo "Training model 2c10k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k5h --model_name 2c10k5h_1e6_256_30 > ../logs/train/step3/2c10k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c10k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k5h --model_name 2c10k5h_1e6_256_30 > ../logs/eval/step3/2c10k5h_1e6_256_30.txt
#
## 2c10k1h_1e6_256_30
#
#echo "Training model 2c10k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k1h --model_name 2c10k1h_1e6_256_30 > ../logs/train/step3/2c10k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c10k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k1h --model_name 2c10k1h_1e6_256_30 > ../logs/eval/step3/2c10k1h_1e6_256_30.txt
#
## 2c10k10_1e6_256_30
#
#echo "Training model 2c10k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c10k10 --model_name 2c10k10_1e6_256_30 > ../logs/train/step3/2c10k10_1e6_256_30.txt
#
#echo "Evaluating model 2c10k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c10k10 --model_name 2c10k10_1e6_256_30 > ../logs/eval/step3/2c10k10_1e6_256_30.txt
#
## 2c5k5k_1e6_256_30
#
#echo "Training model 2c5k5k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k5k --model_name 2c5k5k_1e6_256_30 > ../logs/train/step3/2c5k5k_1e6_256_30.txt
#
#echo "Evaluating model 2c5k5k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k5k --model_name 2c5k5k_1e6_256_30 > ../logs/eval/step3/2c5k5k_1e6_256_30.txt
#
## 2c5k2k_1e6_256_30
#
#echo "Training model 2c5k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k2k --model_name 2c5k2k_1e6_256_30 > ../logs/train/step3/2c5k2k_1e6_256_30.txt
#
#echo "Evaluating model 2c5k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k2k --model_name 2c5k2k_1e6_256_30 > ../logs/eval/step3/2c5k2k_1e6_256_30.txt
#
## 2c5k1k_1e6_256_30
#
#echo "Training model 2c5k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k1k --model_name 2c5k1k_1e6_256_30 > ../logs/train/step3/2c5k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c5k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k1k --model_name 2c5k1k_1e6_256_30 > ../logs/eval/step3/2c5k1k_1e6_256_30.txt
#
## 2c5k5h_1e6_256_30
#
#echo "Training model 2c5k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k5h --model_name 2c5k5h_1e6_256_30 > ../logs/train/step3/2c5k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c5k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k5h --model_name 2c5k5h_1e6_256_30 > ../logs/eval/step3/2c5k5h_1e6_256_30.txt
#
## 2c5k1h_1e6_256_30
#
#echo "Training model 2c5k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k1h --model_name 2c5k1h_1e6_256_30 > ../logs/train/step3/2c5k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c5k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k1h --model_name 2c5k1h_1e6_256_30 > ../logs/eval/step3/2c5k1h_1e6_256_30.txt
#
## 2c5k10_1e6_256_30
#
#echo "Training model 2c5k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c5k10 --model_name 2c5k10_1e6_256_30 > ../logs/train/step3/2c5k10_1e6_256_30.txt
#
#echo "Evaluating model 2c5k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c5k10 --model_name 2c5k10_1e6_256_30 > ../logs/eval/step3/2c5k10_1e6_256_30.txt
#
## 2c3k3k_1e6_256_30
#
#echo "Training model 2c3k3k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k3k --model_name 2c3k3k_1e6_256_30 > ../logs/train/step3/2c3k3k_1e6_256_30.txt
#
#echo "Evaluating model 2c3k3k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k3k --model_name 2c3k3k_1e6_256_30 > ../logs/eval/step3/2c3k3k_1e6_256_30.txt
#
## 2c3k2k_1e6_256_30
#
#echo "Training model 2c3k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k2k --model_name 2c3k2k_1e6_256_30 > ../logs/train/step3/2c3k2k_1e6_256_30.txt
#
#echo "Evaluating model 2c3k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k2k --model_name 2c3k2k_1e6_256_30 > ../logs/eval/step3/2c3k2k_1e6_256_30.txt
#
## 2c3k1k_1e6_256_30
#
#echo "Training model 2c3k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k1k --model_name 2c3k1k_1e6_256_30 > ../logs/train/step3/2c3k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c3k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k1k --model_name 2c3k1k_1e6_256_30 > ../logs/eval/step3/2c3k1k_1e6_256_30.txt
#
## 2c3k5h_1e6_256_30
#
#echo "Training model 2c3k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k5h --model_name 2c3k5h_1e6_256_30 > ../logs/train/step3/2c3k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c3k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k5h --model_name 2c3k5h_1e6_256_30 > ../logs/eval/step3/2c3k5h_1e6_256_30.txt
#
## 2c3k1h_1e6_256_30
#
#echo "Training model 2c3k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k1h --model_name 2c3k1h_1e6_256_30 > ../logs/train/step3/2c3k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c3k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k1h --model_name 2c3k1h_1e6_256_30 > ../logs/eval/step3/2c3k1h_1e6_256_30.txt
#
## 2c3k10_1e6_256_30
#
#echo "Training model 2c3k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c3k10 --model_name 2c3k10_1e6_256_30 > ../logs/train/step3/2c3k10_1e6_256_30.txt
#
#echo "Evaluating model 2c3k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c3k10 --model_name 2c3k10_1e6_256_30 > ../logs/eval/step3/2c3k10_1e6_256_30.txt
#
## 2c2k2k_1e6_256_30
#
#echo "Training model 2c2k2k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c2k2k --model_name 2c2k2k_1e6_256_30 > ../logs/train/step3/2c2k2k_1e6_256_30.txt
#
#echo "Evaluating model 2c2k2k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c2k2k --model_name 2c2k2k_1e6_256_30 > ../logs/eval/step3/2c2k2k_1e6_256_30.txt
#
## 2c2k1k_1e6_256_30
#
#echo "Training model 2c2k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c2k1k --model_name 2c2k1k_1e6_256_30 > ../logs/train/step3/2c2k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c2k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c2k1k --model_name 2c2k1k_1e6_256_30 > ../logs/eval/step3/2c2k1k_1e6_256_30.txt
#
## 2c2k5h_1e6_256_30
#
#echo "Training model 2c2k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c2k5h --model_name 2c2k5h_1e6_256_30 > ../logs/train/step3/2c2k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c2k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c2k5h --model_name 2c2k5h_1e6_256_30 > ../logs/eval/step3/2c2k5h_1e6_256_30.txt
#
## 2c2k1h_1e6_256_30
#
#echo "Training model 2c2k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c2k1h --model_name 2c2k1h_1e6_256_30 > ../logs/train/step3/2c2k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c2k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c2k1h --model_name 2c2k1h_1e6_256_30 > ../logs/eval/step3/2c2k1h_1e6_256_30.txt
#
## 2c2k10_1e6_256_30
#
#echo "Training model 2c2k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c2k10 --model_name 2c2k10_1e6_256_30 > ../logs/train/step3/2c2k10_1e6_256_30.txt
#
#echo "Evaluating model 2c2k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c2k10 --model_name 2c2k10_1e6_256_30 > ../logs/eval/step3/2c2k10_1e6_256_30.txt
#
## 2c1k1k_1e6_256_30
#
#echo "Training model 2c1k1k_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c1k1k --model_name 2c1k1k_1e6_256_30 > ../logs/train/step3/2c1k1k_1e6_256_30.txt
#
#echo "Evaluating model 2c1k1k_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c1k1k --model_name 2c1k1k_1e6_256_30 > ../logs/eval/step3/2c1k1k_1e6_256_30.txt
#
## 2c1k5h_1e6_256_30
#
#echo "Training model 2c1k5h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c1k5h --model_name 2c1k5h_1e6_256_30 > ../logs/train/step3/2c1k5h_1e6_256_30.txt
#
#echo "Evaluating model 2c1k5h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c1k5h --model_name 2c1k5h_1e6_256_30 > ../logs/eval/step3/2c1k5h_1e6_256_30.txt
#
## 2c1k1h_1e6_256_30
#
#echo "Training model 2c1k1h_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c1k1h --model_name 2c1k1h_1e6_256_30 > ../logs/train/step3/2c1k1h_1e6_256_30.txt
#
#echo "Evaluating model 2c1k1h_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c1k1h --model_name 2c1k1h_1e6_256_30 > ../logs/eval/step3/2c1k1h_1e6_256_30.txt
#
## 2c1k10_1e6_256_30
#
#echo "Training model 2c1k10_30, lr = 1e-6, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-6 --batch_size 256 --patience 30 \
#              --classifier 2c1k10 --model_name 2c1k10_1e6_256_30 > ../logs/train/step3/2c1k10_1e6_256_30.txt
#
#echo "Evaluating model 2c1k10_1e6_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier 2c1k10 --model_name 2c1k10_1e6_256_30 > ../logs/eval/step3/2c1k10_1e6_256_30.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../traineval2excel.py --xls_name 'CNN_step3' --archives_folder 'step3'