#!/bin/bash

mkdir -p ../Analysis/logs/train/step3
mkdir -p ../Analysis/logs/eval/step3
mkdir -p ../models/step3

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

## Cnn1_6k_1e4_256_30
#
##echo "Training model Cnn1_6k, lr = 1e-4, epochs = 5, batch_size = 256"
##python ../train_validation.py \
##              --model_folder step3 \
##              --train_path $trn --val_path $val      \
##              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
##              --classifier Cnn1_6k --model_name Cnn1_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_1e4_256_30.txt
##
##echo "Evaluating model Cnn1_6k_1e4_256_30"
##python ../eval.py --train_path $trn --test_path $tst \
##              --model_folder step3 \
##              --classifier Cnn1_6k --model_name Cnn1_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_1e4_256_30.txt
#
## Cnn1_5k_1e4_256_30
#
#echo "Training model Cnn1_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k --model_name Cnn1_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k --model_name Cnn1_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_1e4_256_30.txt
#
## Cnn1_4k_1e4_256_30
#
#echo "Training model Cnn1_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k --model_name Cnn1_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k --model_name Cnn1_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_1e4_256_30.txt
#
## Cnn1_3k_1e4_256_30
#
#echo "Training model Cnn1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k --model_name Cnn1_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_1e4_256_30.txt
#
## Cnn1_2k_1e4_256_30
#
#echo "Training model Cnn1_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k --model_name Cnn1_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k --model_name Cnn1_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_1e4_256_30.txt
#
## Cnn1_1k_1e4_256_30
#
#echo "Training model Cnn1_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k --model_name Cnn1_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k --model_name Cnn1_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_1e4_256_30.txt
#
## Cnn1_5h_1e4_256_30
#
#echo "Training model Cnn1_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h --model_name Cnn1_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h --model_name Cnn1_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_1e4_256_30.txt
#
## Cnn1_2h_1e4_256_30
#
#echo "Training model Cnn1_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h --model_name Cnn1_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h --model_name Cnn1_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_1e4_256_30.txt
#
## Cnn1_1h_1e4_256_30
#
#echo "Training model Cnn1_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h --model_name Cnn1_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h --model_name Cnn1_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_1e4_256_30.txt
#
## Cnn1_10_1e4_256_30
#
#echo "Training model Cnn1_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10 --model_name Cnn1_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10 --model_name Cnn1_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_1e4_256_30.txt
#
## Cnn1_6k_6k_1e4_256_30
#
#echo "Training model Cnn1_6k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_6k --model_name Cnn1_6k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_6k --model_name Cnn1_6k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_6k_1e4_256_30.txt
#
## Cnn1_6k_5k_1e4_256_30
#
#echo "Training model Cnn1_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_5k --model_name Cnn1_6k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_5k_1e4_256_30.txt
#
## Cnn1_6k_4k_1e4_256_30
#
#echo "Training model Cnn1_6k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_4k --model_name Cnn1_6k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_4k --model_name Cnn1_6k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_4k_1e4_256_30.txt
#
## Cnn1_6k_3k_1e4_256_30
#
#echo "Training model Cnn1_6k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_3k --model_name Cnn1_6k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_3k --model_name Cnn1_6k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_3k_1e4_256_30.txt
#
## Cnn1_6k_2k_1e4_256_30
#
#echo "Training model Cnn1_6k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_2k --model_name Cnn1_6k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_2k --model_name Cnn1_6k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_2k_1e4_256_30.txt
#
## Cnn1_6k_1k_1e4_256_30
#
#echo "Training model Cnn1_6k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_1k --model_name Cnn1_6k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_1k --model_name Cnn1_6k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_1k_1e4_256_30.txt
#
## Cnn1_6k_5h_1e4_256_30
#
#echo "Training model Cnn1_6k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_5h --model_name Cnn1_6k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_5h --model_name Cnn1_6k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_5h_1e4_256_30.txt
#
## Cnn1_6k_2h_1e4_256_30
#
#echo "Training model Cnn1_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_2h --model_name Cnn1_6k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_2h_1e4_256_30.txt
#
## Cnn1_6k_1h_1e4_256_30
#
#echo "Training model Cnn1_6k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_1h --model_name Cnn1_6k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_1h --model_name Cnn1_6k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_1h_1e4_256_30.txt
#
## Cnn1_6k_10_1e4_256_30
#
#echo "Training model Cnn1_6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_6k_10 --model_name Cnn1_6k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_6k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_6k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_6k_10 --model_name Cnn1_6k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_6k_10_1e4_256_30.txt
#
## Cnn1_5k_6k_1e4_256_30
#
#echo "Training model Cnn1_5k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_6k --model_name Cnn1_5k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_6k --model_name Cnn1_5k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_6k_1e4_256_30.txt
#
## Cnn1_5k_5k_1e4_256_30
#
#echo "Training model Cnn1_5k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_5k --model_name Cnn1_5k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_5k --model_name Cnn1_5k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_5k_1e4_256_30.txt
#
## Cnn1_5k_4k_1e4_256_30
#
#echo "Training model Cnn1_5k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_4k --model_name Cnn1_5k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_4k --model_name Cnn1_5k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_4k_1e4_256_30.txt
#
## Cnn1_5k_3k_1e4_256_30
#
#echo "Training model Cnn1_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_3k --model_name Cnn1_5k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_3k_1e4_256_30.txt
#
## Cnn1_5k_2k_1e4_256_30
#
#echo "Training model Cnn1_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_2k --model_name Cnn1_5k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_2k_1e4_256_30.txt
#
## Cnn1_5k_1k_1e4_256_30
#
#echo "Training model Cnn1_5k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_1k --model_name Cnn1_5k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_1k --model_name Cnn1_5k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_1k_1e4_256_30.txt
#
## Cnn1_5k_5h_1e4_256_30
#
#echo "Training model Cnn1_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_5h --model_name Cnn1_5k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_5h_1e4_256_30.txt
#
## Cnn1_5k_2h_1e4_256_30
#
#echo "Training model Cnn1_5k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_2h --model_name Cnn1_5k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_2h --model_name Cnn1_5k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_2h_1e4_256_30.txt
#
## Cnn1_5k_1h_1e4_256_30
#
#echo "Training model Cnn1_5k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_1h --model_name Cnn1_5k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_1h --model_name Cnn1_5k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_1h_1e4_256_30.txt
#
## Cnn1_5k_10_1e4_256_30
#
#echo "Training model Cnn1_5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5k_10 --model_name Cnn1_5k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5k_10 --model_name Cnn1_5k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5k_10_1e4_256_30.txt
#
## Cnn1_4k_6k_1e4_256_30
#
#echo "Training model Cnn1_4k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_6k --model_name Cnn1_4k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_6k --model_name Cnn1_4k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_6k_1e4_256_30.txt
#
## Cnn1_4k_5k_1e4_256_30
#
#echo "Training model Cnn1_4k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_5k --model_name Cnn1_4k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_5k --model_name Cnn1_4k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_5k_1e4_256_30.txt
#
## Cnn1_4k_4k_1e4_256_30
#
#echo "Training model Cnn1_4k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_4k --model_name Cnn1_4k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_4k --model_name Cnn1_4k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_4k_1e4_256_30.txt
#
## Cnn1_4k_3k_1e4_256_30
#
#echo "Training model Cnn1_4k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_3k --model_name Cnn1_4k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_3k --model_name Cnn1_4k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_3k_1e4_256_30.txt
#
## Cnn1_4k_2k_1e4_256_30
#
#echo "Training model Cnn1_4k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_2k --model_name Cnn1_4k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_2k --model_name Cnn1_4k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_2k_1e4_256_30.txt
#
## Cnn1_4k_1k_1e4_256_30
#
#echo "Training model Cnn1_4k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_1k --model_name Cnn1_4k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_1k --model_name Cnn1_4k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_1k_1e4_256_30.txt
#
## Cnn1_4k_5h_1e4_256_30
#
#echo "Training model Cnn1_4k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_5h --model_name Cnn1_4k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_5h --model_name Cnn1_4k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_5h_1e4_256_30.txt
#
## Cnn1_4k_2h_1e4_256_30
#
#echo "Training model Cnn1_4k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_2h --model_name Cnn1_4k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_2h --model_name Cnn1_4k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_2h_1e4_256_30.txt
#
## Cnn1_4k_1h_1e4_256_30
#
#echo "Training model Cnn1_4k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_1h --model_name Cnn1_4k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_1h --model_name Cnn1_4k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_1h_1e4_256_30.txt
#
## Cnn1_4k_10_1e4_256_30
#
#echo "Training model Cnn1_4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_4k_10 --model_name Cnn1_4k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_4k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_4k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_4k_10 --model_name Cnn1_4k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_4k_10_1e4_256_30.txt
#
## Cnn1_3k_6k_1e4_256_30
#
#echo "Training model Cnn1_3k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_6k --model_name Cnn1_3k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_6k --model_name Cnn1_3k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_6k_1e4_256_30.txt
#
## Cnn1_3k_5k_1e4_256_30
#
#echo "Training model Cnn1_3k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_5k --model_name Cnn1_3k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_5k --model_name Cnn1_3k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_5k_1e4_256_30.txt
#
## Cnn1_3k_4k_1e4_256_30
#
#echo "Training model Cnn1_3k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_4k --model_name Cnn1_3k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_4k --model_name Cnn1_3k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_4k_1e4_256_30.txt
#
## Cnn1_3k_3k_1e4_256_30
#
#echo "Training model Cnn1_3k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_3k --model_name Cnn1_3k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_3k --model_name Cnn1_3k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_3k_1e4_256_30.txt
#
## Cnn1_3k_2k_1e4_256_30
#
#echo "Training model Cnn1_3k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_2k --model_name Cnn1_3k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_2k --model_name Cnn1_3k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_2k_1e4_256_30.txt
#
## Cnn1_3k_1k_1e4_256_30
#
#echo "Training model Cnn1_3k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_1k --model_name Cnn1_3k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_1k --model_name Cnn1_3k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_1k_1e4_256_30.txt
#
## Cnn1_3k_5h_1e4_256_30
#
#echo "Training model Cnn1_3k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_5h --model_name Cnn1_3k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_5h --model_name Cnn1_3k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_5h_1e4_256_30.txt
#
## Cnn1_3k_2h_1e4_256_30
#
#echo "Training model Cnn1_3k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_2h --model_name Cnn1_3k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_2h --model_name Cnn1_3k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_2h_1e4_256_30.txt
#
## Cnn1_3k_1h_1e4_256_30
#
#echo "Training model Cnn1_3k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_1h --model_name Cnn1_3k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_1h --model_name Cnn1_3k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_1h_1e4_256_30.txt
#
## Cnn1_3k_10_1e4_256_30
#
#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_3k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_3k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_3k_10 --model_name Cnn1_3k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_3k_10_1e4_256_30.txt
#
## Cnn1_2k_6k_1e4_256_30
#
#echo "Training model Cnn1_2k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_6k --model_name Cnn1_2k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_6k --model_name Cnn1_2k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_6k_1e4_256_30.txt
#
## Cnn1_2k_5k_1e4_256_30
#
#echo "Training model Cnn1_2k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_5k --model_name Cnn1_2k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_5k --model_name Cnn1_2k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_5k_1e4_256_30.txt
#
## Cnn1_2k_4k_1e4_256_30
#
#echo "Training model Cnn1_2k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_4k --model_name Cnn1_2k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_4k --model_name Cnn1_2k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_4k_1e4_256_30.txt
#
## Cnn1_2k_3k_1e4_256_30
#
#echo "Training model Cnn1_2k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_3k --model_name Cnn1_2k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_3k --model_name Cnn1_2k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_3k_1e4_256_30.txt
#
## Cnn1_2k_2k_1e4_256_30
#
#echo "Training model Cnn1_2k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_2k --model_name Cnn1_2k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_2k --model_name Cnn1_2k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_2k_1e4_256_30.txt
#
## Cnn1_2k_1k_1e4_256_30
#
#echo "Training model Cnn1_2k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_1k --model_name Cnn1_2k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_1k --model_name Cnn1_2k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_1k_1e4_256_30.txt
#
## Cnn1_2k_5h_1e4_256_30
#
#echo "Training model Cnn1_2k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_5h --model_name Cnn1_2k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_5h --model_name Cnn1_2k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_5h_1e4_256_30.txt
#
## Cnn1_2k_2h_1e4_256_30
#
#echo "Training model Cnn1_2k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_2h --model_name Cnn1_2k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_2h --model_name Cnn1_2k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_2h_1e4_256_30.txt
#
## Cnn1_2k_1h_1e4_256_30
#
#echo "Training model Cnn1_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_1h --model_name Cnn1_2k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_1h_1e4_256_30.txt
#
## Cnn1_2k_10_1e4_256_30
#
#echo "Training model Cnn1_2k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2k_10 --model_name Cnn1_2k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2k_10 --model_name Cnn1_2k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2k_10_1e4_256_30.txt
#
## Cnn1_1k_6k_1e4_256_30
#
#echo "Training model Cnn1_1k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_6k --model_name Cnn1_1k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_6k --model_name Cnn1_1k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_6k_1e4_256_30.txt
#
## Cnn1_1k_5k_1e4_256_30
#
#echo "Training model Cnn1_1k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_5k --model_name Cnn1_1k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_5k --model_name Cnn1_1k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_5k_1e4_256_30.txt
#
## Cnn1_1k_4k_1e4_256_30
#
#echo "Training model Cnn1_1k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_4k --model_name Cnn1_1k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_4k --model_name Cnn1_1k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_4k_1e4_256_30.txt
#
## Cnn1_1k_3k_1e4_256_30
#
#echo "Training model Cnn1_1k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_3k --model_name Cnn1_1k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_3k --model_name Cnn1_1k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_3k_1e4_256_30.txt
#
## Cnn1_1k_2k_1e4_256_30
#
#echo "Training model Cnn1_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_2k --model_name Cnn1_1k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_2k_1e4_256_30.txt
#
## Cnn1_1k_1k_1e4_256_30
#
#echo "Training model Cnn1_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_1k --model_name Cnn1_1k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_1k_1e4_256_30.txt
#
## Cnn1_1k_5h_1e4_256_30
#
#echo "Training model Cnn1_1k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_5h --model_name Cnn1_1k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_5h --model_name Cnn1_1k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_5h_1e4_256_30.txt
#
## Cnn1_1k_2h_1e4_256_30
#
#echo "Training model Cnn1_1k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_2h --model_name Cnn1_1k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_2h --model_name Cnn1_1k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_2h_1e4_256_30.txt
#
## Cnn1_1k_1h_1e4_256_30
#
#echo "Training model Cnn1_1k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_1h --model_name Cnn1_1k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_1h --model_name Cnn1_1k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_1h_1e4_256_30.txt
#
## Cnn1_1k_10_1e4_256_30
#
#echo "Training model Cnn1_1k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1k_10 --model_name Cnn1_1k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1k_10 --model_name Cnn1_1k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1k_10_1e4_256_30.txt
#
## Cnn1_5h_6k_1e4_256_30
#
#echo "Training model Cnn1_5h_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_6k --model_name Cnn1_5h_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_6k --model_name Cnn1_5h_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_6k_1e4_256_30.txt
#
## Cnn1_5h_5k_1e4_256_30
#
#echo "Training model Cnn1_5h_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_5k --model_name Cnn1_5h_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_5k --model_name Cnn1_5h_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_5k_1e4_256_30.txt
#
## Cnn1_5h_4k_1e4_256_30
#
#echo "Training model Cnn1_5h_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_4k --model_name Cnn1_5h_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_4k --model_name Cnn1_5h_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_4k_1e4_256_30.txt
#
## Cnn1_5h_3k_1e4_256_30
#
#echo "Training model Cnn1_5h_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_3k --model_name Cnn1_5h_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_3k --model_name Cnn1_5h_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_3k_1e4_256_30.txt
#
## Cnn1_5h_2k_1e4_256_30
#
#echo "Training model Cnn1_5h_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_2k --model_name Cnn1_5h_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_2k --model_name Cnn1_5h_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_2k_1e4_256_30.txt
#
## Cnn1_5h_1k_1e4_256_30
#
#echo "Training model Cnn1_5h_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_1k --model_name Cnn1_5h_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_1k --model_name Cnn1_5h_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_1k_1e4_256_30.txt
#
## Cnn1_5h_5h_1e4_256_30
#
#echo "Training model Cnn1_5h_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_5h --model_name Cnn1_5h_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_5h --model_name Cnn1_5h_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_5h_1e4_256_30.txt
#
## Cnn1_5h_2h_1e4_256_30
#
#echo "Training model Cnn1_5h_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_2h --model_name Cnn1_5h_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_2h --model_name Cnn1_5h_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_2h_1e4_256_30.txt
#
## Cnn1_5h_1h_1e4_256_30
#
#echo "Training model Cnn1_5h_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_1h --model_name Cnn1_5h_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_1h --model_name Cnn1_5h_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_1h_1e4_256_30.txt
#
## Cnn1_5h_10_1e4_256_30
#
#echo "Training model Cnn1_5h_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_5h_10 --model_name Cnn1_5h_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_5h_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_5h_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_5h_10 --model_name Cnn1_5h_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_5h_10_1e4_256_30.txt
#
## Cnn1_2h_6k_1e4_256_30
#
#echo "Training model Cnn1_2h_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_6k --model_name Cnn1_2h_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_6k --model_name Cnn1_2h_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_6k_1e4_256_30.txt
#
## Cnn1_2h_5k_1e4_256_30
#
#echo "Training model Cnn1_2h_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_5k --model_name Cnn1_2h_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_5k --model_name Cnn1_2h_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_5k_1e4_256_30.txt
#
## Cnn1_2h_4k_1e4_256_30
#
#echo "Training model Cnn1_2h_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_4k --model_name Cnn1_2h_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_4k --model_name Cnn1_2h_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_4k_1e4_256_30.txt
#
## Cnn1_2h_3k_1e4_256_30
#
#echo "Training model Cnn1_2h_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_3k --model_name Cnn1_2h_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_3k --model_name Cnn1_2h_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_3k_1e4_256_30.txt
#
## Cnn1_2h_2k_1e4_256_30
#
#echo "Training model Cnn1_2h_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_2k --model_name Cnn1_2h_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_2k --model_name Cnn1_2h_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_2k_1e4_256_30.txt
#
## Cnn1_2h_1k_1e4_256_30
#
#echo "Training model Cnn1_2h_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_1k --model_name Cnn1_2h_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_1k --model_name Cnn1_2h_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_1k_1e4_256_30.txt
#
## Cnn1_2h_5h_1e4_256_30
#
#echo "Training model Cnn1_2h_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_5h --model_name Cnn1_2h_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_5h --model_name Cnn1_2h_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_5h_1e4_256_30.txt
#
## Cnn1_2h_2h_1e4_256_30
#
#echo "Training model Cnn1_2h_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_2h --model_name Cnn1_2h_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_2h --model_name Cnn1_2h_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_2h_1e4_256_30.txt
#
## Cnn1_2h_1h_1e4_256_30
#
#echo "Training model Cnn1_2h_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_1h --model_name Cnn1_2h_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_1h --model_name Cnn1_2h_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_1h_1e4_256_30.txt
#
## Cnn1_2h_10_1e4_256_30
#
#echo "Training model Cnn1_2h_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_2h_10 --model_name Cnn1_2h_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_2h_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_2h_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_2h_10 --model_name Cnn1_2h_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_2h_10_1e4_256_30.txt
#
## Cnn1_1h_6k_1e4_256_30
#
#echo "Training model Cnn1_1h_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_6k --model_name Cnn1_1h_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_6k --model_name Cnn1_1h_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_6k_1e4_256_30.txt
#
## Cnn1_1h_5k_1e4_256_30
#
#echo "Training model Cnn1_1h_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_5k --model_name Cnn1_1h_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_5k --model_name Cnn1_1h_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_5k_1e4_256_30.txt
#
## Cnn1_1h_4k_1e4_256_30
#
#echo "Training model Cnn1_1h_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_4k --model_name Cnn1_1h_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_4k --model_name Cnn1_1h_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_4k_1e4_256_30.txt
#
## Cnn1_1h_3k_1e4_256_30
#
#echo "Training model Cnn1_1h_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_3k --model_name Cnn1_1h_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_3k --model_name Cnn1_1h_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_3k_1e4_256_30.txt
#
## Cnn1_1h_2k_1e4_256_30
#
#echo "Training model Cnn1_1h_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_2k --model_name Cnn1_1h_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_2k --model_name Cnn1_1h_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_2k_1e4_256_30.txt
#
## Cnn1_1h_1k_1e4_256_30
#
#echo "Training model Cnn1_1h_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_1k --model_name Cnn1_1h_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_1k --model_name Cnn1_1h_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_1k_1e4_256_30.txt
#
## Cnn1_1h_5h_1e4_256_30
#
#echo "Training model Cnn1_1h_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_5h --model_name Cnn1_1h_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_5h --model_name Cnn1_1h_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_5h_1e4_256_30.txt
#
## Cnn1_1h_2h_1e4_256_30
#
#echo "Training model Cnn1_1h_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_2h --model_name Cnn1_1h_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_2h --model_name Cnn1_1h_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_2h_1e4_256_30.txt
#
## Cnn1_1h_1h_1e4_256_30
#
#echo "Training model Cnn1_1h_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_1h --model_name Cnn1_1h_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_1h --model_name Cnn1_1h_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_1h_1e4_256_30.txt
#
## Cnn1_1h_10_1e4_256_30
#
#echo "Training model Cnn1_1h_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_1h_10 --model_name Cnn1_1h_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_1h_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_1h_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_1h_10 --model_name Cnn1_1h_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_1h_10_1e4_256_30.txt
#
## Cnn1_10_6k_1e4_256_30
#
#echo "Training model Cnn1_10_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_6k --model_name Cnn1_10_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_6k --model_name Cnn1_10_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_6k_1e4_256_30.txt
#
## Cnn1_10_5k_1e4_256_30
#
#echo "Training model Cnn1_10_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_5k --model_name Cnn1_10_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_5k --model_name Cnn1_10_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_5k_1e4_256_30.txt
#
## Cnn1_10_4k_1e4_256_30
#
#echo "Training model Cnn1_10_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_4k --model_name Cnn1_10_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_4k --model_name Cnn1_10_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_4k_1e4_256_30.txt
#
## Cnn1_10_3k_1e4_256_30
#
#echo "Training model Cnn1_10_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_3k --model_name Cnn1_10_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_3k --model_name Cnn1_10_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_3k_1e4_256_30.txt
#
## Cnn1_10_2k_1e4_256_30
#
#echo "Training model Cnn1_10_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_2k --model_name Cnn1_10_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_2k --model_name Cnn1_10_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_2k_1e4_256_30.txt
#
## Cnn1_10_1k_1e4_256_30
#
#echo "Training model Cnn1_10_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_1k --model_name Cnn1_10_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_1k --model_name Cnn1_10_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_1k_1e4_256_30.txt
#
## Cnn1_10_5h_1e4_256_30
#
#echo "Training model Cnn1_10_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_5h --model_name Cnn1_10_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_5h --model_name Cnn1_10_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_5h_1e4_256_30.txt
#
## Cnn1_10_2h_1e4_256_30
#
#echo "Training model Cnn1_10_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_2h --model_name Cnn1_10_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_2h --model_name Cnn1_10_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_2h_1e4_256_30.txt
#
## Cnn1_10_1h_1e4_256_30
#
#echo "Training model Cnn1_10_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_1h --model_name Cnn1_10_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_1h --model_name Cnn1_10_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_1h_1e4_256_30.txt
#
## Cnn1_10_10_1e4_256_30
#
#echo "Training model Cnn1_10_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn1_10_10 --model_name Cnn1_10_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn1_10_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn1_10_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn1_10_10 --model_name Cnn1_10_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn1_10_10_1e4_256_30.txt
#
## Cnn2_6k_1e4_256_30
#
#echo "Training model Cnn2_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k --model_name Cnn2_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k --model_name Cnn2_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_1e4_256_30.txt
#
## Cnn2_5k_1e4_256_30
#
#echo "Training model Cnn2_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k --model_name Cnn2_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k --model_name Cnn2_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_1e4_256_30.txt
#
## Cnn2_4k_1e4_256_30
#
#echo "Training model Cnn2_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k --model_name Cnn2_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k --model_name Cnn2_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_1e4_256_30.txt
#
## Cnn2_3k_1e4_256_30
#
#echo "Training model Cnn2_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k --model_name Cnn2_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k --model_name Cnn2_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_1e4_256_30.txt
#
## Cnn2_2k_1e4_256_30
#
#echo "Training model Cnn2_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k --model_name Cnn2_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k --model_name Cnn2_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_1e4_256_30.txt
#
## Cnn2_1k_1e4_256_30
#
#echo "Training model Cnn2_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k --model_name Cnn2_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k --model_name Cnn2_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_1e4_256_30.txt
#
## Cnn2_5h_1e4_256_30
#
#echo "Training model Cnn2_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h --model_name Cnn2_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h --model_name Cnn2_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_1e4_256_30.txt
#
## Cnn2_2h_1e4_256_30
#
#echo "Training model Cnn2_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h --model_name Cnn2_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h --model_name Cnn2_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_1e4_256_30.txt
#
## Cnn2_1h_1e4_256_30
#
#echo "Training model Cnn2_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h --model_name Cnn2_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h --model_name Cnn2_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_1e4_256_30.txt
#
## Cnn2_10_1e4_256_30
#
#echo "Training model Cnn2_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10 --model_name Cnn2_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10 --model_name Cnn2_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_1e4_256_30.txt
#
## Cnn2_6k_6k_1e4_256_30
#
#echo "Training model Cnn2_6k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_6k --model_name Cnn2_6k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_6k --model_name Cnn2_6k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_6k_1e4_256_30.txt
#
## Cnn2_6k_5k_1e4_256_30
#
#echo "Training model Cnn2_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_5k --model_name Cnn2_6k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_5k --model_name Cnn2_6k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_5k_1e4_256_30.txt
#
## Cnn2_6k_4k_1e4_256_30
#
#echo "Training model Cnn2_6k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_4k --model_name Cnn2_6k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_4k --model_name Cnn2_6k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_4k_1e4_256_30.txt
#
## Cnn2_6k_3k_1e4_256_30
#
#echo "Training model Cnn2_6k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_3k --model_name Cnn2_6k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_3k --model_name Cnn2_6k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_3k_1e4_256_30.txt
#
## Cnn2_6k_2k_1e4_256_30
#
#echo "Training model Cnn2_6k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_2k --model_name Cnn2_6k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_2k --model_name Cnn2_6k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_2k_1e4_256_30.txt
#
## Cnn2_6k_1k_1e4_256_30
#
#echo "Training model Cnn2_6k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_1k --model_name Cnn2_6k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_1k --model_name Cnn2_6k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_1k_1e4_256_30.txt
#
## Cnn2_6k_5h_1e4_256_30
#
#echo "Training model Cnn2_6k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_5h --model_name Cnn2_6k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_5h --model_name Cnn2_6k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_5h_1e4_256_30.txt
#
## Cnn2_6k_2h_1e4_256_30
#
#echo "Training model Cnn2_6k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_2h --model_name Cnn2_6k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_2h --model_name Cnn2_6k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_2h_1e4_256_30.txt
#
## Cnn2_6k_1h_1e4_256_30
#
#echo "Training model Cnn2_6k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_1h --model_name Cnn2_6k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_1h --model_name Cnn2_6k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_1h_1e4_256_30.txt
#
## Cnn2_6k_10_1e4_256_30
#
#echo "Training model Cnn2_6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_6k_10 --model_name Cnn2_6k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_6k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_6k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_6k_10 --model_name Cnn2_6k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_6k_10_1e4_256_30.txt
#
## Cnn2_5k_6k_1e4_256_30
#
#echo "Training model Cnn2_5k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_6k --model_name Cnn2_5k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_6k --model_name Cnn2_5k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_6k_1e4_256_30.txt
#
## Cnn2_5k_5k_1e4_256_30
#
#echo "Training model Cnn2_5k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_5k --model_name Cnn2_5k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_5k --model_name Cnn2_5k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_5k_1e4_256_30.txt
#
## Cnn2_5k_4k_1e4_256_30
#
#echo "Training model Cnn2_5k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_4k --model_name Cnn2_5k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_4k --model_name Cnn2_5k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_4k_1e4_256_30.txt
#
## Cnn2_5k_3k_1e4_256_30
#
#echo "Training model Cnn2_5k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_3k --model_name Cnn2_5k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_3k --model_name Cnn2_5k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_3k_1e4_256_30.txt
#
## Cnn2_5k_2k_1e4_256_30
#
#echo "Training model Cnn2_5k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_2k --model_name Cnn2_5k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_2k --model_name Cnn2_5k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_2k_1e4_256_30.txt
#
## Cnn2_5k_1k_1e4_256_30
#
#echo "Training model Cnn2_5k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_1k --model_name Cnn2_5k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_1k --model_name Cnn2_5k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_1k_1e4_256_30.txt
#
## Cnn2_5k_5h_1e4_256_30
#
#echo "Training model Cnn2_5k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_5h --model_name Cnn2_5k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_5h --model_name Cnn2_5k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_5h_1e4_256_30.txt
#
## Cnn2_5k_2h_1e4_256_30
#
#echo "Training model Cnn2_5k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_2h --model_name Cnn2_5k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_2h --model_name Cnn2_5k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_2h_1e4_256_30.txt
#
## Cnn2_5k_1h_1e4_256_30
#
#echo "Training model Cnn2_5k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_1h --model_name Cnn2_5k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_1h --model_name Cnn2_5k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_1h_1e4_256_30.txt
#
## Cnn2_5k_10_1e4_256_30
#
#echo "Training model Cnn2_5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5k_10 --model_name Cnn2_5k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5k_10 --model_name Cnn2_5k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5k_10_1e4_256_30.txt
#
## Cnn2_4k_6k_1e4_256_30
#
#echo "Training model Cnn2_4k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_6k --model_name Cnn2_4k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_6k --model_name Cnn2_4k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_6k_1e4_256_30.txt
#
## Cnn2_4k_5k_1e4_256_30
#
#echo "Training model Cnn2_4k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_5k --model_name Cnn2_4k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_5k --model_name Cnn2_4k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_5k_1e4_256_30.txt
#
## Cnn2_4k_4k_1e4_256_30
#
#echo "Training model Cnn2_4k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_4k --model_name Cnn2_4k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_4k --model_name Cnn2_4k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_4k_1e4_256_30.txt
#
## Cnn2_4k_3k_1e4_256_30
#
#echo "Training model Cnn2_4k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_3k --model_name Cnn2_4k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_3k --model_name Cnn2_4k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_3k_1e4_256_30.txt
#
## Cnn2_4k_2k_1e4_256_30
#
#echo "Training model Cnn2_4k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_2k --model_name Cnn2_4k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_2k --model_name Cnn2_4k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_2k_1e4_256_30.txt
#
## Cnn2_4k_1k_1e4_256_30
#
#echo "Training model Cnn2_4k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_1k --model_name Cnn2_4k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_1k --model_name Cnn2_4k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_1k_1e4_256_30.txt
#
## Cnn2_4k_5h_1e4_256_30
#
#echo "Training model Cnn2_4k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_5h --model_name Cnn2_4k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_5h --model_name Cnn2_4k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_5h_1e4_256_30.txt
#
## Cnn2_4k_2h_1e4_256_30
#
#echo "Training model Cnn2_4k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_2h --model_name Cnn2_4k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_2h --model_name Cnn2_4k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_2h_1e4_256_30.txt
#
## Cnn2_4k_1h_1e4_256_30
#
#echo "Training model Cnn2_4k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_1h --model_name Cnn2_4k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_1h --model_name Cnn2_4k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_1h_1e4_256_30.txt
#
## Cnn2_4k_10_1e4_256_30
#
#echo "Training model Cnn2_4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_4k_10 --model_name Cnn2_4k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_4k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_4k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_4k_10 --model_name Cnn2_4k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_4k_10_1e4_256_30.txt
#
## Cnn2_3k_6k_1e4_256_30
#
#echo "Training model Cnn2_3k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_6k --model_name Cnn2_3k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_6k --model_name Cnn2_3k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_6k_1e4_256_30.txt
#
## Cnn2_3k_5k_1e4_256_30
#
#echo "Training model Cnn2_3k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_5k --model_name Cnn2_3k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_5k --model_name Cnn2_3k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_5k_1e4_256_30.txt
#
## Cnn2_3k_4k_1e4_256_30
#
#echo "Training model Cnn2_3k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_4k --model_name Cnn2_3k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_4k --model_name Cnn2_3k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_4k_1e4_256_30.txt
#
## Cnn2_3k_3k_1e4_256_30
#
#echo "Training model Cnn2_3k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_3k --model_name Cnn2_3k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_3k --model_name Cnn2_3k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_3k_1e4_256_30.txt
#
## Cnn2_3k_2k_1e4_256_30
#
#echo "Training model Cnn2_3k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_2k --model_name Cnn2_3k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_2k --model_name Cnn2_3k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_2k_1e4_256_30.txt
#
## Cnn2_3k_1k_1e4_256_30
#
#echo "Training model Cnn2_3k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_1k --model_name Cnn2_3k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_1k --model_name Cnn2_3k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_1k_1e4_256_30.txt
#
## Cnn2_3k_5h_1e4_256_30
#
#echo "Training model Cnn2_3k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_5h --model_name Cnn2_3k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_5h --model_name Cnn2_3k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_5h_1e4_256_30.txt
#
## Cnn2_3k_2h_1e4_256_30
#
#echo "Training model Cnn2_3k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_2h --model_name Cnn2_3k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_2h --model_name Cnn2_3k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_2h_1e4_256_30.txt
#
## Cnn2_3k_1h_1e4_256_30
#
#echo "Training model Cnn2_3k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_1h --model_name Cnn2_3k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_1h --model_name Cnn2_3k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_1h_1e4_256_30.txt
#
## Cnn2_3k_10_1e4_256_30
#
#echo "Training model Cnn2_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_3k_10 --model_name Cnn2_3k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_3k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_3k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_3k_10 --model_name Cnn2_3k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_3k_10_1e4_256_30.txt
#
## Cnn2_2k_6k_1e4_256_30
#
#echo "Training model Cnn2_2k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_6k --model_name Cnn2_2k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_6k --model_name Cnn2_2k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_6k_1e4_256_30.txt
#
## Cnn2_2k_5k_1e4_256_30
#
#echo "Training model Cnn2_2k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_5k --model_name Cnn2_2k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_5k --model_name Cnn2_2k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_5k_1e4_256_30.txt
#
## Cnn2_2k_4k_1e4_256_30
#
#echo "Training model Cnn2_2k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_4k --model_name Cnn2_2k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_4k --model_name Cnn2_2k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_4k_1e4_256_30.txt
#
## Cnn2_2k_3k_1e4_256_30
#
#echo "Training model Cnn2_2k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_3k --model_name Cnn2_2k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_3k --model_name Cnn2_2k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_3k_1e4_256_30.txt
#
## Cnn2_2k_2k_1e4_256_30
#
#echo "Training model Cnn2_2k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_2k --model_name Cnn2_2k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_2k --model_name Cnn2_2k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_2k_1e4_256_30.txt
#
## Cnn2_2k_1k_1e4_256_30
#
#echo "Training model Cnn2_2k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_1k --model_name Cnn2_2k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_1k --model_name Cnn2_2k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_1k_1e4_256_30.txt
#
## Cnn2_2k_5h_1e4_256_30
#
#echo "Training model Cnn2_2k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_5h --model_name Cnn2_2k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_5h --model_name Cnn2_2k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_5h_1e4_256_30.txt
#
## Cnn2_2k_2h_1e4_256_30
#
#echo "Training model Cnn2_2k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_2h --model_name Cnn2_2k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_2h --model_name Cnn2_2k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_2h_1e4_256_30.txt
#
## Cnn2_2k_1h_1e4_256_30
#
#echo "Training model Cnn2_2k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_1h --model_name Cnn2_2k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_1h --model_name Cnn2_2k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_1h_1e4_256_30.txt
#
## Cnn2_2k_10_1e4_256_30
#
#echo "Training model Cnn2_2k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2k_10 --model_name Cnn2_2k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2k_10 --model_name Cnn2_2k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2k_10_1e4_256_30.txt
#
## Cnn2_1k_6k_1e4_256_30
#
#echo "Training model Cnn2_1k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_6k --model_name Cnn2_1k_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_6k --model_name Cnn2_1k_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_6k_1e4_256_30.txt
#
## Cnn2_1k_5k_1e4_256_30
#
#echo "Training model Cnn2_1k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_5k --model_name Cnn2_1k_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_5k --model_name Cnn2_1k_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_5k_1e4_256_30.txt
#
## Cnn2_1k_4k_1e4_256_30
#
#echo "Training model Cnn2_1k_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_4k --model_name Cnn2_1k_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_4k --model_name Cnn2_1k_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_4k_1e4_256_30.txt
#
## Cnn2_1k_3k_1e4_256_30
#
#echo "Training model Cnn2_1k_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_3k --model_name Cnn2_1k_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_3k --model_name Cnn2_1k_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_3k_1e4_256_30.txt
#
## Cnn2_1k_2k_1e4_256_30
#
#echo "Training model Cnn2_1k_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_2k --model_name Cnn2_1k_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_2k --model_name Cnn2_1k_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_2k_1e4_256_30.txt
#
## Cnn2_1k_1k_1e4_256_30
#
#echo "Training model Cnn2_1k_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_1k --model_name Cnn2_1k_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_1k --model_name Cnn2_1k_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_1k_1e4_256_30.txt
#
## Cnn2_1k_5h_1e4_256_30
#
#echo "Training model Cnn2_1k_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_5h --model_name Cnn2_1k_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_5h --model_name Cnn2_1k_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_5h_1e4_256_30.txt
#
## Cnn2_1k_2h_1e4_256_30
#
#echo "Training model Cnn2_1k_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_2h --model_name Cnn2_1k_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_2h --model_name Cnn2_1k_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_2h_1e4_256_30.txt
#
## Cnn2_1k_1h_1e4_256_30
#
#echo "Training model Cnn2_1k_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_1h --model_name Cnn2_1k_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_1h --model_name Cnn2_1k_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_1h_1e4_256_30.txt
#
## Cnn2_1k_10_1e4_256_30
#
#echo "Training model Cnn2_1k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1k_10 --model_name Cnn2_1k_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1k_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1k_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1k_10 --model_name Cnn2_1k_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1k_10_1e4_256_30.txt
#
## Cnn2_5h_6k_1e4_256_30
#
#echo "Training model Cnn2_5h_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_6k --model_name Cnn2_5h_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_6k --model_name Cnn2_5h_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_6k_1e4_256_30.txt
#
## Cnn2_5h_5k_1e4_256_30
#
#echo "Training model Cnn2_5h_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_5k --model_name Cnn2_5h_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_5k --model_name Cnn2_5h_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_5k_1e4_256_30.txt
#
## Cnn2_5h_4k_1e4_256_30
#
#echo "Training model Cnn2_5h_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_4k --model_name Cnn2_5h_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_4k --model_name Cnn2_5h_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_4k_1e4_256_30.txt
#
## Cnn2_5h_3k_1e4_256_30
#
#echo "Training model Cnn2_5h_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_3k --model_name Cnn2_5h_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_3k --model_name Cnn2_5h_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_3k_1e4_256_30.txt
#
## Cnn2_5h_2k_1e4_256_30
#
#echo "Training model Cnn2_5h_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_2k --model_name Cnn2_5h_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_2k --model_name Cnn2_5h_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_2k_1e4_256_30.txt
#
## Cnn2_5h_1k_1e4_256_30
#
#echo "Training model Cnn2_5h_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_1k --model_name Cnn2_5h_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_1k --model_name Cnn2_5h_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_1k_1e4_256_30.txt
#
## Cnn2_5h_5h_1e4_256_30
#
#echo "Training model Cnn2_5h_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_5h --model_name Cnn2_5h_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_5h --model_name Cnn2_5h_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_5h_1e4_256_30.txt
#
## Cnn2_5h_2h_1e4_256_30
#
#echo "Training model Cnn2_5h_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_2h --model_name Cnn2_5h_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_2h --model_name Cnn2_5h_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_2h_1e4_256_30.txt
#
## Cnn2_5h_1h_1e4_256_30
#
#echo "Training model Cnn2_5h_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_1h --model_name Cnn2_5h_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_1h --model_name Cnn2_5h_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_1h_1e4_256_30.txt
#
## Cnn2_5h_10_1e4_256_30
#
#echo "Training model Cnn2_5h_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_5h_10 --model_name Cnn2_5h_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_5h_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_5h_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_5h_10 --model_name Cnn2_5h_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_5h_10_1e4_256_30.txt
#
## Cnn2_2h_6k_1e4_256_30
#
#echo "Training model Cnn2_2h_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_6k --model_name Cnn2_2h_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_6k --model_name Cnn2_2h_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_6k_1e4_256_30.txt
#
## Cnn2_2h_5k_1e4_256_30
#
#echo "Training model Cnn2_2h_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_5k --model_name Cnn2_2h_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_5k --model_name Cnn2_2h_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_5k_1e4_256_30.txt
#
## Cnn2_2h_4k_1e4_256_30
#
#echo "Training model Cnn2_2h_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_4k --model_name Cnn2_2h_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_4k --model_name Cnn2_2h_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_4k_1e4_256_30.txt
#
## Cnn2_2h_3k_1e4_256_30
#
#echo "Training model Cnn2_2h_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_3k --model_name Cnn2_2h_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_3k --model_name Cnn2_2h_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_3k_1e4_256_30.txt
#
## Cnn2_2h_2k_1e4_256_30
#
#echo "Training model Cnn2_2h_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_2k --model_name Cnn2_2h_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_2k --model_name Cnn2_2h_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_2k_1e4_256_30.txt
#
## Cnn2_2h_1k_1e4_256_30
#
#echo "Training model Cnn2_2h_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_1k --model_name Cnn2_2h_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_1k --model_name Cnn2_2h_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_1k_1e4_256_30.txt
#
## Cnn2_2h_5h_1e4_256_30
#
#echo "Training model Cnn2_2h_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_5h --model_name Cnn2_2h_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_5h --model_name Cnn2_2h_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_5h_1e4_256_30.txt
#
## Cnn2_2h_2h_1e4_256_30
#
#echo "Training model Cnn2_2h_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_2h --model_name Cnn2_2h_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_2h --model_name Cnn2_2h_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_2h_1e4_256_30.txt
#
## Cnn2_2h_1h_1e4_256_30
#
#echo "Training model Cnn2_2h_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_1h --model_name Cnn2_2h_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_1h --model_name Cnn2_2h_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_1h_1e4_256_30.txt
#
## Cnn2_2h_10_1e4_256_30
#
#echo "Training model Cnn2_2h_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_2h_10 --model_name Cnn2_2h_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_2h_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_2h_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_2h_10 --model_name Cnn2_2h_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_2h_10_1e4_256_30.txt
#
## Cnn2_1h_6k_1e4_256_30
#
#echo "Training model Cnn2_1h_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_6k --model_name Cnn2_1h_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_6k --model_name Cnn2_1h_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_6k_1e4_256_30.txt
#
## Cnn2_1h_5k_1e4_256_30
#
#echo "Training model Cnn2_1h_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_5k --model_name Cnn2_1h_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_5k --model_name Cnn2_1h_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_5k_1e4_256_30.txt
#
## Cnn2_1h_4k_1e4_256_30
#
#echo "Training model Cnn2_1h_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_4k --model_name Cnn2_1h_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_4k --model_name Cnn2_1h_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_4k_1e4_256_30.txt
#
## Cnn2_1h_3k_1e4_256_30
#
#echo "Training model Cnn2_1h_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_3k --model_name Cnn2_1h_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_3k --model_name Cnn2_1h_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_3k_1e4_256_30.txt
#
## Cnn2_1h_2k_1e4_256_30
#
#echo "Training model Cnn2_1h_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_2k --model_name Cnn2_1h_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_2k --model_name Cnn2_1h_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_2k_1e4_256_30.txt
#
## Cnn2_1h_1k_1e4_256_30
#
#echo "Training model Cnn2_1h_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_1k --model_name Cnn2_1h_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_1k --model_name Cnn2_1h_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_1k_1e4_256_30.txt
#
## Cnn2_1h_5h_1e4_256_30
#
#echo "Training model Cnn2_1h_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_5h --model_name Cnn2_1h_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_5h --model_name Cnn2_1h_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_5h_1e4_256_30.txt
#
## Cnn2_1h_2h_1e4_256_30
#
#echo "Training model Cnn2_1h_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_2h --model_name Cnn2_1h_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_2h --model_name Cnn2_1h_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_2h_1e4_256_30.txt
#
## Cnn2_1h_1h_1e4_256_30
#
#echo "Training model Cnn2_1h_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_1h --model_name Cnn2_1h_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_1h --model_name Cnn2_1h_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_1h_1e4_256_30.txt
#
## Cnn2_1h_10_1e4_256_30
#
#echo "Training model Cnn2_1h_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_1h_10 --model_name Cnn2_1h_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_1h_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_1h_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_1h_10 --model_name Cnn2_1h_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_1h_10_1e4_256_30.txt
#
## Cnn2_10_6k_1e4_256_30
#
#echo "Training model Cnn2_10_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_6k --model_name Cnn2_10_6k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_6k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_6k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_6k --model_name Cnn2_10_6k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_6k_1e4_256_30.txt
#
## Cnn2_10_5k_1e4_256_30
#
#echo "Training model Cnn2_10_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_5k --model_name Cnn2_10_5k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_5k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_5k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_5k --model_name Cnn2_10_5k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_5k_1e4_256_30.txt
#
## Cnn2_10_4k_1e4_256_30
#
#echo "Training model Cnn2_10_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_4k --model_name Cnn2_10_4k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_4k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_4k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_4k --model_name Cnn2_10_4k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_4k_1e4_256_30.txt
#
## Cnn2_10_3k_1e4_256_30
#
#echo "Training model Cnn2_10_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_3k --model_name Cnn2_10_3k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_3k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_3k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_3k --model_name Cnn2_10_3k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_3k_1e4_256_30.txt
#
## Cnn2_10_2k_1e4_256_30
#
#echo "Training model Cnn2_10_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_2k --model_name Cnn2_10_2k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_2k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_2k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_2k --model_name Cnn2_10_2k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_2k_1e4_256_30.txt
#
## Cnn2_10_1k_1e4_256_30
#
#echo "Training model Cnn2_10_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_1k --model_name Cnn2_10_1k_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_1k_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_1k_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_1k --model_name Cnn2_10_1k_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_1k_1e4_256_30.txt
#
## Cnn2_10_5h_1e4_256_30
#
#echo "Training model Cnn2_10_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_5h --model_name Cnn2_10_5h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_5h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_5h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_5h --model_name Cnn2_10_5h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_5h_1e4_256_30.txt
#
## Cnn2_10_2h_1e4_256_30
#
#echo "Training model Cnn2_10_2h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_2h --model_name Cnn2_10_2h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_2h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_2h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_2h --model_name Cnn2_10_2h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_2h_1e4_256_30.txt
#
## Cnn2_10_1h_1e4_256_30
#
#echo "Training model Cnn2_10_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_1h --model_name Cnn2_10_1h_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_1h_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_1h_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_1h --model_name Cnn2_10_1h_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_1h_1e4_256_30.txt
#
## Cnn2_10_10_1e4_256_30
#
#echo "Training model Cnn2_10_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 20 --lr 1e-4 --batch_size 256 --patience 30 \
#              --classifier Cnn2_10_10 --model_name Cnn2_10_10_1e4_256_30 > ../Analysis/logs/train/step3/Cnn2_10_10_1e4_256_30.txt
#
#echo "Evaluating model Cnn2_10_10_1e4_256_30"
#python ../eval.py --train_path $trn --test_path $tst \
#              --model_folder step3 \
#              --classifier Cnn2_10_10 --model_name Cnn2_10_10_1e4_256_30 > ../Analysis/logs/eval/step3/Cnn2_10_10_1e4_256_30.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../traineval2excel.py --xls_name 'CNN_step3' --archives_folder 'step3'