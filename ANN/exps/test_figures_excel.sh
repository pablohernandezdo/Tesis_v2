#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# 2h5h5k

## Learning rate 1e-3
#echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 20 \
#              --train_path $trn --val_path $val      \
#              --epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_20.txt
#
#echo "Evaluating model 2h5h5k_1e3_256"
#python ../eval.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_20.txt
#
#echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 25 \
#              --train_path $trn --val_path $val      \
#              --epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_25 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_25.txt
#
#echo "Evaluating model 2h5h5k_1e3_256"
#python ../eval.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_25 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_25.txt
#
#echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 30 \
#              --train_path $trn --val_path $val      \
#              --epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_30 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_30.txt
#
#echo "Evaluating model 2h5h5k_1e3_256"
#python ../eval.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_30 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_30.txt
#
#
#echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 35 \
#              --train_path $trn --val_path $val      \
#              --epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_35 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_35.txt
#
#echo "Evaluating model 2h5h5k_1e3_256"
#python ../eval.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_35 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_35.txt
#
#echo "Training model 2h5h5k, lr = 1e-3, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --model_folder step4 --patience 40 \
#              --train_path $trn --val_path $val      \
#              --epochs 5 --lr 1e-3 --batch_size 256 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_40 > ../Analysis/logs/train/step4/2h5h5k_1e3_256_40.txt
#
#echo "Evaluating model 2h5h5k_1e3_256"
#python ../eval.py --test_path $tst \
#              --model_folder step4 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_40 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_40.txt

# Figures and full excel file
python ../figures_and_excel.py \
              --csv_folder ../Analysis/CSVOutputs/Test/step4 \
              --xls_name test \
