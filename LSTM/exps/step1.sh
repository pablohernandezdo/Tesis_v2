#!/bin/bash

mkdir -p ../Analysis/logs/train/step1
mkdir -p ../Analysis/logs/eval/step1
mkdir -p ../models/step1

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Primera vez
# Una capa de salida

echo "Training model Lstm_64_64_5_1_1e3_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_1_1e3_256.txt

echo "Evaluating model Lstm_64_64_5_1_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e3_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_1_1e3_256.txt

echo "Training model Lstm_64_64_5_1_1e4_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_1_1e4_256.txt

echo "Evaluating model Lstm_64_64_5_1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e4_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_1_1e4_256.txt

echo "Training model Lstm_64_64_5_1_1e5_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e5_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_1_1e5_256.txt

echo "Evaluating model Lstm_64_64_5_1_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e5_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_1_1e5_256.txt

echo "Training model Lstm_64_64_5_1_1e6_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e6_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_1_1e6_256.txt

echo "Evaluating model Lstm_64_64_5_1_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_1 --model_name Lstm_64_64_5_1_1e6_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_1_1e6_256.txt

# Dos capas de salida

echo "Training model Lstm_64_64_5_2_1e3_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e3_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_2_1e3_256.txt

echo "Evaluating model Lstm_64_64_5_2_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e3_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_2_1e3_256.txt

echo "Training model Lstm_64_64_5_2_1e4_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e4_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_2_1e4_256.txt

echo "Evaluating model Lstm_64_64_5_2_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e4_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_2_1e4_256.txt

echo "Training model Lstm_64_64_5_2_1e5_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e5_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_2_1e5_256.txt

echo "Evaluating model Lstm_64_64_5_2_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e5_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_2_1e5_256.txt

echo "Training model Lstm_64_64_5_2_1e6_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e6_256 > ../Analysis/logs/train/step1/Lstm_64_64_5_2_1e6_256.txt

echo "Evaluating model Lstm_64_64_5_2_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_64_64_5_2 --model_name Lstm_64_64_5_2_1e6_256 > ../Analysis/logs/eval/step1/Lstm_64_64_5_2_1e6_256.txt

# Segunda vez

echo "Training model Lstm_16_16_1_1_1e3_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_1_1e3_256.txt

echo "Evaluating model Lstm_16_16_1_1_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e3_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_1_1e3_256.txt

echo "Training model Lstm_16_16_1_1_1e3_256, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_1_1e4_256.txt

echo "Evaluating model Lstm_16_16_1_1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e4_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_1_1e4_256.txt

echo "Training model Lstm_16_16_1_1_1e5_256, lr = 1e-5, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e5_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_1_1e5_256.txt

echo "Evaluating model Lstm_16_16_1_1_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e5_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_1_1e5_256.txt

echo "Training model Lstm_16_16_1_1_1e6_256, lr = 1e-6, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e6_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_1_1e6_256.txt

echo "Evaluating model Lstm_16_16_1_1_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_1 --model_name Lstm_16_16_1_1_1e6_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_1_1e6_256.txt

echo "Training model Lstm_16_16_1_2_1e3_256, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e3_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_2_1e3_256.txt

echo "Evaluating model Lstm_16_16_1_2_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e3_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_2_1e3_256.txt

echo "Training model Lstm_16_16_1_2_1e3_256, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e4_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_2_1e4_256.txt

echo "Evaluating model Lstm_16_16_1_2_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e4_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_2_1e4_256.txt

echo "Training model Lstm_16_16_1_2_1e5_256, lr = 1e-5, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e5_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_2_1e5_256.txt

echo "Evaluating model Lstm_16_16_1_2_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e5_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_2_1e5_256.txt

echo "Training model Lstm_16_16_1_2_1e6_256, lr = 1e-6, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e6_256 > ../Analysis/logs/train/step1/Lstm_16_16_1_2_1e6_256.txt

echo "Evaluating model Lstm_16_16_1_2_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier Lstm_16_16_1_2 --model_name Lstm_16_16_1_2_1e6_256 > ../Analysis/logs/eval/step1/Lstm_16_16_1_2_1e6_256.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../traineval2excel.py --xls_name 'LSTM_step1' --archives_folder 'step1'
