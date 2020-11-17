#!/bin/bash

mkdir -p ../Analysis/logs/train/step1
mkdir -p ../Analysis/logs/eval/step1
mkdir -p ../models/step1

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Type 1, 1 hidden layer models

echo "Training model 1c3k, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier 1c3k --model_name 1c3k_1e3_256 > ../Analysis/logs/train/step1/1c3k_1e3_256.txt

echo "Evaluating model 1c3k_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k --model_name 1c3k_1e3_256 > ../Analysis/logs/eval/step1/1c3k_1e3_256.txt

echo "Training model 1c3k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1c3k --model_name 1c3k_1e4_256 > ../Analysis/logs/train/step1/1c3k_1e4_256.txt

echo "Evaluating model 1c3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k --model_name 1c3k_1e4_256 > ../Analysis/logs/eval/step1/1c3k_1e4_256.txt

echo "Training model 1c3k, lr = 1e-5, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier 1c3k --model_name 1c3k_1e5_256 > ../Analysis/logs/train/step1/1c3k_1e5_256.txt

echo "Evaluating model 1c3k_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k --model_name 1c3k_1e5_256 > ../Analysis/logs/eval/step1/1c3k_1e5_256.txt

echo "Training model 1c3k, lr = 1e-6, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier 1c3k --model_name 1c3k_1e6_256 > ../Analysis/logs/train/step1/1c3k_1e6_256.txt

echo "Evaluating model 1c3k_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k --model_name 1c3k_1e6_256 > ../Analysis/logs/eval/step1/1c3k_1e6_256.txt

# Type 1, 2 hidden layer models

echo "Training model 1c3k3k, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier 1c3k3k --model_name 1c3k3k_1e3_256 > ../Analysis/logs/train/step1/1c3k3k_1e3_256.txt

echo "Evaluating model 1c3k3k_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k3k --model_name 1c3k3k_1e3_256 > ../Analysis/logs/eval/step1/1c3k3k_1e3_256.txt

echo "Training model 1c3k3k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1c3k3k --model_name 1c3k3k_1e4_256 > ../Analysis/logs/train/step1/1c3k3k_1e4_256.txt

echo "Evaluating model 1c3k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k3k --model_name 1c3k3k_1e4_256 > ../Analysis/logs/eval/step1/1c3k3k_1e4_256.txt

echo "Training model 1c3k3k, lr = 1e-5, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier 1c3k3k --model_name 1c3k3k_1e5_256 > ../Analysis/logs/train/step1/1c3k3k_1e5_256.txt

echo "Evaluating model 1c3k3k_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k3k --model_name 1c3k3k_1e5_256 > ../Analysis/logs/eval/step1/1c3k3k_1e5_256.txt

echo "Training model 1c3k3k, lr = 1e-6, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier 1c3k3k --model_name 1c3k3k_1e6_256 > ../Analysis/logs/train/step1/1c3k3k_1e6_256.txt

echo "Evaluating model 1c3k3k_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 1c3k3k --model_name 1c3k3k_1e6_256 > ../Analysis/logs/eval/step1/1c3k3k_1e6_256.txt

# Type 2, 1 hidden layer models

echo "Training model 2c3k, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier 2c3k --model_name 2c3k_1e3_256 > ../Analysis/logs/train/step1/2c3k_1e3_256.txt

echo "Evaluating model 2c3k_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k --model_name 2c3k_1e3_256 > ../Analysis/logs/eval/step1/2c3k_1e3_256.txt

echo "Training model 2c3k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 2c3k --model_name 2c3k_1e4_256 > ../Analysis/logs/train/step1/2c3k_1e4_256.txt

echo "Evaluating model 2c3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k --model_name 2c3k_1e4_256 > ../Analysis/logs/eval/step1/2c3k_1e4_256.txt

echo "Training model 2c3k, lr = 1e-5, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier 2c3k --model_name 2c3k_1e5_256 > ../Analysis/logs/train/step1/2c3k_1e5_256.txt

echo "Evaluating model 2c3k_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k --model_name 2c3k_1e5_256 > ../Analysis/logs/eval/step1/2c3k_1e5_256.txt

echo "Training model 2c3k, lr = 1e-6, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier 2c3k --model_name 2c3k_1e6_256 > ../Analysis/logs/train/step1/2c3k_1e6_256.txt

echo "Evaluating model 2c3k_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k --model_name 2c3k_1e6_256 > ../Analysis/logs/eval/step1/2c3k_1e6_256.txt

# Type 2, 2 hidden layer models

echo "Training model 2c3k3k, lr = 1e-3, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-3 --batch_size 256 \
              --classifier 2c3k3k --model_name 2c3k3k_1e3_256 > ../Analysis/logs/train/step1/2c3k3k_1e3_256.txt

echo "Evaluating model 2c3k3k_1e3_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k3k --model_name 2c3k3k_1e3_256 > ../Analysis/logs/eval/step1/2c3k3k_1e3_256.txt

echo "Training model 2c3k3k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 2c3k3k --model_name 2c3k3k_1e4_256 > ../Analysis/logs/train/step1/2c3k3k_1e4_256.txt

echo "Evaluating model 2c3k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k3k --model_name 2c3k3k_1e4_256 > ../Analysis/logs/eval/step1/2c3k3k_1e4_256.txt

echo "Training model 2c3k3k, lr = 1e-5, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-5 --batch_size 256 \
              --classifier 2c3k3k --model_name 2c3k3k_1e5_256 > ../Analysis/logs/train/step1/2c3k3k_1e5_256.txt

echo "Evaluating model 2c3k3k_1e5_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k3k --model_name 2c3k3k_1e5_256 > ../Analysis/logs/eval/step1/2c3k3k_1e5_256.txt

echo "Training model 2c3k3k, lr = 1e-6, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --earlystop 0 --model_folder step1  \
              --n_epochs 5 --lr 1e-6 --batch_size 256 \
              --classifier 2c3k3k --model_name 2c3k3k_1e6_256 > ../Analysis/logs/train/step1/2c3k3k_1e6_256.txt

echo "Evaluating model 2c3k3k_1e6_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step1 \
              --classifier 2c3k3k --model_name 2c3k3k_1e6_256 > ../Analysis/logs/eval/step1/2c3k3k_1e6_256.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../traineval2excel.py --xls_name 'CNN_step1' --archives_folder 'step1'
