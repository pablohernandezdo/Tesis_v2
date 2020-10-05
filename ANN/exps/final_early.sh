#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Training model 2h5k6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_10 > ../logs/train/2h5k6k_1e4_256_10.txt

echo "Evaluating model 2h5k6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_10 > ../logs/eval/2h5k6k_1e4_256_10.txt

echo "Training model 2h5k6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_20 > ../logs/train/2h5k6k_1e4_256_20.txt

echo "Evaluating model 2h5k6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_20 > ../logs/eval/2h5k6k_1e4_256_20.txt

echo "Training model 2h5k6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_25 > ../logs/train/2h5k6k_1e4_256_25.txt

echo "Evaluating model 2h5k6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_25 > ../logs/eval/2h5k6k_1e4_256_25.txt

echo "Training model 2h5k6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_35 > ../logs/train/2h5k6k_1e4_256_35.txt

echo "Evaluating model 2h5k6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_35 > ../logs/eval/2h5k6k_1e4_256_35.txt

echo "Training model 2h5k6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_40 > ../logs/train/2h5k6k_1e4_256_40.txt

echo "Evaluating model 2h5k6k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_40 > ../logs/eval/2h5k6k_1e4_256_40.txt

echo "Training model 2h5k5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_10 > ../logs/train/2h5k5k_1e4_256_10.txt

echo "Evaluating model 2h5k5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_10 > ../logs/eval/2h5k5k_1e4_256_10.txt

echo "Training model 2h5k5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_20 > ../logs/train/2h5k5k_1e4_256_20.txt

echo "Evaluating model 2h5k5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_20 > ../logs/eval/2h5k5k_1e4_256_20.txt

echo "Training model 2h5k5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_25 > ../logs/train/2h5k5k_1e4_256_25.txt

echo "Evaluating model 2h5k5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_25 > ../logs/eval/2h5k5k_1e4_256_25.txt

echo "Training model 2h5k5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_35 > ../logs/train/2h5k5k_1e4_256_35.txt

echo "Evaluating model 2h5k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_35 > ../logs/eval/2h5k5k_1e4_256_35.txt

echo "Training model 2h5k5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_40 > ../logs/train/2h5k5k_1e4_256_40.txt

echo "Evaluating model 2h5k5k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_40 > ../logs/eval/2h5k5k_1e4_256_40.txt

echo "Training model 2h5k4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_10 > ../logs/train/2h5k4k_1e4_256_10.txt

echo "Evaluating model 2h5k4k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_10 > ../logs/eval/2h5k4k_1e4_256_10.txt

echo "Training model 2h5k4k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_20 > ../logs/train/2h5k4k_1e4_256_20.txt

echo "Evaluating model 2h5k4k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_20 > ../logs/eval/2h5k4k_1e4_256_20.txt

echo "Training model 2h5k4k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_25 > ../logs/train/2h5k4k_1e4_256_25.txt

echo "Evaluating model 2h5k4k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_25 > ../logs/eval/2h5k4k_1e4_256_25.txt

echo "Training model 2h5k4k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_35 > ../logs/train/2h5k4k_1e4_256_35.txt

echo "Evaluating model 2h5k4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_35 > ../logs/eval/2h5k4k_1e4_256_35.txt

echo "Training model 2h5k4k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_40 > ../logs/train/2h5k4k_1e4_256_40.txt

echo "Evaluating model 2h5k4k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_40 > ../logs/eval/2h5k4k_1e4_256_40.txt

echo "Training model 2h5k3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_10 > ../logs/train/2h5k3k_1e4_256_10.txt

echo "Evaluating model 2h5k3k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_10 > ../logs/eval/2h5k3k_1e4_256_10.txt

echo "Training model 2h5k3k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_20 > ../logs/train/2h5k3k_1e4_256_20.txt

echo "Evaluating model 2h5k3k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_20 > ../logs/eval/2h5k3k_1e4_256_20.txt

echo "Training model 2h5k3k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_25 > ../logs/train/2h5k3k_1e4_256_25.txt

echo "Evaluating model 2h5k3k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_25 > ../logs/eval/2h5k3k_1e4_256_25.txt

echo "Training model 2h5k3k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_35 > ../logs/train/2h5k3k_1e4_256_35.txt

echo "Evaluating model 2h5k3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_35 > ../logs/eval/2h5k3k_1e4_256_35.txt

echo "Training model 2h5k3k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_40 > ../logs/train/2h5k3k_1e4_256_40.txt

echo "Evaluating model 2h5k3k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_40 > ../logs/eval/2h5k3k_1e4_256_40.txt

echo "Training model 2h5k5h_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_10 > ../logs/train/2h5k5h_1e4_256_10.txt

echo "Evaluating model 2h5k5h_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_10 > ../logs/eval/2h5k5h_1e4_256_10.txt

echo "Training model 2h5k5h_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_20 > ../logs/train/2h5k5h_1e4_256_20.txt

echo "Evaluating model 2h5k5h_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_20 > ../logs/eval/2h5k5h_1e4_256_20.txt

echo "Training model 2h5k5h_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_25 > ../logs/train/2h5k5h_1e4_256_25.txt

echo "Evaluating model 2h5k5h_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_25 > ../logs/eval/2h5k5h_1e4_256_25.txt

echo "Training model 2h5k5h_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_35 > ../logs/train/2h5k5h_1e4_256_35.txt

echo "Evaluating model 2h5k5h_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_35 > ../logs/eval/2h5k5h_1e4_256_35.txt

echo "Training model 2h5k5h_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_40 > ../logs/train/2h5k5h_1e4_256_40.txt

echo "Evaluating model 2h5k5h_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_40 > ../logs/eval/2h5k5h_1e4_256_40.txt

echo "Training model 2h4k6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_10 > ../logs/train/2h4k6k_1e4_256_10.txt

echo "Evaluating model 2h4k6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_10 > ../logs/eval/2h4k6k_1e4_256_10.txt

echo "Training model 2h4k6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_20 > ../logs/train/2h4k6k_1e4_256_20.txt

echo "Evaluating model 2h4k6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_20 > ../logs/eval/2h4k6k_1e4_256_20.txt

echo "Training model 2h4k6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_25 > ../logs/train/2h4k6k_1e4_256_25.txt

echo "Evaluating model 2h4k6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_25 > ../logs/eval/2h4k6k_1e4_256_25.txt

echo "Training model 2h4k6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_35 > ../logs/train/2h4k6k_1e4_256_35.txt

echo "Evaluating model 2h4k6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_35 > ../logs/eval/2h4k6k_1e4_256_35.txt

echo "Training model 2h4k6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_40 > ../logs/train/2h4k6k_1e4_256_40.txt

echo "Evaluating model 2h4k6k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_40 > ../logs/eval/2h4k6k_1e4_256_40.txt

echo "Training model 2h4k5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_10 > ../logs/train/2h4k5k_1e4_256_10.txt

echo "Evaluating model 2h4k5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_10 > ../logs/eval/2h4k5k_1e4_256_10.txt

echo "Training model 2h4k5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_20 > ../logs/train/2h4k5k_1e4_256_20.txt

echo "Evaluating model 2h4k5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_20 > ../logs/eval/2h4k5k_1e4_256_20.txt

echo "Training model 2h4k5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_25 > ../logs/train/2h4k5k_1e4_256_25.txt

echo "Evaluating model 2h4k5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_25 > ../logs/eval/2h4k5k_1e4_256_25.txt

echo "Training model 2h4k5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_35 > ../logs/train/2h4k5k_1e4_256_35.txt

echo "Evaluating model 2h4k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_35 > ../logs/eval/2h4k5k_1e4_256_35.txt

echo "Training model 2h4k5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_40 > ../logs/train/2h4k5k_1e4_256_40.txt

echo "Evaluating model 2h4k5k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_40 > ../logs/eval/2h4k5k_1e4_256_40.txt

echo "Training model 2h4k3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_10 > ../logs/train/2h4k3k_1e4_256_10.txt

echo "Evaluating model 2h4k3k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_10 > ../logs/eval/2h4k3k_1e4_256_10.txt

echo "Training model 2h4k3k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_20 > ../logs/train/2h4k3k_1e4_256_20.txt

echo "Evaluating model 2h4k3k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_20 > ../logs/eval/2h4k3k_1e4_256_20.txt

echo "Training model 2h4k3k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_25 > ../logs/train/2h4k3k_1e4_256_25.txt

echo "Evaluating model 2h4k3k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_25 > ../logs/eval/2h4k3k_1e4_256_25.txt

echo "Training model 2h4k3k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_35 > ../logs/train/2h4k3k_1e4_256_35.txt

echo "Evaluating model 2h4k3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_35 > ../logs/eval/2h4k3k_1e4_256_35.txt

echo "Training model 2h4k3k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_40 > ../logs/train/2h4k3k_1e4_256_40.txt

echo "Evaluating model 2h4k3k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_40 > ../logs/eval/2h4k3k_1e4_256_40.txt

echo "Training model 2h3k6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_10 > ../logs/train/2h3k6k_1e4_256_10.txt

echo "Evaluating model 2h3k6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_10 > ../logs/eval/2h3k6k_1e4_256_10.txt

echo "Training model 2h3k6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_20 > ../logs/train/2h3k6k_1e4_256_20.txt

echo "Evaluating model 2h3k6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_20 > ../logs/eval/2h3k6k_1e4_256_20.txt

echo "Training model 2h3k6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_25 > ../logs/train/2h3k6k_1e4_256_25.txt

echo "Evaluating model 2h3k6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_25 > ../logs/eval/2h3k6k_1e4_256_25.txt

echo "Training model 2h3k6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_35 > ../logs/train/2h3k6k_1e4_256_35.txt

echo "Evaluating model 2h3k6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_35 > ../logs/eval/2h3k6k_1e4_256_35.txt

echo "Training model 2h3k6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_40 > ../logs/train/2h3k6k_1e4_256_40.txt

echo "Evaluating model 2h3k6k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_40 > ../logs/eval/2h3k6k_1e4_256_40.txt

echo "Training model 2h3k5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_10 > ../logs/train/2h3k5k_1e4_256_10.txt

echo "Evaluating model 2h3k5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_10 > ../logs/eval/2h3k5k_1e4_256_10.txt

echo "Training model 2h3k5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_20 > ../logs/train/2h3k5k_1e4_256_20.txt

echo "Evaluating model 2h3k5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_20 > ../logs/eval/2h3k5k_1e4_256_20.txt

echo "Training model 2h3k5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_25 > ../logs/train/2h3k5k_1e4_256_25.txt

echo "Evaluating model 2h3k5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_25 > ../logs/eval/2h3k5k_1e4_256_25.txt

echo "Training model 2h3k5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_35 > ../logs/train/2h3k5k_1e4_256_35.txt

echo "Evaluating model 2h3k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_35 > ../logs/eval/2h3k5k_1e4_256_35.txt

echo "Training model 2h3k5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_40 > ../logs/train/2h3k5k_1e4_256_40.txt

echo "Evaluating model 2h3k5k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_40 > ../logs/eval/2h3k5k_1e4_256_40.txt

echo "Training model 2h2k6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_10 > ../logs/train/2h2k6k_1e4_256_10.txt

echo "Evaluating model 2h2k6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_10 > ../logs/eval/2h2k6k_1e4_256_10.txt

echo "Training model 2h2k6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_20 > ../logs/train/2h2k6k_1e4_256_20.txt

echo "Evaluating model 2h2k6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_20 > ../logs/eval/2h2k6k_1e4_256_20.txt

echo "Training model 2h2k6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_25 > ../logs/train/2h2k6k_1e4_256_25.txt

echo "Evaluating model 2h2k6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_25 > ../logs/eval/2h2k6k_1e4_256_25.txt

echo "Training model 2h2k6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_35 > ../logs/train/2h2k6k_1e4_256_35.txt

echo "Evaluating model 2h2k6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_35 > ../logs/eval/2h2k6k_1e4_256_35.txt

echo "Training model 2h2k6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_40 > ../logs/train/2h2k6k_1e4_256_40.txt

echo "Evaluating model 2h2k6k_1e4_256_40"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_40 > ../logs/eval/2h2k6k_1e4_256_40.txt

echo "Training model 2h2k5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_10 > ../logs/train/2h2k5k_1e4_256_10.txt

echo "Evaluating model 2h2k5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_10 > ../logs/eval/2h2k5k_1e4_256_10.txt

echo "Training model 2h2k5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_20 > ../logs/train/2h2k5k_1e4_256_20.txt

echo "Evaluating model 2h2k5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_20 > ../logs/eval/2h2k5k_1e4_256_20.txt

echo "Training model 2h2k5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_25 > ../logs/train/2h2k5k_1e4_256_25.txt

echo "Evaluating model 2h2k5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_25 > ../logs/eval/2h2k5k_1e4_256_25.txt

echo "Training model 2h2k5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_35 > ../logs/train/2h2k5k_1e4_256_35.txt

echo "Evaluating model 2h2k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_35 > ../logs/eval/2h2k5k_1e4_256_35.txt

echo "Training model 2h2k5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_40 > ../logs/train/2h2k5k_1e4_256_40.txt

echo "Evaluating model 2h2k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_40 > ../logs/eval/2h2k5k_1e4_256_40.txt

echo "Training model 2h2k4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_10 > ../logs/train/2h2k4k_1e4_256_10.txt

echo "Evaluating model 2h2k4k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_10 > ../logs/eval/2h2k4k_1e4_256_10.txt

echo "Training model 2h2k4k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_20 > ../logs/train/2h2k4k_1e4_256_20.txt

echo "Evaluating model 2h2k4k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_20 > ../logs/eval/2h2k4k_1e4_256_20.txt

echo "Training model 2h2k4k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_25 > ../logs/train/2h2k4k_1e4_256_25.txt

echo "Evaluating model 2h2k4k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_25 > ../logs/eval/2h2k4k_1e4_256_25.txt

echo "Training model 2h2k4k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_35 > ../logs/train/2h2k4k_1e4_256_35.txt

echo "Evaluating model 2h2k4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_35 > ../logs/eval/2h2k4k_1e4_256_35.txt

echo "Training model 2h2k4k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_40 > ../logs/train/2h2k4k_1e4_256_40.txt

echo "Evaluating model 2h2k4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_40 > ../logs/eval/2h2k4k_1e4_256_40.txt

echo "Training model 2h2k3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_10 > ../logs/train/2h2k3k_1e4_256_10.txt

echo "Evaluating model 2h2k3k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_10 > ../logs/eval/2h2k3k_1e4_256_10.txt

echo "Training model 2h2k3k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_20 > ../logs/train/2h2k3k_1e4_256_20.txt

echo "Evaluating model 2h2k3k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_20 > ../logs/eval/2h2k3k_1e4_256_20.txt

echo "Training model 2h2k3k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_25 > ../logs/train/2h2k3k_1e4_256_25.txt

echo "Evaluating model 2h2k3k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_25 > ../logs/eval/2h2k3k_1e4_256_25.txt

echo "Training model 2h2k3k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_35 > ../logs/train/2h2k3k_1e4_256_35.txt

echo "Evaluating model 2h2k3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_35 > ../logs/eval/2h2k3k_1e4_256_35.txt

echo "Training model 2h2k3k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_40 > ../logs/train/2h2k3k_1e4_256_40.txt

echo "Evaluating model 2h2k3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_40 > ../logs/eval/2h2k3k_1e4_256_40.txt

echo "Training model 2h2k2k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_10 > ../logs/train/2h2k2k_1e4_256_10.txt

echo "Evaluating model 2h2k2k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_10 > ../logs/eval/2h2k2k_1e4_256_10.txt

echo "Training model 2h2k2k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_20 > ../logs/train/2h2k2k_1e4_256_20.txt

echo "Evaluating model 2h2k2k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_20 > ../logs/eval/2h2k2k_1e4_256_20.txt

echo "Training model 2h2k2k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_25 > ../logs/train/2h2k2k_1e4_256_25.txt

echo "Evaluating model 2h2k2k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_25 > ../logs/eval/2h2k2k_1e4_256_25.txt

echo "Training model 2h2k2k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_35 > ../logs/train/2h2k2k_1e4_256_35.txt

echo "Evaluating model 2h2k2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_35 > ../logs/eval/2h2k2k_1e4_256_35.txt

echo "Training model 2h2k2k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_40 > ../logs/train/2h2k2k_1e4_256_40.txt

echo "Evaluating model 2h2k2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_40 > ../logs/eval/2h2k2k_1e4_256_40.txt

echo "Training model 2h1k6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_10 > ../logs/train/2h1k6k_1e4_256_10.txt

echo "Evaluating model 2h1k6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_10 > ../logs/eval/2h1k6k_1e4_256_10.txt

echo "Training model 2h1k6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_20 > ../logs/train/2h1k6k_1e4_256_20.txt

echo "Evaluating model 2h1k6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_20 > ../logs/eval/2h1k6k_1e4_256_20.txt

echo "Training model 2h1k6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_25 > ../logs/train/2h1k6k_1e4_256_25.txt

echo "Evaluating model 2h1k6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_25 > ../logs/eval/2h1k6k_1e4_256_25.txt

echo "Training model 2h1k6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_35 > ../logs/train/2h1k6k_1e4_256_35.txt

echo "Evaluating model 2h1k6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_35 > ../logs/eval/2h1k6k_1e4_256_35.txt

echo "Training model 2h1k6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_40 > ../logs/train/2h1k6k_1e4_256_40.txt

echo "Evaluating model 2h1k6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_40 > ../logs/eval/2h1k6k_1e4_256_40.txt

echo "Training model 2h1k5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_10 > ../logs/train/2h1k5k_1e4_256_10.txt

echo "Evaluating model 2h1k5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_10 > ../logs/eval/2h1k5k_1e4_256_10.txt

echo "Training model 2h1k5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_20 > ../logs/train/2h1k5k_1e4_256_20.txt

echo "Evaluating model 2h1k5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_20 > ../logs/eval/2h1k5k_1e4_256_20.txt

echo "Training model 2h1k5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_25 > ../logs/train/2h1k5k_1e4_256_25.txt

echo "Evaluating model 2h1k5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_25 > ../logs/eval/2h1k5k_1e4_256_25.txt

echo "Training model 2h1k5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_35 > ../logs/train/2h1k5k_1e4_256_35.txt

echo "Evaluating model 2h1k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_35 > ../logs/eval/2h1k5k_1e4_256_35.txt

echo "Training model 2h1k5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_40 > ../logs/train/2h1k5k_1e4_256_40.txt

echo "Evaluating model 2h1k5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_40 > ../logs/eval/2h1k5k_1e4_256_40.txt

echo "Training model 2h1k4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_10 > ../logs/train/2h1k4k_1e4_256_10.txt

echo "Evaluating model 2h1k4k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_10 > ../logs/eval/2h1k4k_1e4_256_10.txt

echo "Training model 2h1k4k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_20 > ../logs/train/2h1k4k_1e4_256_20.txt

echo "Evaluating model 2h1k4k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_20 > ../logs/eval/2h1k4k_1e4_256_20.txt

echo "Training model 2h1k4k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_25 > ../logs/train/2h1k4k_1e4_256_25.txt

echo "Evaluating model 2h1k4k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_25 > ../logs/eval/2h1k4k_1e4_256_25.txt

echo "Training model 2h1k4k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_35 > ../logs/train/2h1k4k_1e4_256_35.txt

echo "Evaluating model 2h1k4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_35 > ../logs/eval/2h1k4k_1e4_256_35.txt

echo "Training model 2h1k4k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_40 > ../logs/train/2h1k4k_1e4_256_40.txt

echo "Evaluating model 2h1k4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_40 > ../logs/eval/2h1k4k_1e4_256_40.txt

echo "Training model 2h1k3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_10 > ../logs/train/2h1k3k_1e4_256_10.txt

echo "Evaluating model 2h1k3k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_10 > ../logs/eval/2h1k3k_1e4_256_10.txt

echo "Training model 2h1k3k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_20 > ../logs/train/2h1k3k_1e4_256_20.txt

echo "Evaluating model 2h1k3k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_20 > ../logs/eval/2h1k3k_1e4_256_20.txt

echo "Training model 2h1k3k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_25 > ../logs/train/2h1k3k_1e4_256_25.txt

echo "Evaluating model 2h1k3k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_25 > ../logs/eval/2h1k3k_1e4_256_25.txt

echo "Training model 2h1k3k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_35 > ../logs/train/2h1k3k_1e4_256_35.txt

echo "Evaluating model 2h1k3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_35 > ../logs/eval/2h1k3k_1e4_256_35.txt

echo "Training model 2h1k3k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_40 > ../logs/train/2h1k3k_1e4_256_40.txt

echo "Evaluating model 2h1k3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_40 > ../logs/eval/2h1k3k_1e4_256_40.txt

echo "Training model 2h1k2k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_10 > ../logs/train/2h1k2k_1e4_256_10.txt

echo "Evaluating model 2h1k2k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_10 > ../logs/eval/2h1k2k_1e4_256_10.txt

echo "Training model 2h1k2k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_20 > ../logs/train/2h1k2k_1e4_256_20.txt

echo "Evaluating model 2h1k2k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_20 > ../logs/eval/2h1k2k_1e4_256_20.txt

echo "Training model 2h1k2k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_25 > ../logs/train/2h1k2k_1e4_256_25.txt

echo "Evaluating model 2h1k2k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_25 > ../logs/eval/2h1k2k_1e4_256_25.txt

echo "Training model 2h1k2k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_35 > ../logs/train/2h1k2k_1e4_256_35.txt

echo "Evaluating model 2h1k2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_35 > ../logs/eval/2h1k2k_1e4_256_35.txt

echo "Training model 2h1k2k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_40 > ../logs/train/2h1k2k_1e4_256_40.txt

echo "Evaluating model 2h1k2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_40 > ../logs/eval/2h1k2k_1e4_256_40.txt

echo "Training model 2h1k5h_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_10 > ../logs/train/2h1k5h_1e4_256_10.txt

echo "Evaluating model 2h1k5h_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_10 > ../logs/eval/2h1k5h_1e4_256_10.txt

echo "Training model 2h1k5h_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_20 > ../logs/train/2h1k5h_1e4_256_20.txt

echo "Evaluating model 2h1k5h_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_20 > ../logs/eval/2h1k5h_1e4_256_20.txt

echo "Training model 2h1k5h_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_25 > ../logs/train/2h1k5h_1e4_256_25.txt

echo "Evaluating model 2h1k5h_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_25 > ../logs/eval/2h1k5h_1e4_256_25.txt

echo "Training model 2h1k5h_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_35 > ../logs/train/2h1k5h_1e4_256_35.txt

echo "Evaluating model 2h1k5h_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_35 > ../logs/eval/2h1k5h_1e4_256_35.txt

echo "Training model 2h1k5h_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_40 > ../logs/train/2h1k5h_1e4_256_40.txt

echo "Evaluating model 2h1k5h_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_40 > ../logs/eval/2h1k5h_1e4_256_40.txt

echo "Training model 2h5h6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_10 > ../logs/train/2h5h6k_1e4_256_10.txt

echo "Evaluating model 2h5h6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_10 > ../logs/eval/2h5h6k_1e4_256_10.txt

echo "Training model 2h5h6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_20 > ../logs/train/2h5h6k_1e4_256_20.txt

echo "Evaluating model 2h5h6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_20 > ../logs/eval/2h5h6k_1e4_256_20.txt

echo "Training model 2h5h6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_25 > ../logs/train/2h5h6k_1e4_256_25.txt

echo "Evaluating model 2h5h6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_25 > ../logs/eval/2h5h6k_1e4_256_25.txt

echo "Training model 2h5h6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_35 > ../logs/train/2h5h6k_1e4_256_35.txt

echo "Evaluating model 2h5h6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_35 > ../logs/eval/2h5h6k_1e4_256_35.txt

echo "Training model 2h5h6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_40 > ../logs/train/2h5h6k_1e4_256_40.txt

echo "Evaluating model 2h5h6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_40 > ../logs/eval/2h5h6k_1e4_256_40.txt

echo "Training model 2h5h5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_10 > ../logs/train/2h5h5k_1e4_256_10.txt

echo "Evaluating model 2h5h5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_10 > ../logs/eval/2h5h5k_1e4_256_10.txt

echo "Training model 2h5h5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_20 > ../logs/train/2h5h5k_1e4_256_20.txt

echo "Evaluating model 2h5h5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_20 > ../logs/eval/2h5h5k_1e4_256_20.txt

echo "Training model 2h5h5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_25 > ../logs/train/2h5h5k_1e4_256_25.txt

echo "Evaluating model 2h5h5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_25 > ../logs/eval/2h5h5k_1e4_256_25.txt

echo "Training model 2h5h5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_35 > ../logs/train/2h5h5k_1e4_256_35.txt

echo "Evaluating model 2h5h5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_35 > ../logs/eval/2h5h5k_1e4_256_35.txt

echo "Training model 2h5h5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_40 > ../logs/train/2h5h5k_1e4_256_40.txt

echo "Evaluating model 2h5h5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_40 > ../logs/eval/2h5h5k_1e4_256_40.txt

echo "Training model 2h5h4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_10 > ../logs/train/2h5h4k_1e4_256_10.txt

echo "Evaluating model 2h5h4k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_10 > ../logs/eval/2h5h4k_1e4_256_10.txt

echo "Training model 2h5h4k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_20 > ../logs/train/2h5h4k_1e4_256_20.txt

echo "Evaluating model 2h5h4k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_20 > ../logs/eval/2h5h4k_1e4_256_20.txt

echo "Training model 2h5h4k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_25 > ../logs/train/2h5h4k_1e4_256_25.txt

echo "Evaluating model 2h5h4k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_25 > ../logs/eval/2h5h4k_1e4_256_25.txt

echo "Training model 2h5h4k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_35 > ../logs/train/2h5h4k_1e4_256_35.txt

echo "Evaluating model 2h5h4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_35 > ../logs/eval/2h5h4k_1e4_256_35.txt

echo "Training model 2h5h4k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_40 > ../logs/train/2h5h4k_1e4_256_40.txt

echo "Evaluating model 2h5h4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_40 > ../logs/eval/2h5h4k_1e4_256_40.txt

echo "Training model 2h5h3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_10 > ../logs/train/2h5h3k_1e4_256_10.txt

echo "Evaluating model 2h5h3k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_10 > ../logs/eval/2h5h3k_1e4_256_10.txt

echo "Training model 2h5h3k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_20 > ../logs/train/2h5h3k_1e4_256_20.txt

echo "Evaluating model 2h5h3k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_20 > ../logs/eval/2h5h3k_1e4_256_20.txt

echo "Training model 2h5h3k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_25 > ../logs/train/2h5h3k_1e4_256_25.txt

echo "Evaluating model 2h5h3k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_25 > ../logs/eval/2h5h3k_1e4_256_25.txt

echo "Training model 2h5h3k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_35 > ../logs/train/2h5h3k_1e4_256_35.txt

echo "Evaluating model 2h5h3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_35 > ../logs/eval/2h5h3k_1e4_256_35.txt

echo "Training model 2h5h3k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_40 > ../logs/train/2h5h3k_1e4_256_40.txt

echo "Evaluating model 2h5h3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_40 > ../logs/eval/2h5h3k_1e4_256_40.txt

echo "Training model 2h5h2k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_10 > ../logs/train/2h5h2k_1e4_256_10.txt

echo "Evaluating model 2h5h2k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_10 > ../logs/eval/2h5h2k_1e4_256_10.txt

echo "Training model 2h5h2k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_20 > ../logs/train/2h5h2k_1e4_256_20.txt

echo "Evaluating model 2h5h2k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_20 > ../logs/eval/2h5h2k_1e4_256_20.txt

echo "Training model 2h5h2k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_25 > ../logs/train/2h5h2k_1e4_256_25.txt

echo "Evaluating model 2h5h2k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_25 > ../logs/eval/2h5h2k_1e4_256_25.txt

echo "Training model 2h5h2k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_35 > ../logs/train/2h5h2k_1e4_256_35.txt

echo "Evaluating model 2h5h2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_35 > ../logs/eval/2h5h2k_1e4_256_35.txt

echo "Training model 2h5h2k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_40 > ../logs/train/2h5h2k_1e4_256_40.txt

echo "Evaluating model 2h5h2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_40 > ../logs/eval/2h5h2k_1e4_256_40.txt

echo "Training model 2h5h1k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_10 > ../logs/train/2h5h1k_1e4_256_10.txt

echo "Evaluating model 2h5h1k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_10 > ../logs/eval/2h5h1k_1e4_256_10.txt

echo "Training model 2h5h1k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_20 > ../logs/train/2h5h1k_1e4_256_20.txt

echo "Evaluating model 2h5h1k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_20 > ../logs/eval/2h5h1k_1e4_256_20.txt

echo "Training model 2h5h1k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_25 > ../logs/train/2h5h1k_1e4_256_25.txt

echo "Evaluating model 2h5h1k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_25 > ../logs/eval/2h5h1k_1e4_256_25.txt

echo "Training model 2h5h1k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_35 > ../logs/train/2h5h1k_1e4_256_35.txt

echo "Evaluating model 2h5h1k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_35 > ../logs/eval/2h5h1k_1e4_256_35.txt

echo "Training model 2h5h1k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_40 > ../logs/train/2h5h1k_1e4_256_40.txt

echo "Evaluating model 2h5h1k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_40 > ../logs/eval/2h5h1k_1e4_256_40.txt

echo "Training model 2h1h6k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_10 > ../logs/train/2h1h6k_1e4_256_10.txt

echo "Evaluating model 2h1h6k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_10 > ../logs/eval/2h1h6k_1e4_256_10.txt

echo "Training model 2h1h6k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_20 > ../logs/train/2h1h6k_1e4_256_20.txt

echo "Evaluating model 2h1h6k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_20 > ../logs/eval/2h1h6k_1e4_256_20.txt

echo "Training model 2h1h6k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_25 > ../logs/train/2h1h6k_1e4_256_25.txt

echo "Evaluating model 2h1h6k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_25 > ../logs/eval/2h1h6k_1e4_256_25.txt

echo "Training model 2h1h6k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_35 > ../logs/train/2h1h6k_1e4_256_35.txt

echo "Evaluating model 2h1h6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_35 > ../logs/eval/2h1h6k_1e4_256_35.txt

echo "Training model 2h1h6k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_40 > ../logs/train/2h1h6k_1e4_256_40.txt

echo "Evaluating model 2h1h6k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_40 > ../logs/eval/2h1h6k_1e4_256_40.txt

echo "Training model 2h1h5k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_10 > ../logs/train/2h1h5k_1e4_256_10.txt

echo "Evaluating model 2h1h5k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_10 > ../logs/eval/2h1h5k_1e4_256_10.txt

echo "Training model 2h1h5k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_20 > ../logs/train/2h1h5k_1e4_256_20.txt

echo "Evaluating model 2h1h5k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_20 > ../logs/eval/2h1h5k_1e4_256_20.txt

echo "Training model 2h1h5k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_25 > ../logs/train/2h1h5k_1e4_256_25.txt

echo "Evaluating model 2h1h5k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_25 > ../logs/eval/2h1h5k_1e4_256_25.txt

echo "Training model 2h1h5k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_35 > ../logs/train/2h1h5k_1e4_256_35.txt

echo "Evaluating model 2h1h5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_35 > ../logs/eval/2h1h5k_1e4_256_35.txt

echo "Training model 2h1h5k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_40 > ../logs/train/2h1h5k_1e4_256_40.txt

echo "Evaluating model 2h1h5k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_40 > ../logs/eval/2h1h5k_1e4_256_40.txt

echo "Training model 2h1h4k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_10 > ../logs/train/2h1h4k_1e4_256_10.txt

echo "Evaluating model 2h1h4k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_10 > ../logs/eval/2h1h4k_1e4_256_10.txt

echo "Training model 2h1h4k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_20 > ../logs/train/2h1h4k_1e4_256_20.txt

echo "Evaluating model 2h1h4k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_20 > ../logs/eval/2h1h4k_1e4_256_20.txt

echo "Training model 2h1h4k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_25 > ../logs/train/2h1h4k_1e4_256_25.txt

echo "Evaluating model 2h1h4k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_25 > ../logs/eval/2h1h4k_1e4_256_25.txt

echo "Training model 2h1h4k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_35 > ../logs/train/2h1h4k_1e4_256_35.txt

echo "Evaluating model 2h1h4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_35 > ../logs/eval/2h1h4k_1e4_256_35.txt

echo "Training model 2h1h4k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_40 > ../logs/train/2h1h4k_1e4_256_40.txt

echo "Evaluating model 2h1h4k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_40 > ../logs/eval/2h1h4k_1e4_256_40.txt

echo "Training model 2h1h3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_10 > ../logs/train/2h1h3k_1e4_256_10.txt

echo "Evaluating model 2h1h3k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_10 > ../logs/eval/2h1h3k_1e4_256_10.txt

echo "Training model 2h1h3k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_20 > ../logs/train/2h1h3k_1e4_256_20.txt

echo "Evaluating model 2h1h3k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_20 > ../logs/eval/2h1h3k_1e4_256_20.txt

echo "Training model 2h1h3k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_25 > ../logs/train/2h1h3k_1e4_256_25.txt

echo "Evaluating model 2h1h3k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_25 > ../logs/eval/2h1h3k_1e4_256_25.txt

echo "Training model 2h1h3k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_35 > ../logs/train/2h1h3k_1e4_256_35.txt

echo "Evaluating model 2h1h3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_35 > ../logs/eval/2h1h3k_1e4_256_35.txt

echo "Training model 2h1h3k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_40 > ../logs/train/2h1h3k_1e4_256_40.txt

echo "Evaluating model 2h1h3k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_40 > ../logs/eval/2h1h3k_1e4_256_40.txt

echo "Training model 2h1h2k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_10 > ../logs/train/2h1h2k_1e4_256_10.txt

echo "Evaluating model 2h1h2k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_10 > ../logs/eval/2h1h2k_1e4_256_10.txt

echo "Training model 2h1h2k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_20 > ../logs/train/2h1h2k_1e4_256_20.txt

echo "Evaluating model 2h1h2k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_20 > ../logs/eval/2h1h2k_1e4_256_20.txt

echo "Training model 2h1h2k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_25 > ../logs/train/2h1h2k_1e4_256_25.txt

echo "Evaluating model 2h1h2k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_25 > ../logs/eval/2h1h2k_1e4_256_25.txt

echo "Training model 2h1h2k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_35 > ../logs/train/2h1h2k_1e4_256_35.txt

echo "Evaluating model 2h1h2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_35 > ../logs/eval/2h1h2k_1e4_256_35.txt

echo "Training model 2h1h2k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_40 > ../logs/train/2h1h2k_1e4_256_40.txt

echo "Evaluating model 2h1h2k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_40 > ../logs/eval/2h1h2k_1e4_256_40.txt

echo "Training model 2h1h1k_10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 10 \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_10 > ../logs/train/2h1h1k_1e4_256_10.txt

echo "Evaluating model 2h1h1k_1e4_256_10"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_10 > ../logs/eval/2h1h1k_1e4_256_10.txt

echo "Training model 2h1h1k_20, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 20 \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_20 > ../logs/train/2h1h1k_1e4_256_20.txt

echo "Evaluating model 2h1h1k_1e4_256_20"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_20 > ../logs/eval/2h1h1k_1e4_256_20.txt

echo "Training model 2h1h1k_25, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 25 \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_25 > ../logs/train/2h1h1k_1e4_256_25.txt

echo "Evaluating model 2h1h1k_1e4_256_25"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_25 > ../logs/eval/2h1h1k_1e4_256_25.txt

echo "Training model 2h1h1k_35, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 35 \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_35 > ../logs/train/2h1h1k_1e4_256_35.txt

echo "Evaluating model 2h1h1k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_35 > ../logs/eval/2h1h1k_1e4_256_35.txt

echo "Training model 2h1h1k_40, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 --patience 40 \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_40 > ../logs/train/2h1h1k_1e4_256_40.txt

echo "Evaluating model 2h1h1k_1e4_256_35"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_40 > ../logs/eval/2h1h1k_1e4_256_40.txt
