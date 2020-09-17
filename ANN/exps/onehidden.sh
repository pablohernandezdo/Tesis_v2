#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Training model 1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h6k --model_name 1h6k_1e4_256 > ../logs/train/1h6k_1e4_256.txt

echo "Evaluating model 1h6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h6k --model_name 1h6k_1e4_256 > ../logs/eval/1h6k_1e4_256.txt

echo "Training model 1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h5k --model_name 1h5k_1e4_256 > ../logs/train/1h5k_1e4_256.txt

echo "Evaluating model 1h5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h5k --model_name 1h5k_1e4_256 > ../logs/eval/1h5k_1e4_256.txt

echo "Training model 1h4k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h4k --model_name 1h4k_1e4_256 > ../logs/train/1h4k_1e4_256.txt

echo "Evaluating model 1h4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h4k --model_name 1h4k_1e4_256 > ../logs/eval/1h4k_1e4_256.txt

echo "Training model 1h3k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h3k --model_name 1h3k_1e4_256 > ../logs/train/1h3k_1e4_256.txt

echo "Evaluating model 1h3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h3k --model_name 1h3k_1e4_256 > ../logs/eval/1h3k_1e4_256.txt

echo "Training model 1h2k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h2k --model_name 1h2k_1e4_256 > ../logs/train/1h2k_1e4_256.txt

echo "Evaluating model 1h2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h2k --model_name 1h2k_1e4_256 > ../logs/eval/1h2k_1e4_256.txt

echo "Training model 1h1k, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h1k --model_name 1h1k_1e4_256 > ../logs/train/1h1k_1e4_256.txt

echo "Evaluating model 1h1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h1k --model_name 1h1k_1e4_256 > ../logs/eval/1h1k_1e4_256.txt

echo "Training model 1h5h, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h5h --model_name 1h5h_1e4_256 > ../logs/train/1h5h_1e4_256.txt

echo "Evaluating model 1h5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h5h --model_name 1h5h_1e4_256 > ../logs/eval/1h5h_1e4_256.txt

echo "Training model 1h1h, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h1h --model_name 1h1h_1e4_256 > ../logs/train/1h1h_1e4_256.txt

echo "Evaluating model 1h1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h1h --model_name 1h1h_1e4_256 > ../logs/eval/1h1h_1e4_256.txt

echo "Training model 1h10, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h10 --model_name 1h10_1e4_256 > ../logs/train/1h10_1e4_256.txt

echo "Evaluating model 1h10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h10 --model_name 1h10_1e4_256 > ../logs/eval/1h10_1e4_256.txt

echo "Training model 1h1, lr = 1e-4, epochs = 5, batch_size = 256"
python ../train_validation.py \
              --train_path $trn --val_path $val      \
              --n_epochs 5 --lr 1e-4 --batch_size 256 \
              --classifier 1h1 --model_name 1h1_1e4_256 > ../logs/train/1h1_1e4_256.txt

echo "Evaluating model 1h1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --classifier 1h1 --model_name 1h1_1e4_256 > ../logs/eval/1h1_1e4_256.txt
